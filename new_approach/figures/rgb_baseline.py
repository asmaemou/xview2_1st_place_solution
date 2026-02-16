#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZERO-ARGS:
    python make_panel_c_RGB_baseline_pipeline.py

Outputs (RGB-only baseline, END-TO-END):
  - panel_c_rgb_baseline_pred_mask.png
  - panel_c_rgb_baseline_pred_overlay.png

What it uses
------------
Stage-1 localization (single ckpt + optional Platt):
  - Auto-finds stage-1 loc checkpoint in STAGE1_DIR_REL
  - p(building) = sigmoid(a*logit + b)
  - build_mask = p >= LOC_THRESH

Stage-2 damage (single stage-2 best ckpt + vector scaling calib):
  - Auto-finds stage-2 best checkpoint in STAGE2_DIR_REL
  - Loads vector scaling calib .npz for that ckpt
  - Calibrated probs over classes 1..4
  - damage_pred = argmax(prob) + 1

Final prediction:
  pred_final = 0 where not building, else damage_pred in {1..4}

Notes
-----
- This matches your RGB-only baseline pipeline:
    input x6 = concat([pre_RGB, post_RGB]) (normalized ImageNet)
    stage-1 uses ONLY pre (x6[:,0:3]) for localization gating
- Saves a colored mask and an overlay on the POST image.
"""

import sys
from pathlib import Path
from os import path
import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# =========================
# USER SETTINGS
# =========================
HERE = Path(__file__).resolve().parent

# pick one tile (basename pattern used in your codebase)
# Example: "AOI1-tile_1-3" (no _pre/_post suffix)
TILE_ID = "AOI1-tile_1-3"

# dataset folders (relative)
IMG_DIR_REL = r"idabd/images"

# stage-1 and stage-2 output folders (relative)
STAGE1_DIR_REL = r"idabd_stage1_loc_ft_checkpoints"
STAGE2_DIR_REL = r"idabd_stage2_damage_ft_checkpoints_RGB_ONLY_BASELINE"

# thresholds
LOC_THRESH = 0.50

# outputs
OUT_MASK    = "panel_c_rgb_baseline_pred_mask.png"
OUT_OVERLAY = "panel_c_rgb_baseline_pred_overlay.png"
# =========================


# --- make sure zoo import works ---
for cand in [HERE, HERE.parent, HERE.parent.parent]:
    if (cand / "zoo").is_dir():
        sys.path.insert(0, str(cand))
        break


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =========================
# Utilities: find files
# =========================
def find_pre_post_paths(img_dir: Path, tile_id: str):
    # try common extensions
    exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    for ext in exts:
        pre  = img_dir / f"{tile_id}_pre_disaster{ext}"
        post = img_dir / f"{tile_id}_post_disaster{ext}"
        if pre.exists() and post.exists():
            return pre, post
    raise FileNotFoundError(f"Could not find pre/post for tile_id='{tile_id}' in {img_dir}")

def find_stage1_ckpt(stage1_dir: Path) -> Path:
    c1 = sorted(stage1_dir.glob("*_idabd_ft_best.pth"))
    c2 = sorted(stage1_dir.glob("*_idabd_ft.best.pth"))
    allc = c1 + c2
    if not allc:
        raise FileNotFoundError(f"No stage1 ckpts found in {stage1_dir}")
    return allc[0]

def find_platt(stage1_dir: Path, ckpt: Path):
    stem = ckpt.stem
    for suf in ["_idabd_ft.best", "_idabd_ft_best", "_idabd_ft.last", "_idabd_ft_last"]:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    cand = stage1_dir / f"{stem}_idabd_platt.npz"
    if cand.exists():
        d = np.load(str(cand))
        return float(d["a"]), float(d["b"]), cand.name
    return 1.0, 0.0, None

def find_stage2_best_ckpt(stage2_dir: Path) -> Path:
    ckpts = sorted(stage2_dir.glob("*_idabd_stage2_ft_best.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No stage2 best ckpts found in {stage2_dir} (pattern '*_idabd_stage2_ft_best.pth')")
    return ckpts[0]

def find_stage2_calib(stage2_dir: Path, ckpt: Path) -> Path:
    # expected: <stem>_idabd_stage2_calib_vector_scaling.npz
    noext = ckpt.stem
    for suf in ["_idabd_stage2_ft_best", "_idabd_stage2_ft_last"]:
        if noext.endswith(suf):
            stem = noext[: -len(suf)]
            break
    else:
        stem = noext

    cand = stage2_dir / f"{stem}_idabd_stage2_calib_vector_scaling.npz"
    if cand.exists():
        return cand

    # fallback: try a loose glob
    g = sorted(stage2_dir.glob(f"{stem}*_calib*vector*scaling*.npz"))
    if g:
        return g[0]

    raise FileNotFoundError(
        f"Calibration npz not found for stage2 ckpt:\n"
        f"  ckpt={ckpt}\n"
        f"  expected={cand}\n"
        f"  stage2_dir={stage2_dir}"
    )


# =========================
# Preprocess
# =========================
def pad_to_factor(img, factor=32):
    h, w = img.shape[:2]
    new_h = int(np.ceil(h / factor) * factor)
    new_w = int(np.ceil(w / factor) * factor)
    pad_h = new_h - h
    pad_w = new_w - w
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT_101)
    return padded, (pad_h, pad_w)

def unpad2d(arr2d, pad_hw):
    ph, pw = pad_hw
    if ph == 0 and pw == 0:
        return arr2d
    return arr2d[: arr2d.shape[0] - ph, : arr2d.shape[1] - pw]

def preprocess_rgb(img_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))  # CHW
    return torch.from_numpy(img).float()

def preprocess_6ch(pre_bgr: np.ndarray, post_bgr: np.ndarray) -> torch.Tensor:
    pre = preprocess_rgb(pre_bgr)
    post = preprocess_rgb(post_bgr)
    return torch.cat([pre, post], dim=0)  # 6xHxW


# =========================
# Checkpoint loading
# =========================
def strip_prefixes(sd: dict):
    prefixes = ["module.", "model.", "net."]
    out = {}
    for k, v in sd.items():
        kk = k
        for pfx in prefixes:
            if kk.startswith(pfx):
                kk = kk[len(pfx):]
        out[kk] = v
    return out

def load_state_dict_any(weight_path: str):
    try:
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(weight_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    if not isinstance(sd, dict):
        raise ValueError(f"Checkpoint not a state_dict mapping: {weight_path}")
    return strip_prefixes(sd)

def detect_backbone_from_filename(fname_lower: str) -> str:
    for key in ["dpn92", "res34", "res50", "se154"]:
        if key in fname_lower:
            return key
    return ""


# =========================
# Build models from weights
# =========================
def build_loc_model_from_weight(weight_path: str):
    import zoo.models as zm
    fname = path.basename(weight_path).lower()
    bb = detect_backbone_from_filename(fname)
    if bb == "dpn92":
        model = zm.Dpn92_Unet_Loc(pretrained=None)
    elif bb == "res34":
        model = zm.Res34_Unet_Loc(pretrained=False)
    elif bb == "res50":
        model = zm.SeResNext50_Unet_Loc(pretrained=None)
    elif bb == "se154":
        model = zm.SeNet154_Unet_Loc(pretrained=None)
    else:
        # fallback to your original strict prefix behavior
        if fname.startswith("dpn92"):
            model = zm.Dpn92_Unet_Loc(pretrained=None)
        elif fname.startswith("res34"):
            model = zm.Res34_Unet_Loc(pretrained=False)
        elif fname.startswith("res50"):
            model = zm.SeResNext50_Unet_Loc(pretrained=None)
        elif fname.startswith("se154"):
            model = zm.SeNet154_Unet_Loc(pretrained=None)
        else:
            raise ValueError(f"Unrecognized localization weight prefix: {fname}")

    sd = load_state_dict_any(weight_path)
    model.load_state_dict(sd, strict=True)
    return model

def build_damage_model_from_weight(weight_path: str):
    import zoo.models as zm
    fname = path.basename(weight_path).lower()
    bb = detect_backbone_from_filename(fname)
    if bb == "dpn92":
        model = zm.Dpn92_Unet_Double(pretrained=None)
    elif bb == "res34":
        model = zm.Res34_Unet_Double(pretrained=False)
    elif bb == "res50":
        model = zm.SeResNext50_Unet_Double(pretrained=None)
    elif bb == "se154":
        model = zm.SeNet154_Unet_Double(pretrained=None)
    else:
        # fallback to your original strict prefix behavior
        if fname.startswith("dpn92"):
            model = zm.Dpn92_Unet_Double(pretrained=None)
        elif fname.startswith("res34"):
            model = zm.Res34_Unet_Double(pretrained=False)
        elif fname.startswith("res50"):
            model = zm.SeResNext50_Unet_Double(pretrained=None)
        elif fname.startswith("se154"):
            model = zm.SeNet154_Unet_Double(pretrained=None)
        else:
            raise ValueError(f"Unrecognized cls weight prefix: {fname}")

    sd = load_state_dict_any(weight_path)
    model.load_state_dict(sd, strict=True)
    return model


# =========================
# Inference helpers
# =========================
@torch.no_grad()
def predict_build_mask_from_pre(loc_model, pre_bgr, device, a, b, thresh):
    pre_bgr, pad_hw = pad_to_factor(pre_bgr, 32)
    x = preprocess_rgb(pre_bgr).unsqueeze(0).to(device)

    logit = loc_model(x)
    if isinstance(logit, (tuple, list)):
        logit = logit[0]
    if logit.ndim == 3:
        logit = logit.unsqueeze(1)
    if logit.ndim == 4 and logit.shape[1] > 1:
        logit = logit[:, 0:1, :, :]

    logit = a * logit + b
    prob = torch.sigmoid(logit)[0, 0].detach().cpu().numpy()
    prob = unpad2d(prob, pad_hw)
    return (prob >= thresh)

@torch.no_grad()
def predict_damage_calibrated(stage2_model, x6, device, W, b):
    """
    stage2 logits are Bx5xHxW; use channels 1..4.
    Calibrate with vector scaling in FP32, then softmax.
    Returns:
      pred (HxW) in {1..4}, probs (4,H,W) float32
    """
    x6 = x6.to(device)
    logits = stage2_model(x6)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    z = logits[:, 1:5, :, :]  # Bx4xHxW

    # calibration in FP32 (AMP-safe)
    with torch.cuda.amp.autocast(enabled=False):
        z = z.float()
        Wf = W.float()
        bf = b.float()
        z2 = z.permute(0, 2, 3, 1).contiguous()     # BxHxWx4
        z2 = torch.matmul(z2, Wf.t()) + bf          # BxHxWx4
        z2 = z2.permute(0, 3, 1, 2).contiguous()    # Bx4xHxW
        probs = F.softmax(z2, dim=1)                # Bx4xHxW

    pred = torch.argmax(probs, dim=1) + 1           # BxHxW (1..4)
    return pred[0].detach().cpu().numpy().astype(np.uint8), probs[0].detach().cpu().numpy().astype(np.float32)

def make_color_mask(pred_final_0to4: np.ndarray) -> np.ndarray:
    """
    Color map:
      0 background: black
      1 no-damage:  green
      2 minor:      yellow
      3 major:      orange
      4 destroyed:  red
    """
    h, w = pred_final_0to4.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    # BGR colors for OpenCV composition
    colors = {
        0: (0, 0, 0),
        1: (0, 255, 0),
        2: (0, 255, 255),
        3: (0, 165, 255),
        4: (0, 0, 255),
    }
    for k, c in colors.items():
        vis[pred_final_0to4 == k] = c
    return vis

def overlay_on_post(post_bgr: np.ndarray, color_mask_bgr: np.ndarray, alpha=0.45) -> np.ndarray:
    """
    Blend ONLY where mask is non-zero (any channel > 0).
    Safe shape handling: uses a 2D boolean mask.
    """
    m = (color_mask_bgr.sum(axis=2) > 0)  # HxW boolean

    blended = cv2.addWeighted(post_bgr, 1 - alpha, color_mask_bgr, alpha, 0)
    out = post_bgr.copy()
    out[m] = blended[m]  # m is HxW -> selects Nx3 rows, OK
    return out


def save_panel(img_bgr: np.ndarray, out_path: str, caption: str, dpi=300):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(3.2, 5.2), dpi=dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[10, 4], hspace=0.05)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_txt = fig.add_subplot(gs[1, 0])

    ax_img.imshow(img_rgb)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_frame_on(True)

    ax_txt.axis("off")
    ax_txt.text(0.02, 0.95, caption, ha="left", va="top", fontsize=14)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved:", out_path)


def main():
    img_dir = HERE / IMG_DIR_REL
    stage1_dir = HERE / STAGE1_DIR_REL
    stage2_dir = HERE / STAGE2_DIR_REL

    if not img_dir.exists():
        raise FileNotFoundError(img_dir)
    if not stage1_dir.exists():
        raise FileNotFoundError(stage1_dir)
    if not stage2_dir.exists():
        raise FileNotFoundError(stage2_dir)

    pre_path, post_path = find_pre_post_paths(img_dir, TILE_ID)

    # --- load stage-1 ---
    loc_ckpt = find_stage1_ckpt(stage1_dir)
    loc_a, loc_b, platt_name = find_platt(stage1_dir, loc_ckpt)

    # --- load stage-2 ---
    s2_ckpt = find_stage2_best_ckpt(stage2_dir)
    s2_calib = find_stage2_calib(stage2_dir, s2_ckpt)

    dcal = np.load(str(s2_calib))
    W = torch.from_numpy(dcal["W"].astype(np.float32))
    b = torch.from_numpy(dcal["b"].astype(np.float32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loc_model = build_loc_model_from_weight(str(loc_ckpt)).to(device).eval()
    stage2_model = build_damage_model_from_weight(str(s2_ckpt)).to(device).eval()

    for p in loc_model.parameters():
        p.requires_grad = False
    for p in stage2_model.parameters():
        p.requires_grad = False

    W = W.to(device)
    b = b.to(device)

    pre_bgr = cv2.imread(str(pre_path), cv2.IMREAD_COLOR)
    post_bgr = cv2.imread(str(post_path), cv2.IMREAD_COLOR)
    if pre_bgr is None:
        raise FileNotFoundError(pre_path)
    if post_bgr is None:
        raise FileNotFoundError(post_path)

    # pad both so x6 matches model stride, then unpad back for outputs
    pre_pad, pad_hw = pad_to_factor(pre_bgr, 32)
    post_pad, _ = pad_to_factor(post_bgr, 32)

    x6 = preprocess_6ch(pre_pad, post_pad).unsqueeze(0)  # 1x6xH'xW'

    print("[TILE ]", TILE_ID)
    print("[LOC  ]", loc_ckpt.name)
    print("[PLATT]", platt_name if platt_name else "(none) a=1 b=0")
    print("[LTHR ]", LOC_THRESH)
    print("[S2   ]", s2_ckpt.name)
    print("[CALIB]", s2_calib.name)

    # stage-1 gate (on PRE)
    build_mask = predict_build_mask_from_pre(loc_model, pre_bgr, device, loc_a, loc_b, LOC_THRESH)  # HxW (unpadded)
    # stage-2 damage (on padded x6)
    pred_damage_pad, _ = predict_damage_calibrated(stage2_model, x6, device, W, b)  # H'xW'
    pred_damage = unpad2d(pred_damage_pad, pad_hw)  # HxW

    # final 0..4
    pred_final = np.zeros_like(pred_damage, dtype=np.uint8)
    pred_final[build_mask] = pred_damage[build_mask]

    # render
    color_mask = make_color_mask(pred_final)  # BGR
    overlay = overlay_on_post(post_bgr, color_mask, alpha=0.45)

    save_panel(
        color_mask,
        OUT_MASK,
        "(c) RGB-only baseline\nEND-TO-END prediction\n(mask: 0..4)\nStage-1 loc gate +\nStage-2 calibrated damage"
    )
    save_panel(
        overlay,
        OUT_OVERLAY,
        "(c) RGB-only baseline\nEND-TO-END prediction\n(overlay on post)\nStage-1 loc gate +\nStage-2 calibrated damage"
    )


if __name__ == "__main__":
    main()
