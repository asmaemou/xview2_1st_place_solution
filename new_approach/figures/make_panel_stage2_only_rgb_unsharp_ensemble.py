#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZERO-ARGS:
    python make_panel_stage2_only_RGB_UNSHARP_ensemble.py

Outputs:
  - panel_stage2_RGB_UNSHARP_only_mask.png
  - panel_stage2_RGB_UNSHARP_only_overlay_post.png

Stage-2 ONLY (no Stage-1 gating):
  x6 = concat([pre_RGB, post_RGB]) -> each stage2 model -> vector scaling -> softmax
  ensemble by averaging probs across models -> argmax -> {1..4}

Folder used:
  idabd_stage2_damage_ft_checkpoints_RGB_UNSHARP_ONLY
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

# You can set this to "AOI1-tile_1-3" (recommended)
# If you accidentally pass "..._pre_disaster" or "..._post_disaster", it will normalize.
TILE_ID = "AOI1-tile_1-3"

IMG_DIR_REL   = r"idabd/images"
STAGE2_DIR_REL = r"idabd_stage2_damage_ft_checkpoints_RGB_UNSHARP_ONLY"

STAGE2_GLOB = "*_idabd_stage2_ft_best.pth"
STAGE2_MAX_MODELS = 0   # 0 = no cap (use all)

OUT_MASK    = "panel_stage2_RGB_UNSHARP_only_mask.png"
OUT_OVERLAY = "panel_stage2_RGB_UNSHARP_only_overlay_post.png"
# =========================


# --- make sure zoo import works ---
for cand in [HERE, HERE.parent, HERE.parent.parent]:
    if (cand / "zoo").is_dir():
        sys.path.insert(0, str(cand))
        break


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_tile_id(s: str) -> str:
    s = Path(s).stem
    s = s.replace("_pre_disaster", "").replace("_post_disaster", "")
    return s


def find_pre_post_paths(img_dir: Path, tile_id: str):
    exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    for ext in exts:
        pre  = img_dir / f"{tile_id}_pre_disaster{ext}"
        post = img_dir / f"{tile_id}_post_disaster{ext}"
        if pre.exists() and post.exists():
            return pre, post
    raise FileNotFoundError(f"Could not find pre/post for tile_id='{tile_id}' in {img_dir}")


# ---------- pad / preprocess ----------
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


# ---------- ckpt helpers ----------
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
        raise ValueError(f"Unrecognized stage-2 backbone in filename: {fname}")

    sd = load_state_dict_any(weight_path)
    model.load_state_dict(sd, strict=True)
    return model


def strip_known_suffix(name_noext: str, suffixes):
    for suf in suffixes:
        if name_noext.endswith(suf):
            return name_noext[:-len(suf)]
    return name_noext


def find_stage2_calib(stage2_dir: Path, ckpt_path: str) -> Path:
    """
    Expected:
      ckpt:  <stem>_idabd_stage2_ft_best.pth
      calib: <stem>_idabd_stage2_calib_vector_scaling.npz
    """
    base = path.basename(ckpt_path)
    noext = path.splitext(base)[0]
    stem = strip_known_suffix(noext, ["_idabd_stage2_ft_best", "_idabd_stage2_ft_last"])

    cand = stage2_dir / f"{stem}_idabd_stage2_calib_vector_scaling.npz"
    if cand.exists():
        return cand

    # fallback loose glob
    g = sorted(stage2_dir.glob(f"{stem}*_calib*vector*scaling*.npz"))
    if g:
        return g[0]

    raise FileNotFoundError(f"Missing calibration npz for: {ckpt_path}")


def load_stage2_pack(stage2_dir: Path, device, stage2_glob: str, max_models: int):
    ckpts = sorted(glob.glob(str(stage2_dir / stage2_glob)))
    if not ckpts:
        raise FileNotFoundError(f"No Stage-2 checkpoints found in {stage2_dir} with glob='{stage2_glob}'")

    if max_models and max_models > 0 and len(ckpts) > max_models:
        print(f"[STAGE2] WARNING: found {len(ckpts)} models; keeping first {max_models} (sorted).")
        ckpts = ckpts[:max_models]

    pack = []
    for ck in ckpts:
        calib_path = find_stage2_calib(stage2_dir, ck)
        d = np.load(str(calib_path))

        W = torch.from_numpy(d["W"].astype(np.float32)).to(device)
        b = torch.from_numpy(d["b"].astype(np.float32)).to(device)

        m = build_damage_model_from_weight(ck).to(device).eval()
        for p in m.parameters():
            p.requires_grad = False

        pack.append((m, W, b))

    return pack, ckpts


@torch.no_grad()
def predict_damage_ensemble_cal(stage2_pack, x6, device):
    """
    Returns pred (HxW) in {1..4} (on padded size).
    """
    prob_sum = None

    for (m, W, b) in stage2_pack:
        logits = m(x6)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        z = logits[:, 1:5, :, :]  # Bx4xHxW

        # vector scaling in FP32 (safe, no deprecated autocast)
        z = z.float()
        z2 = z.permute(0, 2, 3, 1).contiguous()      # BxHxWx4
        z2 = torch.matmul(z2, W.float().t()) + b.float()
        z2 = z2.permute(0, 3, 1, 2).contiguous()     # Bx4xHxW
        probs = F.softmax(z2, dim=1)                 # Bx4xHxW

        prob_sum = probs if prob_sum is None else (prob_sum + probs)

    prob_mean = prob_sum / float(len(stage2_pack))
    pred = torch.argmax(prob_mean, dim=1) + 1        # 1..4
    return pred[0].detach().cpu().numpy().astype(np.uint8)


def make_color_mask_damage_1to4(pred_1to4: np.ndarray) -> np.ndarray:
    """
    BGR colormap:
      1 no-damage:  green
      2 minor:      yellow
      3 major:      orange
      4 destroyed:  red
    """
    h, w = pred_1to4.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    colors = {
        1: (0, 255, 0),
        2: (0, 255, 255),
        3: (0, 165, 255),
        4: (0, 0, 255),
    }
    for k, c in colors.items():
        vis[pred_1to4 == k] = c
    return vis


def overlay_color_on_post(post_bgr: np.ndarray, color_bgr: np.ndarray, alpha=0.45) -> np.ndarray:
    m = (color_bgr.sum(axis=2) > 0)  # HxW
    blended = cv2.addWeighted(post_bgr, 1 - alpha, color_bgr, alpha, 0)
    out = post_bgr.copy()
    out[m] = blended[m]
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
    tile = normalize_tile_id(TILE_ID)

    img_dir = HERE / IMG_DIR_REL
    stage2_dir = HERE / STAGE2_DIR_REL

    if not img_dir.is_dir():
        raise FileNotFoundError(img_dir)
    if not stage2_dir.is_dir():
        raise FileNotFoundError(stage2_dir)

    pre_path, post_path = find_pre_post_paths(img_dir, tile)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    stage2_pack, ckpts_used = load_stage2_pack(
        stage2_dir, device=device,
        stage2_glob=STAGE2_GLOB,
        max_models=STAGE2_MAX_MODELS
    )
    print(f"[TILE ] {tile}")
    print(f"[S2   ] models loaded: {len(ckpts_used)}")
    for c in ckpts_used:
        print("       -", Path(c).name)

    pre_bgr = cv2.imread(str(pre_path), cv2.IMREAD_COLOR)
    post_bgr = cv2.imread(str(post_path), cv2.IMREAD_COLOR)
    if pre_bgr is None:
        raise FileNotFoundError(pre_path)
    if post_bgr is None:
        raise FileNotFoundError(post_path)

    pre_pad, pad_hw = pad_to_factor(pre_bgr, 32)
    post_pad, _ = pad_to_factor(post_bgr, 32)
    x6 = preprocess_6ch(pre_pad, post_pad).unsqueeze(0).to(device)

    pred_pad = predict_damage_ensemble_cal(stage2_pack, x6, device=device)  # 1..4 on padded
    pred = unpad2d(pred_pad, pad_hw)

    color = make_color_mask_damage_1to4(pred)
    overlay = overlay_color_on_post(post_bgr, color, alpha=0.45)

    save_panel(
        color, OUT_MASK,
        "(Stage-2) RGB_UNSHARP_ONLY\nCalibrated ENSEMBLE damage (1..4)\nPRE+POST input — no Stage-1 gating"
    )
    save_panel(
        overlay, OUT_OVERLAY,
        "(Stage-2) RGB_UNSHARP_ONLY\nOverlay on POST\nCalibrated ENSEMBLE — no Stage-1 gating"
    )


if __name__ == "__main__":
    main()
