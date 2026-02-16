#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZERO-ARGS:
    python make_mask_only_stage2_rgb_contrast_edge_ensemble.py

Outputs (NO caption, NO borders):
  - mask_only_RGB_CONTRAST_EDGE_ONLY.png
  - overlay_only_RGB_CONTRAST_EDGE_ONLY.png   (optional)

Stage-2 ONLY (no Stage-1 gating):
  x6 = concat([pre_RGB, post_RGB]) -> each stage2 model -> vector scaling -> softmax
  ensemble by averaging probs across models -> argmax -> {1..4}

Folder used:
  idabd_stage2_damage_ft_checkpoints_RGB_CONTRAST_EDGE_ONLY
"""

import sys
from pathlib import Path
from os import path
import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F


# =========================
# USER SETTINGS
# =========================
HERE = Path(__file__).resolve().parent

# Use base tile id like "AOI1-tile_1-3"
# If you accidentally pass "..._pre_disaster" / "..._post_disaster" or a filename, it will normalize.
TILE_ID = "AOI1-tile_1-3"

IMG_DIR_REL     = r"idabd/images"
STAGE2_DIR_REL  = r"idabd_stage2_damage_ft_checkpoints_RGB_CONTRAST_EDGE_ONLY"

STAGE2_GLOB = "*_idabd_stage2_ft_best.pth"
STAGE2_MAX_MODELS = 0   # 0 = no cap (use all)

# Save overlay too?
SAVE_OVERLAY = True
OVERLAY_ALPHA = 0.45

OUT_MASK_ONLY    = "mask_only_RGB_CONTRAST_EDGE_ONLY.png"
OUT_OVERLAY_ONLY = "overlay_only_RGB_CONTRAST_EDGE_ONLY.png"
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
def predict_damage_ensemble_cal(stage2_pack, x6):
    """
    Returns pred (HxW) in {1..4} (on padded size).
    """
    prob_sum = None

    for (m, W, b) in stage2_pack:
        logits = m(x6)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        z = logits[:, 1:5, :, :]  # Bx4xHxW

        # vector scaling + softmax in FP32
        z = z.float()
        z2 = z.permute(0, 2, 3, 1).contiguous()         # BxHxWx4
        z2 = torch.matmul(z2, W.float().t()) + b.float()
        z2 = z2.permute(0, 3, 1, 2).contiguous()        # Bx4xHxW
        probs = F.softmax(z2, dim=1)                    # Bx4xHxW

        prob_sum = probs if prob_sum is None else (prob_sum + probs)

    prob_mean = prob_sum / float(len(stage2_pack))
    pred = torch.argmax(prob_mean, dim=1) + 1           # 1..4
    return pred[0].detach().cpu().numpy().astype(np.uint8)


def make_color_mask_damage_1to4(pred_1to4: np.ndarray) -> np.ndarray:
    """
    BGR colormap:
      1 no-damage:  green
      2 minor:      yellow
      3 major:      orange
      4 destroyed:  red
    """
    vis = np.zeros((pred_1to4.shape[0], pred_1to4.shape[1], 3), dtype=np.uint8)
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
    """
    Overlay only where mask is non-zero.
    """
    m = (color_bgr.sum(axis=2) > 0)  # HxW
    blended = cv2.addWeighted(post_bgr, 1 - alpha, color_bgr, alpha, 0)
    out = post_bgr.copy()
    out[m] = blended[m]
    return out


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

    pred_pad = predict_damage_ensemble_cal(stage2_pack, x6)  # 1..4 (padded)
    pred = unpad2d(pred_pad, pad_hw)                         # 1..4 (original)

    # mask-only output (NO caption, NO border)
    color_mask = make_color_mask_damage_1to4(pred)
    cv2.imwrite(OUT_MASK_ONLY, color_mask)
    print("[OK] Saved mask-only:", OUT_MASK_ONLY)

    # optional overlay-only output
    if SAVE_OVERLAY:
        overlay = overlay_color_on_post(post_bgr, color_mask, alpha=OVERLAY_ALPHA)
        cv2.imwrite(OUT_OVERLAY_ONLY, overlay)
        print("[OK] Saved overlay-only:", OUT_OVERLAY_ONLY)


if __name__ == "__main__":
    main()
