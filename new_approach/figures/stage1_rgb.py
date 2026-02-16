#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZERO-ARGS:
    python make_panel_stage1_only_rgb.py

Outputs:
  - panel_stage1_build_mask.png
  - panel_stage1_build_overlay_pre.png

Stage-1 localization:
  PRE image -> loc_model -> (optional Platt) -> sigmoid -> building mask
"""

import sys
from pathlib import Path
from os import path
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt


# =========================
# USER SETTINGS
# =========================
HERE = Path(__file__).resolve().parent
TILE_ID = "AOI1-tile_1-3"

IMG_DIR_REL    = r"idabd/images"
STAGE1_DIR_REL  = r"idabd_stage1_loc_ft_checkpoints"
LOC_THRESH      = 0.50

OUT_MASK    = "panel_stage1_build_mask.png"
OUT_OVERLAY = "panel_stage1_build_overlay_pre.png"
# =========================


# --- make sure zoo import works ---
for cand in [HERE, HERE.parent, HERE.parent.parent]:
    if (cand / "zoo").is_dir():
        sys.path.insert(0, str(cand))
        break


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).float()


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
        raise ValueError(f"Unrecognized stage-1 backbone in filename: {fname}")
    sd = load_state_dict_any(weight_path)
    model.load_state_dict(sd, strict=True)
    return model

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


@torch.no_grad()
def predict_build_prob_and_mask(loc_model, pre_bgr, device, a, b, thresh):
    pre_pad, pad_hw = pad_to_factor(pre_bgr, 32)
    x = preprocess_rgb(pre_pad).unsqueeze(0).to(device)

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

    mask = (prob >= thresh)
    return prob, mask


def overlay_mask_on_img(img_bgr: np.ndarray, mask: np.ndarray, color_bgr=(0, 255, 0), alpha=0.45):
    color = np.zeros_like(img_bgr, dtype=np.uint8)
    color[mask] = color_bgr
    blended = cv2.addWeighted(img_bgr, 1 - alpha, color, alpha, 0)
    out = img_bgr.copy()
    out[mask] = blended[mask]
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

    pre_path, _ = find_pre_post_paths(img_dir, TILE_ID)

    loc_ckpt = find_stage1_ckpt(stage1_dir)
    a, b, platt_name = find_platt(stage1_dir, loc_ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loc_model = build_loc_model_from_weight(str(loc_ckpt)).to(device).eval()

    pre_bgr = cv2.imread(str(pre_path), cv2.IMREAD_COLOR)
    if pre_bgr is None:
        raise FileNotFoundError(pre_path)

    print("[TILE ]", TILE_ID)
    print("[LOC  ]", loc_ckpt.name)
    print("[PLATT]", platt_name if platt_name else "(none) a=1 b=0")
    print("[THR  ]", LOC_THRESH)

    prob, mask = predict_build_prob_and_mask(loc_model, pre_bgr, device, a, b, LOC_THRESH)

    # simple green mask visualization
    vis = np.zeros_like(pre_bgr, dtype=np.uint8)
    vis[mask] = (0, 255, 0)

    overlay = overlay_mask_on_img(pre_bgr, mask, color_bgr=(0, 255, 0), alpha=0.45)

    save_panel(
        vis, OUT_MASK,
        "(Stage-1) Building localization\nPRE image → building mask\nGreen = predicted building"
    )
    save_panel(
        overlay, OUT_OVERLAY,
        "(Stage-1) Building localization\nOverlay on PRE image\nGreen = predicted building"
    )


if __name__ == "__main__":
    main()
