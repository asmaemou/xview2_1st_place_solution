#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZERO-ARGS:
    python make_panel_c_stage1.py

Produces panel_c.png (like your Fig (c)):
- Stage-1 localization prediction after applying fusion-based augmentation
- Black background + green building mask
- Caption under the image

Robust to your filename style:
  dpn92_loc_0_tuned_best_idabd_ft.best.pth   (note: ".ft.best")
  dpn92_loc_0_tuned_best_idabd_platt.npz
"""

import sys
from pathlib import Path
import glob
from os import path

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS (you only need to edit POST_IMG_REL if needed)
# ============================================================
HERE = Path(__file__).resolve().parent

# Your post-disaster image (relative to new_approach)
POST_IMG_REL = r"idabd/images/AOI1-tile_1-3_post_disaster.png"

# Your stage-1 checkpoints folder (relative to new_approach)
STAGE1_DIR_REL = r"idabd_stage1_loc_ft_checkpoints"

OUT_PATH = "panel_c.png"
THRESH = 0.50
# ============================================================


# ---------------------------------------------------------------------
# Make sure "zoo" is importable (same trick you used)
# ---------------------------------------------------------------------
for cand in [HERE, HERE.parent, HERE.parent.parent]:
    if (cand / "zoo").is_dir():
        sys.path.insert(0, str(cand))
        break


# ---------------------------------------------------------------------
# Imagenet normalization (same as training)
# ---------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------
# Fusion augmentation (Canny + HistEq + Unsharp + CLAHE)
# ---------------------------------------------------------------------
def canny_edges_rgb(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def hist_equalize_rgb(img_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def unsharp_mask_rgb(img_bgr: np.ndarray, amount=1.0, radius=3) -> np.ndarray:
    blur = cv2.GaussianBlur(img_bgr, (0, 0), radius)
    sharp = cv2.addWeighted(img_bgr, 1 + amount, blur, -amount, 0)
    return sharp

def clahe_rgb(img_bgr: np.ndarray, clip=2.0, tile=(8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def fusion_augment(img_bgr: np.ndarray) -> np.ndarray:
    e = canny_edges_rgb(img_bgr)
    h = hist_equalize_rgb(img_bgr)
    u = unsharp_mask_rgb(img_bgr, amount=1.0, radius=3)
    c = clahe_rgb(img_bgr, clip=2.0, tile=(8, 8))
    fused = (
        0.60 * img_bgr.astype(np.float32) +
        0.10 * e.astype(np.float32) +
        0.10 * h.astype(np.float32) +
        0.10 * u.astype(np.float32) +
        0.10 * c.astype(np.float32)
    )
    return np.clip(fused, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------
# Preprocess + padding
# ---------------------------------------------------------------------
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
    pad_h, pad_w = pad_hw
    if pad_h == 0 and pad_w == 0:
        return arr2d
    return arr2d[: arr2d.shape[0] - pad_h, : arr2d.shape[1] - pad_w]

def preprocess_rgb(img_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))  # CHW
    return torch.from_numpy(img).float()


# ---------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------
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

def build_loc_model_from_weight(weight_path: str):
    import zoo.models as zm
    fname = path.basename(weight_path).lower()

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


def find_stage1_ckpt(stage1_dir: Path) -> Path:
    """
    Your checkpoints can be:
      *_idabd_ft_best.pth
      *_idabd_ft.best.pth
    so we search both.
    """
    g1 = sorted(stage1_dir.glob("*_idabd_ft_best.pth"))
    g2 = sorted(stage1_dir.glob("*_idabd_ft.best.pth"))
    allc = g1 + g2
    if not allc:
        raise FileNotFoundError(
            f"No checkpoints found in:\n  {stage1_dir}\n"
            f"Tried globs:\n"
            f"  *_idabd_ft_best.pth\n"
            f"  *_idabd_ft.best.pth\n"
        )
    return allc[0]


def find_platt_for_ckpt(stage1_dir: Path, ckpt_path: Path) -> Path | None:
    """
    Your naming:
      ckpt:  dpn92_loc_0_tuned_best_idabd_ft.best.pth
      platt: dpn92_loc_0_tuned_best_idabd_platt.npz

    So:
      remove "_idabd_ft.best" or "_idabd_ft_best" from stem, then add "_idabd_platt.npz"
    """
    stem = ckpt_path.stem  # filename without ".pth"
    for suf in ["_idabd_ft.best", "_idabd_ft_best", "_idabd_ft_last", "_idabd_ft.last"]:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break

    cand = stage1_dir / f"{stem}_idabd_platt.npz"
    return cand if cand.exists() else None


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
@torch.no_grad()
def main():
    post_path = HERE / POST_IMG_REL
    stage1_dir = HERE / STAGE1_DIR_REL

    print("[PATH] post:", post_path)
    print("[PATH] stage1_dir:", stage1_dir)

    if not post_path.exists():
        raise FileNotFoundError(f"Post image not found:\n  {post_path}")
    if not stage1_dir.exists():
        raise FileNotFoundError(f"Stage-1 folder not found:\n  {stage1_dir}")

    ckpt_path = find_stage1_ckpt(stage1_dir)
    platt_path = find_platt_for_ckpt(stage1_dir, ckpt_path)

    print("[CKPT] ", ckpt_path.name)
    if platt_path is not None:
        d = np.load(str(platt_path))
        a = float(d["a"])
        b = float(d["b"])
        print("[PLATT]", platt_path.name, f"(a={a:.6f}, b={b:.6f})")
    else:
        a, b = 1.0, 0.0
        print("[PLATT] not found -> using a=1, b=0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEV] ", device)

    model = build_loc_model_from_weight(str(ckpt_path)).to(device).eval()

    img_bgr = cv2.imread(str(post_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read:\n  {post_path}")

    # Apply fusion augmentation (so this is "after applying fusion-based augmentation")
    fused_bgr = fusion_augment(img_bgr)

    # Pad + preprocess
    fused_bgr, pad_hw = pad_to_factor(fused_bgr, 32)
    x = preprocess_rgb(fused_bgr).unsqueeze(0).to(device)  # 1x3xHxW

    # Forward
    logit = model(x)
    if isinstance(logit, (tuple, list)):
        logit = logit[0]
    if logit.ndim == 3:
        logit = logit.unsqueeze(1)
    if logit.ndim == 4 and logit.shape[1] > 1:
        logit = logit[:, 0:1, :, :]

    # Platt + sigmoid + threshold
    logit = a * logit + b
    prob = torch.sigmoid(logit)[0, 0].detach().cpu().numpy()
    prob = unpad2d(prob, pad_hw)
    build = (prob >= THRESH)

    # Render black + green
    vis = np.zeros((build.shape[0], build.shape[1], 3), dtype=np.uint8)
    vis[build] = (0, 255, 0)

    # Add caption under panel
    fig = plt.figure(figsize=(3.2, 5.2), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[10, 4], hspace=0.05)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_txt = fig.add_subplot(gs[1, 0])

    ax_img.imshow(vis)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_frame_on(True)

    ax_txt.axis("off")
    caption = "(c)  Post disaster\nbuilding localization\nafter applying\nfusion-based\naugmentation"
    ax_txt.text(0.02, 0.95, caption, ha="left", va="top", fontsize=14)

    fig.savefig(OUT_PATH, bbox_inches="tight")
    print("[OK] Saved:", OUT_PATH)


if __name__ == "__main__":
    main()
