#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZERO-ARGS:
    python make_panel_c_stage1_compare.py

Outputs:
  - panel_c_with_fusion.png
  - panel_c_without_fusion.png

Both use the same Stage-1 checkpoint (+ Platt if available).
Only difference: input preprocessing
  - WITH: fusion_augment(img)
  - WITHOUT: raw img
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
POST_IMG_REL = r"idabd/images/AOI1-tile_1-3_post_disaster.png"
STAGE1_DIR_REL = r"idabd_stage1_loc_ft_checkpoints"
THRESH = 0.50

OUT_WITH = "panel_c_with_fusion.png"
OUT_WITHOUT = "panel_c_without_fusion.png"
# =========================


# --- make sure zoo import works ---
for cand in [HERE, HERE.parent, HERE.parent.parent]:
    if (cand / "zoo").is_dir():
        sys.path.insert(0, str(cand))
        break


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------- Fusion augmentation ----------
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


# ---------- Preprocess ----------
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


# ---------- Model loading ----------
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
def predict_build_mask(model, img_bgr, device, a, b, thresh):
    img_bgr, pad_hw = pad_to_factor(img_bgr, 32)
    x = preprocess_rgb(img_bgr).unsqueeze(0).to(device)

    logit = model(x)
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

def render_panel(build_mask: np.ndarray, out_path: str, caption: str):
    vis = np.zeros((build_mask.shape[0], build_mask.shape[1], 3), dtype=np.uint8)
    vis[build_mask] = (0, 255, 0)

    fig = plt.figure(figsize=(3.2, 5.2), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[10, 4], hspace=0.05)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_txt = fig.add_subplot(gs[1, 0])

    ax_img.imshow(vis)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_frame_on(True)

    ax_txt.axis("off")
    ax_txt.text(0.02, 0.95, caption, ha="left", va="top", fontsize=14)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved:", out_path)


def main():
    post_path = HERE / POST_IMG_REL
    stage1_dir = HERE / STAGE1_DIR_REL

    if not post_path.exists():
        raise FileNotFoundError(post_path)
    if not stage1_dir.exists():
        raise FileNotFoundError(stage1_dir)

    ckpt = find_stage1_ckpt(stage1_dir)
    a, b, platt_name = find_platt(stage1_dir, ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_loc_model_from_weight(str(ckpt)).to(device).eval()

    img_bgr = cv2.imread(str(post_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(post_path)

    print("[CKPT ]", ckpt.name)
    print("[PLATT]", platt_name if platt_name else "(none) a=1 b=0")
    print("[THR  ]", THRESH)

    # WITHOUT fusion
    m0 = predict_build_mask(model, img_bgr, device, a, b, THRESH)
    render_panel(
        m0, OUT_WITHOUT,
        "(c)  Post disaster\nbuilding localization\nWITHOUT fusion-based\naugmentation"
    )

    # WITH fusion
    img_fused = fusion_augment(img_bgr)
    m1 = predict_build_mask(model, img_fused, device, a, b, THRESH)
    render_panel(
        m1, OUT_WITH,
        "(c)  Post disaster\nbuilding localization\nafter applying\nfusion-based\naugmentation"
    )


if __name__ == "__main__":
    main()
