#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZERO-ARGS:
    python make_panel_stage1_compare_DA.py

Outputs:
  - panel_stage1_NO_DA.png    (baseline: xView2/xBD checkpoint, no Platt)
  - panel_stage1_WITH_DA.png  (domain-adapted: IDABD fine-tuned checkpoint + Platt if available)

Both run on the SAME post-disaster RGB image.
(No fusion preprocessing here; this isolates "domain adaptation" only.)
If you also want fusion, tell me and I'll add 4 outputs (DA/noDA × fusion/noFusion).
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

# Post-disaster image
POST_IMG_REL = r"idabd/images/AOI1-tile_1-3_post_disaster.png"

# (1) WITH domain adaptation: your fine-tuned stage1 folder (contains *_idabd_ft.best.pth and *_idabd_platt.npz)
STAGE1_DA_DIR_REL = r"idabd_stage1_loc_ft_checkpoints"

# (2) WITHOUT domain adaptation: original xView2/xBD weights folder (contains dpn92_loc*.pth, res34_loc*.pth, ...)
BASELINE_WEIGHTS_DIR_REL = r"weights"

# Choose WHICH backbone to compare (must exist in BOTH places)
# Examples: "dpn92", "res34", "res50", "se154"
BACKBONE = "dpn92"

# Threshold
THRESH = 0.50

OUT_NO_DA  = "panel_stage1_NO_DA.png"
OUT_WITH_DA = "panel_stage1_WITH_DA.png"
# =========================


# --- make sure zoo import works ---
for cand in [HERE, HERE.parent, HERE.parent.parent]:
    if (cand / "zoo").is_dir():
        sys.path.insert(0, str(cand))
        break


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


# ---------- Find checkpoints ----------
def find_baseline_ckpt(weights_dir: Path, backbone: str) -> Path:
    """
    Pick the first matching baseline checkpoint, e.g. dpn92_loc*.pth
    """
    cands = sorted(weights_dir.glob(f"{backbone}_loc*.pth"))
    if not cands:
        raise FileNotFoundError(
            f"No baseline stage1 weights found in {weights_dir} with glob {backbone}_loc*.pth"
        )
    return cands[0]

def find_da_ckpt(da_dir: Path, backbone: str) -> Path:
    """
    Pick the first matching IDABD fine-tuned checkpoint, supports:
      *_idabd_ft_best.pth
      *_idabd_ft.best.pth
    """
    c1 = sorted(da_dir.glob(f"{backbone}*_idabd_ft_best.pth"))
    c2 = sorted(da_dir.glob(f"{backbone}*_idabd_ft.best.pth"))
    allc = c1 + c2
    if not allc:
        raise FileNotFoundError(
            f"No DA stage1 ckpts found in {da_dir} for backbone={backbone}.\n"
            f"Tried:\n  {backbone}*_idabd_ft_best.pth\n  {backbone}*_idabd_ft.best.pth"
        )
    return allc[0]

def find_platt_for_da_ckpt(da_dir: Path, ckpt: Path):
    """
    For DA model only: try to load platt params (a,b).
    If missing, return (1,0).
    """
    stem = ckpt.stem
    for suf in ["_idabd_ft.best", "_idabd_ft_best", "_idabd_ft.last", "_idabd_ft_last"]:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    cand = da_dir / f"{stem}_idabd_platt.npz"
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
    da_dir = HERE / STAGE1_DA_DIR_REL
    base_dir = HERE / BASELINE_WEIGHTS_DIR_REL

    if not post_path.exists():
        raise FileNotFoundError(f"Post image not found:\n  {post_path}")
    if not da_dir.exists():
        raise FileNotFoundError(f"DA folder not found:\n  {da_dir}")
    if not base_dir.exists():
        raise FileNotFoundError(f"Baseline weights folder not found:\n  {base_dir}")

    img_bgr = cv2.imread(str(post_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read:\n  {post_path}")

    # Pick checkpoints
    ckpt_no_da = find_baseline_ckpt(base_dir, BACKBONE)
    ckpt_da = find_da_ckpt(da_dir, BACKBONE)
    a_da, b_da, platt_name = find_platt_for_da_ckpt(da_dir, ckpt_da)

    print("[IMG]   ", post_path.name)
    print("[BONE]  ", BACKBONE)
    print("[NO_DA] ", ckpt_no_da.name, "(a=1,b=0; no Platt)")
    print("[WITH_DA]", ckpt_da.name, f"(Platt={platt_name if platt_name else 'none'})")
    print("[THR]   ", THRESH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- NO DOMAIN ADAPTATION ---
    m0 = build_loc_model_from_weight(str(ckpt_no_da)).to(device).eval()
    mask0 = predict_build_mask(m0, img_bgr, device, a=1.0, b=0.0, thresh=THRESH)
    render_panel(
        mask0,
        OUT_NO_DA,
        "(c)  Post disaster\nbuilding localization\nWITHOUT domain adaptation\n(xBD pretrained)"
    )

    # --- WITH DOMAIN ADAPTATION ---
    m1 = build_loc_model_from_weight(str(ckpt_da)).to(device).eval()
    mask1 = predict_build_mask(m1, img_bgr, device, a=a_da, b=b_da, thresh=THRESH)
    render_panel(
        mask1,
        OUT_WITH_DA,
        "(c)  Post disaster\nbuilding localization\nWITH domain adaptation\n(IDABD FT + Platt)"
    )


if __name__ == "__main__":
    main()
