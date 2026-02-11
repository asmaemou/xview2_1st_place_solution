#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
STAGE-2 DAMAGE CLASSIFICATION (IDABD) — FUSION AUGMENTATION ONLY (NO DOMAIN ADAPTATION)
AUTO-RUN FRIENDLY:
  python train_idabd_stage2_pipeline_FUSION_ONLY_noDA.py

Features:
- Fusion augmentation (Canny + HistEq + Unsharp + CLAHE)
- Destroyed-aware crops (helps class-4)
- Optional class weights
- Early stopping (VAL loss)
- Pipeline evaluation using Stage-1 localization gating

Stage-1 gating:
  p(building) = sigmoid(a*logit + b)   (Platt optional)
  build_mask  = p(building) >= loc_thresh

Stage-2:
  predicts 1..4 on building pixels, final map:
    pred_final = 0 if NOT building else damage_pred

Mask assumptions (POST masks):
  0   = background
  1   = No Damage
  2   = Minor
  3   = Major
  4   = Destroyed
  255 = ignore
================================================================================
"""

import os
from os import path
import sys
from pathlib import Path
import glob
import argparse
import random
import csv

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------
# Make sure "zoo" is importable
# ---------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
for cand in [THIS_DIR, THIS_DIR.parent, THIS_DIR.parent.parent]:
    if (cand / "zoo").is_dir():
        sys.path.insert(0, str(cand))
        break


# ---------------------------------------------------------------------
# Defaults (will be auto-resolved if run with no args)
# ---------------------------------------------------------------------
IMG_DIR = "../idabd/images"
MASK_DIR = "../idabd/masks"
WEIGHTS_DIR = "weights"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IGNORE_LABEL = 255
CLASS_NAMES_4 = ["No-Damage", "Minor", "Major", "Destroyed"]
BUILD_TENSOR_CPU = torch.tensor([1, 2, 3, 4], dtype=torch.long)


# ---------------------------------------------------------------------
# CSV fields (uncalibrated only)
# ---------------------------------------------------------------------
CSV_FIELDS = [
    "seed",
    "init_weight",
    "loc_weight",
    "loc_platt",
    "loc_thresh",

    "pipeline_acc",
    "pipeline_loc_precision",
    "pipeline_loc_recall",
    "pipeline_loc_f1",
    "pipeline_macroF1_damage",
    "pipeline_f1_no_damage",
    "pipeline_f1_minor",
    "pipeline_f1_major",
    "pipeline_f1_destroyed",
]


def ensure_csv(csv_path: str, overwrite: bool = False):
    if not csv_path:
        return
    out_dir = path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if overwrite and path.exists(csv_path):
        os.remove(csv_path)

    if not path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            f.flush()


def append_csv_row(csv_path: str, row: dict):
    if not csv_path:
        return
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writerow(row)
        f.flush()


# =============================================================================
# AUTO CONFIG helpers
# =============================================================================
def _first_existing_dir(candidates):
    for d in candidates:
        if d and Path(d).is_dir():
            return str(Path(d).resolve())
    return ""


def _resolve_idabd_paths(script_dir: Path):
    """
    Try common layouts relative to this script.
    Returns (img_dir, mask_dir) or ("","") if not found.
    """
    candidates = [
        script_dir / "idabd" / "images",
        script_dir / ".." / "idabd" / "images",
        script_dir / ".." / ".." / "idabd" / "images",
        script_dir / "data" / "idabd" / "images",
    ]
    img_dir = _first_existing_dir(candidates)

    candidates = [
        script_dir / "idabd" / "masks",
        script_dir / ".." / "idabd" / "masks",
        script_dir / ".." / ".." / "idabd" / "masks",
        script_dir / "data" / "idabd" / "masks",
    ]
    mask_dir = _first_existing_dir(candidates)

    return img_dir, mask_dir


def _resolve_weights_dir(script_dir: Path):
    candidates = [
        script_dir / "weights",
        script_dir / ".." / "weights",
        script_dir / ".." / ".." / "weights",
    ]
    return _first_existing_dir(candidates)


def _resolve_loc_dir(script_dir: Path):
    candidates = [
        script_dir / "idabd_stage1_loc_ft_checkpoints",
        script_dir / ".." / "idabd_stage1_loc_ft_checkpoints",
        script_dir / ".." / ".." / "idabd_stage1_loc_ft_checkpoints",
    ]
    return _first_existing_dir(candidates)


def _find_best_loc_weight(loc_dir: str, weights_dir: str):
    """
    Prefer *_idabd_ft_best.pth in idabd_stage1_loc_ft_checkpoints.
    Fallback to any *loc*.pth in weights_dir.
    """
    patterns = []
    if loc_dir:
        patterns += [
            path.join(loc_dir, "*loc*_idabd_ft_best*.pth"),
            path.join(loc_dir, "*loc*_ft_best*.pth"),
            path.join(loc_dir, "*loc*.pth"),
        ]
    if weights_dir:
        patterns += [
            path.join(weights_dir, "*loc*_idabd_ft_best*.pth"),
            path.join(weights_dir, "*loc*_best*.pth"),
            path.join(weights_dir, "*loc*.pth"),
        ]

    hits = []
    for ptn in patterns:
        hits += glob.glob(ptn)
    hits = sorted(list(dict.fromkeys(hits)))
    return hits[0] if hits else ""


def _find_platt_for_loc(loc_weight_path: str):
    """
    If loc_weight is dpn92_loc_..._idabd_ft_best.pth,
    then look for dpn92_loc_..._idabd_platt.npz next to it.
    """
    if not loc_weight_path:
        return ""
    d = Path(loc_weight_path).resolve().parent
    base = Path(loc_weight_path).name
    # Common naming from your pipeline:
    # dpn92_loc_0_tuned_best_idabd_ft_best.pth -> dpn92_loc_0_tuned_best_idabd_platt.npz
    cand = base.replace("_idabd_ft_best.pth", "_idabd_platt.npz").replace(".pth", "_platt.npz")
    platt = d / cand
    if platt.exists():
        return str(platt)
    # fallback: any platt in same folder
    alt = sorted(glob.glob(str(d / "*platt*.npz")))
    return alt[0] if alt else ""


# =============================================================================
# Pairing: (pre, post, post_mask) by tile_id
# =============================================================================
def tile_id_from_name(fname: str) -> str:
    base = path.splitext(path.basename(fname))[0]
    base = base.replace("_pre_disaster", "").replace("_post_disaster", "")
    return base


def list_split_files(root: str, split: str):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    files = []
    for e in exts:
        files += glob.glob(path.join(root, e))
    files = sorted(files)
    if split == "pre":
        files = [f for f in files if "_pre_disaster" in path.basename(f)]
    elif split == "post":
        files = [f for f in files if "_post_disaster" in path.basename(f)]
    return files


def find_mask(mask_dir: str, tile_id: str, gt_split: str):
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        cand = path.join(mask_dir, f"{tile_id}_{gt_split}_disaster{ext}")
        if path.exists(cand):
            return cand
    return None


def build_stage2_triplets(img_dir: str, mask_dir: str, gt_split: str = "post"):
    pre_imgs  = list_split_files(img_dir, "pre")
    post_imgs = list_split_files(img_dir, "post")

    pre_map  = {tile_id_from_name(p): p for p in pre_imgs}
    post_map = {tile_id_from_name(p): p for p in post_imgs}

    tile_ids = sorted(set(pre_map.keys()) & set(post_map.keys()))
    triplets = []
    for tid in tile_ids:
        m = find_mask(mask_dir, tid, gt_split)
        if m is None:
            continue
        triplets.append((pre_map[tid], post_map[tid], m))
    return triplets


# =============================================================================
# Fusion augmentation
# =============================================================================
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
    return cv2.addWeighted(img_bgr, 1 + amount, blur, -amount, 0)


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


# =============================================================================
# Preprocess
# =============================================================================
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


def preprocess_rgb(img_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))  # CHW
    return torch.from_numpy(img)


def preprocess_6ch(pre_bgr: np.ndarray, post_bgr: np.ndarray) -> torch.Tensor:
    pre = preprocess_rgb(pre_bgr)
    post = preprocess_rgb(post_bgr)
    return torch.cat([pre, post], dim=0)  # 6xHxW


def load_mask_raw(mask_path: str) -> np.ndarray:
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(mask_path)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return m.astype(np.int64)


def has_destroyed(mask_path: str) -> bool:
    m = load_mask_raw(mask_path)
    v = (m != IGNORE_LABEL)
    if v.sum() == 0:
        return False
    return bool((m[v] == 4).any())


# =============================================================================
# Dataset (Stage-2): DESTROYED-AWARE CROPS
# =============================================================================
class IdaBDStage2Damage(Dataset):
    def __init__(
        self,
        triplets,
        crop_size=512,
        is_train=True,
        fusion_p=0.75,
        min_build_px=50,
        seed=0,
        focus_destroyed_p=0.6,
        min_destroyed_px=50,
        crop_attempts=20,
    ):
        self.triplets = triplets
        self.crop_size = int(crop_size) if crop_size else 0
        self.is_train = is_train
        self.fusion_p = float(fusion_p)
        self.min_build_px = int(min_build_px)
        self.rng = random.Random(seed)

        self.focus_destroyed_p = float(focus_destroyed_p)
        self.min_destroyed_px = int(min_destroyed_px)
        self.crop_attempts = int(crop_attempts)

    def __len__(self):
        return len(self.triplets)

    def _random_crop(self, pre, post, mask, size):
        h, w = mask.shape[:2]
        if h < size or w < size:
            pad_h = max(0, size - h)
            pad_w = max(0, size - w)
            pre  = cv2.copyMakeBorder(pre,  0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            post = cv2.copyMakeBorder(post, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="edge")
            h, w = mask.shape[:2]

        want_destroyed = self.is_train and (self.rng.random() < self.focus_destroyed_p)

        for _ in range(self.crop_attempts):
            y0 = self.rng.randint(0, h - size)
            x0 = self.rng.randint(0, w - size)
            m = mask[y0:y0+size, x0:x0+size]

            if want_destroyed:
                dpx = int((m == 4).sum())
                if dpx >= self.min_destroyed_px:
                    return pre[y0:y0+size, x0:x0+size], post[y0:y0+size, x0:x0+size], m
            else:
                build_px = int(np.isin(m, [1, 2, 3, 4]).sum())
                if build_px >= self.min_build_px:
                    return pre[y0:y0+size, x0:x0+size], post[y0:y0+size, x0:x0+size], m

        # fallback: relax
        for _ in range(self.crop_attempts):
            y0 = self.rng.randint(0, h - size)
            x0 = self.rng.randint(0, w - size)
            m = mask[y0:y0+size, x0:x0+size]
            build_px = int(np.isin(m, [1, 2, 3, 4]).sum())
            if build_px >= max(1, self.min_build_px // 2):
                return pre[y0:y0+size, x0:x0+size], post[y0:y0+size, x0:x0+size], m

        # final fallback
        y0 = self.rng.randint(0, h - size)
        x0 = self.rng.randint(0, w - size)
        return (
            pre[y0:y0+size, x0:x0+size],
            post[y0:y0+size, x0:x0+size],
            mask[y0:y0+size, x0:x0+size],
        )

    def __getitem__(self, idx):
        pre_path, post_path, mask_path = self.triplets[idx]

        pre  = cv2.imread(pre_path, cv2.IMREAD_COLOR)
        post = cv2.imread(post_path, cv2.IMREAD_COLOR)
        if pre is None:
            raise FileNotFoundError(pre_path)
        if post is None:
            raise FileNotFoundError(post_path)

        m = load_mask_raw(mask_path)

        if self.is_train and (self.rng.random() < self.fusion_p):
            pre = fusion_augment(pre)
            post = fusion_augment(post)

        if self.is_train and self.crop_size > 0:
            pre, post, m = self._random_crop(pre, post, m, self.crop_size)

        pre, pad_hw = pad_to_factor(pre, 32)
        post, _ = pad_to_factor(post, 32)
        if pad_hw != (0, 0):
            m = np.pad(m, ((0, pad_hw[0]), (0, pad_hw[1])), mode="edge")

        x = preprocess_6ch(pre, post)
        y = torch.from_numpy(m).long()
        return x.float(), y.long(), path.basename(post_path)


# =============================================================================
# Checkpoint loading
# =============================================================================
def strip_prefixes(sd: dict):
    prefixes = ["module.", "model.", "net."]
    out = {}
    for k, v in sd.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p):
                kk = kk[len(p):]
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


# =============================================================================
# Build models from weights
# =============================================================================
def build_damage_model_from_weight(weight_path: str):
    import zoo.models as zm
    fname = path.basename(weight_path).lower()

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


# =============================================================================
# Optional class weights (multiclass)
# =============================================================================
def compute_class_weights_from_triplets(triplets, max_images=200, seed=0):
    rng = random.Random(seed)
    items = list(triplets)
    rng.shuffle(items)
    items = items[:min(len(items), max_images)]

    counts = np.zeros(4, dtype=np.int64)  # labels 1..4
    for _, _, mpath in items:
        m = load_mask_raw(mpath)
        for i, lab in enumerate([1, 2, 3, 4]):
            counts[i] += np.sum(m == lab)

    counts = np.maximum(counts, 1)
    total = counts.sum()
    weights = total / (4.0 * counts.astype(np.float64))
    return weights.astype(np.float32), counts


# =============================================================================
# Loss: CrossEntropy on building pixels only (mask in {1..4})
# =============================================================================
def masked_ce_loss(logits, raw_mask, ce_loss: nn.Module):
    target = torch.full_like(raw_mask, IGNORE_LABEL)
    build_tensor = BUILD_TENSOR_CPU.to(raw_mask.device)
    valid = (raw_mask != IGNORE_LABEL)
    build = valid & torch.isin(raw_mask, build_tensor)
    target[build] = raw_mask[build]  # 1..4

    if build.sum().item() == 0:
        return None
    return ce_loss(logits, target)


# =============================================================================
# Stage-2 prediction (NO calibration)
# =============================================================================
@torch.no_grad()
def predict_damage_1to4_from_logits(logits):
    return torch.argmax(logits[:, 1:5, :, :], dim=1) + 1


# =============================================================================
# Stage-1 (loc) building mask prediction (with optional Platt scaling)
# =============================================================================
@torch.no_grad()
def predict_build_mask_from_x6(loc_model, x6, a, b, thresh):
    pre = x6[:, 0:3, :, :]  # normalized pre
    logit = loc_model(pre)
    if isinstance(logit, (tuple, list)):
        logit = logit[0]

    if logit.ndim == 3:
        logit = logit.unsqueeze(1)
    if logit.ndim == 4 and logit.shape[1] > 1:
        logit = logit[:, 0:1, :, :]

    logit = a * logit + b
    prob = torch.sigmoid(logit)[:, 0, :, :]  # BxHxW
    return prob >= thresh


# =============================================================================
# Confusion matrices + metrics
# =============================================================================
def _confusion_add(conf, gt_flat, pr_flat, ncls):
    idx = (gt_flat * ncls + pr_flat).astype(np.int64)
    binc = np.bincount(idx, minlength=ncls * ncls)
    conf += binc.reshape(ncls, ncls)


def f1s_from_conf(conf):
    n = conf.shape[0]
    f1s = []
    for c in range(n):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)
        f1s.append(float(f1))
    return f1s


def acc_from_conf(conf):
    return float(np.trace(conf) / (conf.sum() + 1e-9))


def prf_from_counts(tp, fp, fn):
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f = (2 * tp) / (2 * tp + fp + fn + 1e-9)
    return float(p), float(r), float(f)


# =============================================================================
# PIPELINE evaluation (Stage-1 loc -> Stage-2 damage) — UNCALIBRATED ONLY
# =============================================================================
@torch.no_grad()
def eval_loader_pipeline(model, loader, device, loc_model, loc_a, loc_b, loc_thresh, amp=False):
    model.eval()
    loc_model.eval()

    conf5 = np.zeros((5, 5), dtype=np.int64)
    loc_tp = 0.0
    loc_fp = 0.0
    loc_fn = 0.0

    build_tensor = BUILD_TENSOR_CPU.to(device)

    if amp and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        def amp_ctx():
            return torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"))
    else:
        def amp_ctx():
            return torch.cuda.amp.autocast(enabled=(device.type == "cuda" and amp))

    for x, raw, _ in loader:
        x = x.to(device)
        raw = raw.to(device)

        build_mask = predict_build_mask_from_x6(loc_model, x, loc_a, loc_b, loc_thresh)

        with amp_ctx():
            logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

        pred_damage = predict_damage_1to4_from_logits(logits)

        pred_final = torch.zeros_like(pred_damage)
        pred_final[build_mask] = pred_damage[build_mask]

        valid = (raw != IGNORE_LABEL)
        gt_build = valid & torch.isin(raw, build_tensor)
        pr_build = valid & build_mask

        loc_tp += float((pr_build & gt_build).sum().item())
        loc_fp += float((pr_build & (~gt_build)).sum().item())
        loc_fn += float(((~pr_build) & gt_build).sum().item())

        gt_np = raw.detach().cpu().numpy().astype(np.int64)
        pr_np = pred_final.detach().cpu().numpy().astype(np.int64)

        for i in range(gt_np.shape[0]):
            g = gt_np[i]
            p = pr_np[i]
            v = (g != IGNORE_LABEL)
            if v.sum() == 0:
                continue
            g_valid = g[v]
            p_valid = np.clip(p[v], 0, 4)
            _confusion_add(conf5, g_valid, p_valid, 5)

    acc5 = acc_from_conf(conf5)
    f1s5 = f1s_from_conf(conf5)
    f1s_damage_1to4 = [f1s5[c] for c in [1, 2, 3, 4]]
    macro_build = float(np.mean(f1s_damage_1to4))

    loc_p, loc_r, loc_f1 = prf_from_counts(loc_tp, loc_fp, loc_fn)
    return acc5, macro_build, f1s_damage_1to4, loc_p, loc_r, loc_f1


# =============================================================================
# VAL loss (best checkpoint + early stopping)
# =============================================================================
@torch.no_grad()
def eval_val_loss(model, loader, device, ce_loss, amp=False):
    model.eval()

    if amp and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        def amp_ctx():
            return torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"))
    else:
        def amp_ctx():
            return torch.cuda.amp.autocast(enabled=(device.type == "cuda" and amp))

    losses = []
    skipped = 0
    for x, raw, _ in loader:
        x = x.to(device)
        raw = raw.to(device)

        with amp_ctx():
            logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            loss = masked_ce_loss(logits, raw, ce_loss)

        if loss is None:
            skipped += 1
            continue
        losses.append(float(loss.item()))

    mean_loss = float(np.mean(losses)) if losses else 1e9
    return mean_loss, skipped


# =============================================================================
# Training + pipeline evaluation for one init checkpoint
# =============================================================================
def train_and_eval_pipeline_one(args, init_weight: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] {device}")
    print(f"[INIT ] {init_weight}")

    # -------------------------
    # Load Stage-1 localization (mandatory)
    # -------------------------
    if not args.loc_weight:
        raise ValueError("This pipeline script requires --loc_weight.")
    if not path.exists(args.loc_weight):
        raise FileNotFoundError(f"--loc_weight not found: {args.loc_weight}")

    loc_model = build_loc_model_from_weight(args.loc_weight).to(device).eval()
    for p in loc_model.parameters():
        p.requires_grad = False
    print(f"[LOC  ] weights: {args.loc_weight}")

    loc_a, loc_b = 1.0, 0.0
    if args.loc_platt:
        if not path.exists(args.loc_platt):
            raise FileNotFoundError(f"--loc_platt not found: {args.loc_platt}")
        d = np.load(args.loc_platt)
        loc_a = float(d["a"])
        loc_b = float(d["b"])
        print(f"[LOC  ] platt: {args.loc_platt} (a={loc_a:.6f}, b={loc_b:.6f})")
    else:
        print("[LOC  ] platt: (none) -> using a=1, b=0")

    print(f"[LOC  ] thresh: {args.loc_thresh}")

    # -------------------------
    # Build IDABD triplets + split
    # -------------------------
    triplets = build_stage2_triplets(args.img_dir, args.mask_dir, gt_split=args.gt_split)
    if not triplets:
        raise FileNotFoundError("No (pre, post, mask) triplets found. Check img_dir/mask_dir paths.")

    rng = random.Random(args.seed)
    rng.shuffle(triplets)
    n = len(triplets)

    n_test = max(1, int(n * args.test_ratio))
    n_val  = max(1, int(n * args.val_ratio))
    if n_test + n_val >= n:
        n_test = max(1, min(n_test, n - 2))
        n_val  = max(1, min(n_val,  n - 1 - n_test))

    test_tr  = triplets[:n_test]
    val_tr   = triplets[n_test:n_test+n_val]
    train_tr = triplets[n_test+n_val:]

    # Ensure TRAIN has some destroyed tiles (helpful)
    if args.min_train_destroyed_tiles > 0:
        train_d = [t for t in train_tr if has_destroyed(t[2])]
        if len(train_d) < args.min_train_destroyed_tiles:
            val_d = [t for t in val_tr if has_destroyed(t[2])]
            need = args.min_train_destroyed_tiles - len(train_d)
            moved = 0
            for t in val_d:
                if moved >= need:
                    break
                if len(train_tr) == 0:
                    break
                train_swap = train_tr[0]
                train_tr.remove(train_swap)
                val_tr.append(train_swap)

                if t in val_tr:
                    val_tr.remove(t)
                    train_tr.append(t)
                    moved += 1
            print(f"[SPLIT] ensured TRAIN destroyed tiles: {len([t for t in train_tr if has_destroyed(t[2])])} (moved={moved})")

    print(f"[DATA ] total={n} | train={len(train_tr)} | val={len(val_tr)} | test={len(test_tr)}")

    # -------------------------
    # Loss setup
    # -------------------------
    if args.use_class_weights:
        w4, counts = compute_class_weights_from_triplets(train_tr, max_images=args.weight_count_images, seed=args.seed)
        w4[3] *= float(args.destroyed_weight_mul)
        print(f"[WGT  ] pixel counts (train subset): {counts.tolist()}")
        print(f"[WGT  ] class weights (1..4) AFTER destroyed_mul={args.destroyed_weight_mul}: {w4.tolist()}")
        weight5 = np.ones(5, dtype=np.float32)
        weight5[1:5] = w4
        ce_loss = nn.CrossEntropyLoss(
            ignore_index=IGNORE_LABEL,
            reduction="mean",
            weight=torch.from_numpy(weight5).to(device)
        )
    else:
        ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL, reduction="mean")

    # -------------------------
    # Loaders
    # -------------------------
    train_ds = IdaBDStage2Damage(
        train_tr,
        crop_size=args.crop,
        is_train=True,
        fusion_p=args.fusion_p,
        min_build_px=args.min_build_px,
        seed=args.seed,
        focus_destroyed_p=args.focus_destroyed_p,
        min_destroyed_px=args.min_destroyed_px,
        crop_attempts=args.crop_attempts,
    )
    val_ds   = IdaBDStage2Damage(val_tr, crop_size=0, is_train=False, fusion_p=0.0, min_build_px=0, seed=args.seed)
    test_ds  = IdaBDStage2Damage(test_tr, crop_size=0, is_train=False, fusion_p=0.0, min_build_px=0, seed=args.seed)

    train_ld = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_ld = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(0, args.workers // 2), pin_memory=True
    )
    test_ld = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=max(0, args.workers // 2), pin_memory=True
    )

    # -------------------------
    # Build Stage-2 model from init weights, fine-tune on IDABD
    # -------------------------
    model = build_damage_model_from_weight(init_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Modern GradScaler
    if device.type == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    os.makedirs(args.out_dir, exist_ok=True)
    base = path.splitext(path.basename(init_weight))[0]
    best_ckpt_path = path.join(args.out_dir, f"{base}_idabd_stage2_ft_best.pth")
    last_ckpt_path = path.join(args.out_dir, f"{base}_idabd_stage2_ft_last.pth")

    best_val = 1e9
    best_epoch = -1

    patience = int(args.early_stop_patience)
    warmup = int(args.early_stop_warmup)
    min_delta = float(args.early_stop_min_delta)
    no_improve = 0

    if args.amp and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        def amp_ctx():
            return torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"))
    else:
        def amp_ctx():
            return torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp))

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        skipped = 0

        for x, raw, _ in train_ld:
            x = x.to(device, non_blocking=True)
            raw = raw.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with amp_ctx():
                logits = model(x)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                loss = masked_ce_loss(logits, raw, ce_loss)

            if loss is None:
                skipped += 1
                continue

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

        mean_loss = float(np.mean(losses)) if losses else 0.0
        val_loss, val_skipped = eval_val_loss(model, val_ld, device, ce_loss, amp=args.amp)

        print(f"[EPOCH {epoch:03d}/{args.epochs}] "
              f"train_loss={mean_loss:.5f} (skipped={skipped}) | "
              f"val_loss={val_loss:.5f} (skipped={val_skipped})")

        # save last
        torch.save({
            "epoch": epoch,
            "init_weight": init_weight,
            "state_dict": model.state_dict(),
            "args": vars(args),
            "train_loss": mean_loss,
            "val_loss": val_loss,
        }, last_ckpt_path)

        improved = (val_loss < (best_val - min_delta))
        if improved:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "init_weight": init_weight,
                "state_dict": model.state_dict(),
                "args": vars(args),
                "train_loss": mean_loss,
                "val_loss": val_loss,
            }, best_ckpt_path)
            print(f"[SAVE ] Best (val) -> {best_ckpt_path} (best_val={best_val:.6f})")
        else:
            if epoch >= warmup:
                no_improve += 1

        if patience > 0 and epoch >= warmup and no_improve >= patience:
            print(f"[EARLY STOP] No VAL improvement for {patience} epochs "
                  f"(min_delta={min_delta}). Stopping at epoch {epoch}. "
                  f"Best was epoch {best_epoch} with val_loss={best_val:.6f}.")
            break

    # -------------------------
    # Load best and pipeline eval
    # -------------------------
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["state_dict"], strict=True)
    model.eval()
    print(f"\n[LOAD ] BEST checkpoint for test: {best_ckpt_path}")

    acc, macroB, f1sD, locP, locR, locF = eval_loader_pipeline(
        model, test_ld, device,
        loc_model=loc_model, loc_a=loc_a, loc_b=loc_b, loc_thresh=args.loc_thresh,
        amp=args.amp,
    )

    print("\n==================== PIPELINE TEST (AUTO, FUSION ONLY, NO-DA) ====================")
    print("END-TO-END: Stage-1 loc gating -> Stage-2 damage (VALID pixels only; mask!=255)")
    print(f"pipeline_acc(0..4)={acc:.6f}")
    print(f"F1 Localization (Building vs Background)={locF:.6f} (P={locP:.6f}, R={locR:.6f})")
    print(f"macroF1(Damage 1..4, end-to-end conf)={macroB:.6f}")
    print("Per-damage F1 (end-to-end conf):")
    for i in range(4):
        print(f"F1 {CLASS_NAMES_4[i]:>9s}: {f1sD[i]:.6f}")
    print("===============================================================================\n")

    row = {
        "seed": args.seed,
        "init_weight": init_weight,
        "loc_weight": args.loc_weight,
        "loc_platt": args.loc_platt,
        "loc_thresh": args.loc_thresh,

        "pipeline_acc": acc,
        "pipeline_loc_precision": locP,
        "pipeline_loc_recall": locR,
        "pipeline_loc_f1": locF,
        "pipeline_macroF1_damage": macroB,
        "pipeline_f1_no_damage": f1sD[0],
        "pipeline_f1_minor": f1sD[1],
        "pipeline_f1_major": f1sD[2],
        "pipeline_f1_destroyed": f1sD[3],
    }
    append_csv_row(args.csv_path, row)
    print(f"[CSV  ] Appended -> {args.csv_path}")


# =============================================================================
# Main
# =============================================================================
def dedup_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _apply_auto_defaults_if_no_args(args):
    """
    If the user ran only:
      python train_idabd_stage2_pipeline_FUSION_ONLY_noDA.py
    then auto-resolve dirs + loc weights + set train_all.
    """
    script_dir = THIS_DIR

    img_auto, mask_auto = _resolve_idabd_paths(script_dir)
    wdir_auto = _resolve_weights_dir(script_dir)
    locdir_auto = _resolve_loc_dir(script_dir)

    # Only override if user didn't provide explicit paths
    if args.img_dir == IMG_DIR and img_auto:
        args.img_dir = img_auto
    if args.mask_dir == MASK_DIR and mask_auto:
        args.mask_dir = mask_auto
    if args.weights_dir == WEIGHTS_DIR and wdir_auto:
        args.weights_dir = wdir_auto

    # Find loc weight if missing
    if not args.loc_weight:
        lw = _find_best_loc_weight(locdir_auto, args.weights_dir)
        args.loc_weight = lw

    # Find platt if user didn't specify (optional)
    if not args.loc_platt and args.loc_weight:
        args.loc_platt = _find_platt_for_loc(args.loc_weight)

    # Default output + csv
    if args.out_dir == "idabd_stage2_damage_ft_checkpoints_FUSION_ONLY_noDA":
        # keep default; ensure it's created later
        pass
    if args.csv_path == "idabd_stage2_pipeline_results_FUSION_ONLY_noDA.csv":
        # keep default
        pass

    # If init_weight is empty, run train_all by default
    if not args.init_weight:
        args.train_all = True

    # Print AUTO CONFIG
    print("\n[AUTO CONFIG]")
    print(f"  img_dir     : {args.img_dir}")
    print(f"  mask_dir    : {args.mask_dir}")
    print(f"  weights_dir : {args.weights_dir}")
    print(f"  loc_dir     : {locdir_auto if locdir_auto else '(not found)'}")
    print(f"  loc_weight  : {args.loc_weight if args.loc_weight else '(NOT FOUND)'}")
    print(f"  loc_platt   : {args.loc_platt if args.loc_platt else '(none)'}")
    print(f"  train_all   : {args.train_all}")
    print(f"  out_dir     : {args.out_dir}")
    print(f"  csv_path    : {args.csv_path}")
    print("")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--img_dir", default=IMG_DIR)
    ap.add_argument("--mask_dir", default=MASK_DIR)
    ap.add_argument("--weights_dir", default=WEIGHTS_DIR)

    ap.add_argument("--gt_split", choices=["pre", "post"], default="post")

    # split
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)

    # training
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")

    # early stopping
    ap.add_argument("--early_stop_patience", type=int, default=15)
    ap.add_argument("--early_stop_warmup", type=int, default=5)
    ap.add_argument("--early_stop_min_delta", type=float, default=0.0)

    # crop + fusion
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--fusion_p", type=float, default=0.75)
    ap.add_argument("--min_build_px", type=int, default=50)

    # destroyed-aware sampling
    ap.add_argument("--focus_destroyed_p", type=float, default=0.6)
    ap.add_argument("--min_destroyed_px", type=int, default=50)
    ap.add_argument("--crop_attempts", type=int, default=20)

    # class weights
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--weight_count_images", type=int, default=200)
    ap.add_argument("--destroyed_weight_mul", type=float, default=3.0)

    # stage-2 init weights
    ap.add_argument("--init_weight", default="")
    ap.add_argument("--train_all", action="store_true")

    ap.add_argument("--out_dir", default="idabd_stage2_damage_ft_checkpoints_FUSION_ONLY_noDA")

    # enforce destroyed presence in TRAIN (optional safety)
    ap.add_argument("--min_train_destroyed_tiles", type=int, default=1)

    # Stage-1 localization for pipeline evaluation
    ap.add_argument("--loc_weight", type=str, default="",
                    help="Path to stage-1 localization weight: *_idabd_ft_best.pth (required or auto-found)")
    ap.add_argument("--loc_platt", type=str, default="",
                    help="Path to stage-1 Platt params: *_platt.npz (optional / auto-found)")
    ap.add_argument("--loc_thresh", type=float, default=0.5)

    # CSV
    ap.add_argument("--csv_path", default="idabd_stage2_pipeline_results_FUSION_ONLY_noDA.csv")
    ap.add_argument("--overwrite_csv", action="store_true")

    args = ap.parse_args()

    # If user ran with no CLI args, auto-config.
    # (This check is reliable: only script name means len(sys.argv)==1)
    if len(sys.argv) == 1:
        _apply_auto_defaults_if_no_args(args)

    # basic validation
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    if not args.loc_weight:
        raise ValueError(
            "loc_weight is required. Auto-find failed.\n"
            "Provide it explicitly:\n"
            "  --loc_weight idabd_stage1_loc_ft_checkpoints/<your_loc>_idabd_ft_best.pth"
        )
    if not path.exists(args.loc_weight):
        raise FileNotFoundError(f"--loc_weight not found: {args.loc_weight}")

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ensure_csv(args.csv_path, overwrite=args.overwrite_csv)
    print(f"[CSV  ] Ready: {args.csv_path}")

    def _collect_train_all_weights(weights_dir):
        patterns = [
            "dpn92_cls*.pth",
            "res34_cls*.pth",
            "res34_cls2*.pth",
            "res50_cls*.pth",
            "se154_cls*.pth",
        ]
        weight_files = []
        for ptn in patterns:
            weight_files += glob.glob(path.join(weights_dir, ptn))
        weight_files = sorted(dedup_preserve_order(weight_files))
        return weight_files

    if args.train_all:
        weight_files = _collect_train_all_weights(args.weights_dir)
        if not weight_files:
            raise FileNotFoundError(f"No cls weights found in {args.weights_dir}")

        print("[WEIGHTS] Stage-2 init checkpoints to run:")
        for w in weight_files:
            print("  -", w)

        for w in weight_files:
            train_and_eval_pipeline_one(args, w)
    else:
        if not args.init_weight:
            raise ValueError("Provide --init_weight <path> OR use --train_all")
        if not path.exists(args.init_weight):
            raise FileNotFoundError(args.init_weight)
        train_and_eval_pipeline_one(args, args.init_weight)


if __name__ == "__main__":
    main()
