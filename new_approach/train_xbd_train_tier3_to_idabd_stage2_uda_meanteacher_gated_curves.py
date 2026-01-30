#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
UDA: XBD(train + tier3) -> IDA-BD  (Stage-2 Damage)
Mean Teacher + Building-Gated Pseudo Labels (Stage-1 localization gate)

Key stability fixes (8GB GPU + Windows friendly):
  ✅ Disable DataLoader pin_memory (prevents pin thread CUDA OOM on Windows)
  ✅ Default WORKERS=0 (most stable on Windows; increase later if stable)
  ✅ Gradient accumulation (ACCUM_STEPS) to keep memory low
  ✅ Split backward passes (source loss backward, then target loss backward)
     -> avoids keeping BOTH graphs at the same time (big memory win)
  ✅ Ensure losses are always tensors (prevents "does not require grad" crash)

Run:
    python train_xbd_train_tier3_to_idabd_stage2_uda_meanteacher_gated_curves.py
================================================================================
"""

# =============================================================================
# CONFIG (edit this only)
# =============================================================================
CONFIG = {
    # ----------------------------
    # Paths
    # ----------------------------
    "XBD_SOURCE_ROOTS": ["xbd/train", "xbd/tier3"],   # labeled source
    "IDABD_IMG_DIR": "idabd/images",                 # target images (pre+post)
    "IDABD_MASK_DIR": "idabd/masks",                 # target masks (VAL/TEST only)
    "WEIGHTS_DIR": "weights",

    # ----------------------------
    # Train/Eval split
    # ----------------------------
    "GT_SPLIT": "post",
    "VAL_RATIO": 0.10,
    "TEST_RATIO": 0.10,

    # ----------------------------
    # Training
    # ----------------------------
    "EPOCHS": 10,
    "BATCH": 1,                 # keep small for 8GB GPU
    "ACCUM_STEPS": 2,           # effective batch = BATCH * ACCUM_STEPS
    "LR": 1e-4,
    "WD": 1e-4,
    "WORKERS": 0,               # Windows stable default
    "PIN_MEMORY": False,        # IMPORTANT: prevents pin thread CUDA OOM
    "AMP": True,
    "SEED": 0,

    # crops + augment
    "CROP": 512,                # if still OOM, reduce to 384
    "SRC_FUSION_P": 0.0,
    "TGT_FUSION_P": 0.75,

    # loss weights and ramp
    "SRC_LAMBDA": 1.0,
    "UDA_LAMBDA": 1.0,
    "UDA_RAMP_EPOCHS": 5,

    # Mean Teacher EMA
    "EMA_DECAY": 0.999,

    # pseudo label filtering
    "PL_CONF": 0.85,
    "PL_CONF_DESTROYED": 0.75,
    "PL_TEMP": 1.0,
    "UDA_USE_KL": False,

    # Stage-1 localization gate
    "LOC_THRESH": 0.5,
    "LOC_PLATT": "",

    # weights
    "LOC_WEIGHT": "",
    "INIT_WEIGHT": "",
    "TRAIN_ALL": False,

    # Curves
    "CURVE_EVAL_SPLIT": "val",
    "OVERWRITE_RUN_FILES": True,

    # Output
    "OUT_DIR": "xbd_train_tier3_to_idabd_UDA_MT_checkpoints",

    # Pairing debug prints
    "PAIR_DEBUG": True,
}
# =============================================================================

import os
# Helps fragmentation sometimes (safe to keep). Must be before CUDA allocations.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

from os import path
import sys
from pathlib import Path
import glob
import random
import csv
import copy
import re
from itertools import cycle

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Ensure "zoo" is importable
# ---------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
for cand in [THIS_DIR, THIS_DIR.parent, THIS_DIR.parent.parent]:
    if (cand / "zoo").is_dir():
        sys.path.insert(0, str(cand))
        break

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IGNORE_LABEL = 255
CLASS_NAMES_4 = ["No-Damage", "Minor", "Major", "Destroyed"]
BUILD_TENSOR_CPU = torch.tensor([1, 2, 3, 4], dtype=torch.long)

# ---------------------------------------------------------------------
# Metrics CSV
# ---------------------------------------------------------------------
METRICS_FIELDS = [
    "run_tag",
    "init_weight",
    "loc_weight",
    "seed",
    "epoch",
    "split",
    "train_loss",
    "val_loss",
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

def ensure_csv(csv_path: str, fields, overwrite: bool = False):
    if not csv_path:
        return
    out_dir = path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if overwrite and path.exists(csv_path):
        os.remove(csv_path)
    if not path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            f.flush()

def append_csv_row(csv_path: str, fields, row: dict):
    if not csv_path:
        return
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writerow(row)
        f.flush()

# =============================================================================
# Train/Val/Test split helper
# =============================================================================
def split_list(items, val_ratio: float, test_ratio: float, seed: int = 0):
    items = list(items)
    rng = random.Random(int(seed))
    rng.shuffle(items)

    n = len(items)
    n_test = max(1, int(round(n * float(test_ratio))))
    n_val  = max(1, int(round(n * float(val_ratio))))

    if n_test + n_val >= n:
        n_test = max(1, min(n_test, n - 2))
        n_val  = max(1, min(n_val,  n - 1 - n_test))

    test_items = items[:n_test]
    val_items  = items[n_test:n_test + n_val]
    train_items= items[n_test + n_val:]
    return train_items, val_items, test_items

# =============================================================================
# Pairing & mask matching
# =============================================================================
def _all_images(root: str):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    files = []
    for e in exts:
        files += glob.glob(path.join(root, e))
    return sorted(files)

def _strip_known_suffixes(stem: str) -> str:
    s = stem
    for _ in range(3):
        s2 = re.sub(r"(_target|_mask|_labels?|_label)$", "", s, flags=re.IGNORECASE)
        if s2 == s:
            break
        s = s2
    return s

def _strip_prepost_disaster(stem: str) -> str:
    return re.sub(r"_(pre|post)_disaster$", "", stem, flags=re.IGNORECASE)

def _parse_split_and_tile_id(p: str):
    base = path.splitext(path.basename(p))[0]
    base = _strip_known_suffixes(base)
    m = re.search(r"_(pre|post)_disaster$", base, flags=re.IGNORECASE)
    if m:
        split = m.group(1).lower()
        tile_id = base[:m.start()]
        return split, tile_id
    return None, _strip_prepost_disaster(base)

def _build_mask_index(mask_dir: str):
    idx = {}
    for mp in _all_images(mask_dir):
        stem = path.splitext(path.basename(mp))[0]
        stem_l = stem.lower()
        idx[stem_l] = mp
        norm = _strip_known_suffixes(stem).lower()
        idx[norm] = mp
        tile_only = _strip_prepost_disaster(norm).lower()
        idx[tile_only] = mp
    return idx

def find_mask_for_image(mask_dir: str, image_path: str):
    bname = path.basename(image_path)
    direct = path.join(mask_dir, bname)
    if path.exists(direct):
        return direct

    stem, ext = path.splitext(bname)

    cand = path.join(mask_dir, f"{stem}_target{ext}")
    if path.exists(cand):
        return cand

    tile_only = _strip_prepost_disaster(stem)
    cand2 = path.join(mask_dir, f"{tile_only}_target{ext}")
    if path.exists(cand2):
        return cand2

    for e in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        for c in [
            path.join(mask_dir, f"{tile_only}_target{e}"),
            path.join(mask_dir, f"{stem}_target{e}"),
            path.join(mask_dir, f"{stem}{e}"),
        ]:
            if path.exists(c):
                return c

    idx = _build_mask_index(mask_dir)
    if stem.lower() in idx:
        return idx[stem.lower()]
    if tile_only.lower() in idx:
        return idx[tile_only.lower()]
    return None

def build_stage2_triplets(img_dir: str, mask_dir: str, gt_split: str = "post", debug: bool = False):
    if not path.isdir(img_dir):
        raise FileNotFoundError(f"img_dir not found: {img_dir}")
    if not path.isdir(mask_dir):
        raise FileNotFoundError(f"mask_dir not found: {mask_dir}")

    imgs = _all_images(img_dir)
    pre_map, post_map = {}, {}

    for p in imgs:
        sp, tid = _parse_split_and_tile_id(p)
        if sp == "pre":
            pre_map[tid] = p
        elif sp == "post":
            post_map[tid] = p

    tile_ids = sorted(set(pre_map.keys()) & set(post_map.keys()))
    triplets = []
    for tid in tile_ids:
        pre_p = pre_map[tid]
        post_p = post_map[tid]
        ref_img = post_p if gt_split == "post" else pre_p
        m = find_mask_for_image(mask_dir, ref_img)
        if m is None:
            for e in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                c = path.join(mask_dir, f"{tid}_target{e}")
                if path.exists(c):
                    m = c
                    break
        if m is not None:
            triplets.append((pre_p, post_p, m))

    if debug:
        print(f"[PAIR] img_dir={img_dir}")
        print(f"[PAIR] mask_dir={mask_dir}")
        print(f"[PAIR] found images: total={len(imgs)} pre={len(pre_map)} post={len(post_map)} paired_tiles={len(tile_ids)}")
        print(f"[PAIR] built triplets: {len(triplets)} (gt_split={gt_split})")

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
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    l2 = cl.apply(l)
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
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)

def preprocess_6ch(pre_bgr: np.ndarray, post_bgr: np.ndarray) -> torch.Tensor:
    return torch.cat([preprocess_rgb(pre_bgr), preprocess_rgb(post_bgr)], dim=0)

def load_mask_raw(mask_path: str) -> np.ndarray:
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(mask_path)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return m.astype(np.int64)

# =============================================================================
# Datasets
# =============================================================================
class LabeledDamageDataset(Dataset):
    def __init__(self, triplets, crop_size=512, is_train=True, fusion_p=0.0, seed=0):
        self.triplets = triplets
        self.crop_size = int(crop_size) if crop_size else 0
        self.is_train = is_train
        self.fusion_p = float(fusion_p)
        self.rng = random.Random(seed)

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
        y0 = self.rng.randint(0, h - size)
        x0 = self.rng.randint(0, w - size)
        return pre[y0:y0+size, x0:x0+size], post[y0:y0+size, x0:x0+size], mask[y0:y0+size, x0:x0+size]

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

        x = preprocess_6ch(pre, post).float()
        y = torch.from_numpy(m).long()
        return x, y, path.basename(post_path)

class UnlabeledDamageDataset(Dataset):
    def __init__(self, triplets, crop_size=512, fusion_p=0.75, seed=0):
        self.triplets = triplets
        self.crop_size = int(crop_size) if crop_size else 0
        self.fusion_p = float(fusion_p)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.triplets)

    def _random_crop_pair(self, pre, post, size):
        h, w = pre.shape[:2]
        if h < size or w < size:
            pad_h = max(0, size - h)
            pad_w = max(0, size - w)
            pre  = cv2.copyMakeBorder(pre,  0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            post = cv2.copyMakeBorder(post, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            h, w = pre.shape[:2]
        y0 = self.rng.randint(0, h - size)
        x0 = self.rng.randint(0, w - size)
        return pre[y0:y0+size, x0:x0+size], post[y0:y0+size, x0:x0+size]

    def __getitem__(self, idx):
        pre_path, post_path, _mask_path = self.triplets[idx]
        pre  = cv2.imread(pre_path, cv2.IMREAD_COLOR)
        post = cv2.imread(post_path, cv2.IMREAD_COLOR)
        if pre is None:
            raise FileNotFoundError(pre_path)
        if post is None:
            raise FileNotFoundError(post_path)

        if self.crop_size > 0:
            pre, post = self._random_crop_pair(pre, post, self.crop_size)

        pre_w, post_w = pre, post
        pre_s, post_s = pre.copy(), post.copy()
        if self.rng.random() < self.fusion_p:
            pre_s = fusion_augment(pre_s)
            post_s = fusion_augment(post_s)

        pre_w, _ = pad_to_factor(pre_w, 32)
        post_w, _ = pad_to_factor(post_w, 32)
        pre_s, _ = pad_to_factor(pre_s, 32)
        post_s, _ = pad_to_factor(post_s, 32)

        xw = preprocess_6ch(pre_w, post_w).float()
        xs = preprocess_6ch(pre_s, post_s).float()
        return xw, xs, path.basename(post_path)

# =============================================================================
# Checkpoint loading and model builders
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
        raise ValueError(f"Unrecognized stage2 weight prefix: {fname}")
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
        raise ValueError(f"Unrecognized loc weight prefix: {fname}")
    sd = load_state_dict_any(weight_path)
    model.load_state_dict(sd, strict=True)
    return model

# =============================================================================
# Mean Teacher + losses
# =============================================================================
def masked_ce_loss_stage2(logits, raw_mask, ce_loss: nn.Module):
    target = torch.full_like(raw_mask, IGNORE_LABEL)
    build_tensor = BUILD_TENSOR_CPU.to(raw_mask.device)
    valid = (raw_mask != IGNORE_LABEL)
    build = valid & torch.isin(raw_mask, build_tensor)
    target[build] = raw_mask[build]
    if build.sum().item() == 0:
        return None
    return ce_loss(logits, target)

@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, decay: float):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.data.mul_(decay).add_(s.data, alpha=1.0 - decay)
    for tb, sb in zip(teacher.buffers(), student.buffers()):
        tb.copy_(sb)

def uda_pseudo_label_loss(
    student_logits, teacher_logits, build_mask,
    conf_thresh=0.85,
    temp=1.0,
    conf_thresh_destroyed=None,
    use_kl=False,
):
    t = teacher_logits[:, 1:5, :, :] / max(1e-6, float(temp))
    s = student_logits[:, 1:5, :, :]

    t_prob = torch.softmax(t, dim=1)
    conf, pseudo = torch.max(t_prob, dim=1)  # 0..3

    if conf_thresh_destroyed is not None:
        thr = torch.full_like(conf, float(conf_thresh))
        thr[pseudo == 3] = float(conf_thresh_destroyed)
        keep = conf >= thr
    else:
        keep = conf >= float(conf_thresh)

    valid = build_mask & keep
    if valid.sum().item() == 0:
        return None

    target = pseudo.clone()
    target[~valid] = -1
    loss_ce = F.cross_entropy(s, target, ignore_index=-1)

    if not use_kl:
        return loss_ce

    s_logp = F.log_softmax(s, dim=1)
    kl_map = F.kl_div(s_logp, t_prob, reduction="none").sum(dim=1)
    return loss_ce + kl_map[valid].mean()

# =============================================================================
# Stage-1 building gate
# =============================================================================
@torch.no_grad()
def predict_build_mask_from_x6(loc_model, x6, a, b, thresh):
    pre = x6[:, 0:3, :, :]
    logit = loc_model(pre)
    if isinstance(logit, (tuple, list)):
        logit = logit[0]
    if logit.ndim == 3:
        logit = logit.unsqueeze(1)
    if logit.ndim == 4 and logit.shape[1] > 1:
        logit = logit[:, 0:1, :, :]
    logit = a * logit + b
    prob = torch.sigmoid(logit)[:, 0, :, :]
    return prob >= thresh

# =============================================================================
# Metrics + pipeline eval
# =============================================================================
def _confusion_add(conf, gt_flat, pr_flat, ncls):
    idx = (gt_flat * ncls + pr_flat).astype(np.int64)
    binc = np.bincount(idx, minlength=ncls * ncls)
    conf += binc.reshape(ncls, ncls)

def f1s_from_conf(conf):
    n = conf.shape[0]
    out = []
    for c in range(n):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        out.append(float((2 * tp) / (2 * tp + fp + fn + 1e-9)))
    return out

def acc_from_conf(conf):
    return float(np.trace(conf) / (conf.sum() + 1e-9))

def prf_from_counts(tp, fp, fn):
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f = (2 * tp) / (2 * tp + fp + fn + 1e-9)
    return float(p), float(r), float(f)

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

        pred_damage = torch.argmax(logits[:, 1:5, :, :], dim=1) + 1
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
            _confusion_add(conf5, g[v], np.clip(p[v], 0, 4), 5)

    acc5 = acc_from_conf(conf5)
    f1s5 = f1s_from_conf(conf5)
    f1s_damage_1to4 = [f1s5[c] for c in [1, 2, 3, 4]]
    macro_build = float(np.mean(f1s_damage_1to4))
    loc_p, loc_r, loc_f1 = prf_from_counts(loc_tp, loc_fp, loc_fn)
    return acc5, macro_build, f1s_damage_1to4, loc_p, loc_r, loc_f1

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
            loss = masked_ce_loss_stage2(logits, raw, ce_loss)
        if loss is None:
            skipped += 1
            continue
        losses.append(float(loss.item()))
    return (float(np.mean(losses)) if losses else 1e9), skipped

# =============================================================================
# Weight selection + naming + plotting
# =============================================================================
def auto_pick_loc_weight(weights_dir: str) -> str:
    pats = [
        path.join(weights_dir, "*loc*_0_tuned_best*.pth"),
        path.join(weights_dir, "*loc*_0_*best*.pth"),
        path.join(weights_dir, "*loc*_0*.pth"),
        path.join(weights_dir, "*loc*.pth"),
    ]
    for ptn in pats:
        hits = sorted(glob.glob(ptn))
        if hits:
            return hits[0]
    return ""

def auto_pick_init_weight(weights_dir: str) -> str:
    pats = [
        path.join(weights_dir, "*cls*_0_tuned_best*.pth"),
        path.join(weights_dir, "*cls2*_0_tuned_best*.pth"),
        path.join(weights_dir, "*cls*_0_*best*.pth"),
        path.join(weights_dir, "*cls*.pth"),
    ]
    for ptn in pats:
        hits = sorted(glob.glob(ptn))
        if hits:
            return hits[0]
    return ""

def list_all_stage2_inits(weights_dir: str):
    pats = [
        "dpn92_cls*.pth",
        "res34_cls*.pth",
        "res34_cls2*.pth",
        "res50_cls*.pth",
        "se154_cls*.pth",
    ]
    files = []
    for ptn in pats:
        files += glob.glob(path.join(weights_dir, ptn))
    out, seen = [], set()
    for f in sorted(files):
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out

def make_run_tag(init_weight: str, seed: int):
    base = path.splitext(path.basename(init_weight))[0]
    return f"{base}_seed{seed}_XBDtrainTier3_to_IDABD_UDA_MeanTeacher_Gated"

def plot_f1_curves_png(metrics_csv: str, png_path: str, title: str, curve_split: str = "val_curve"):
    if not path.exists(metrics_csv):
        print(f"[PLOT] metrics_csv not found: {metrics_csv}")
        return
    rows = []
    with open(metrics_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("split", "") == curve_split:
                rows.append(row)
    if not rows:
        print(f"[PLOT] no rows for split={curve_split} in {metrics_csv}")
        return

    epochs = [int(x["epoch"]) for x in rows]
    loc_f1 = [float(x["pipeline_loc_f1"]) for x in rows]
    f1_no  = [float(x["pipeline_f1_no_damage"]) for x in rows]
    f1_mi  = [float(x["pipeline_f1_minor"]) for x in rows]
    f1_ma  = [float(x["pipeline_f1_major"]) for x in rows]
    f1_de  = [float(x["pipeline_f1_destroyed"]) for x in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loc_f1, label="Loc F1 (building)")
    plt.plot(epochs, f1_no,  label="F1 No-Damage")
    plt.plot(epochs, f1_mi,  label="F1 Minor")
    plt.plot(epochs, f1_ma,  label="F1 Major")
    plt.plot(epochs, f1_de,  label="F1 Destroyed")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(path.dirname(png_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved -> {png_path}")

def pick_mask_folder(root: str) -> str:
    for name in ["masks", "targets", "labels"]:
        d = path.join(root, name)
        if path.isdir(d):
            return d
    return ""

def make_loader(ds, batch_size, shuffle, workers, pin_memory, drop_last):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(workers),
        drop_last=bool(drop_last),
        pin_memory=bool(pin_memory),
    )
    # Only use prefetch/persistent if workers > 0
    if int(workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)

# =============================================================================
# Core
# =============================================================================
def train_and_eval_one(cfg, init_weight: str, loc_weight: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] {device}")
    print(f"[INIT ] {init_weight}")
    print(f"[LOC  ] {loc_weight}")

    run_tag = make_run_tag(init_weight, cfg["SEED"])
    os.makedirs(cfg["OUT_DIR"], exist_ok=True)

    metrics_csv = path.join(cfg["OUT_DIR"], f"{run_tag}_metrics.csv")
    curves_png  = path.join(cfg["OUT_DIR"], f"{run_tag}_f1_curves.png")
    best_ckpt   = path.join(cfg["OUT_DIR"], f"{run_tag}_best.pth")
    last_ckpt   = path.join(cfg["OUT_DIR"], f"{run_tag}_last.pth")

    ensure_csv(metrics_csv, METRICS_FIELDS, overwrite=bool(cfg["OVERWRITE_RUN_FILES"]))
    print(f"[FILES] metrics_csv -> {metrics_csv}")
    print(f"[FILES] curves_png  -> {curves_png}")

    # reproducibility
    random.seed(cfg["SEED"])
    np.random.seed(cfg["SEED"])
    torch.manual_seed(cfg["SEED"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["SEED"])

    # Stage-1 loc
    loc_model = build_loc_model_from_weight(loc_weight).to(device).eval()
    for p in loc_model.parameters():
        p.requires_grad = False

    loc_a, loc_b = 1.0, 0.0
    if cfg["LOC_PLATT"] and path.exists(cfg["LOC_PLATT"]):
        d = np.load(cfg["LOC_PLATT"])
        loc_a = float(d["a"]); loc_b = float(d["b"])
        print(f"[LOC  ] platt: {cfg['LOC_PLATT']} (a={loc_a:.6f}, b={loc_b:.6f})")
    else:
        print("[LOC  ] platt: (none) -> a=1, b=0")
    print(f"[LOC  ] thresh: {cfg['LOC_THRESH']}")

    # Source triplets
    src_triplets = []
    for root in cfg["XBD_SOURCE_ROOTS"]:
        img_dir = path.join(root, "images")
        mask_dir = pick_mask_folder(root)
        if not mask_dir:
            raise FileNotFoundError(f"[SRC] no targets/masks/labels in: {root}")
        tr = build_stage2_triplets(img_dir, mask_dir, gt_split=cfg["GT_SPLIT"], debug=bool(cfg["PAIR_DEBUG"]))
        if not tr:
            raise FileNotFoundError(f"[SRC] no triplets in {root}. Check naming in {img_dir}.")
        src_triplets += tr
    print(f"[SRC  ] total labeled triplets from XBD roots = {len(src_triplets)}")

    # Target triplets
    idabd_all = build_stage2_triplets(cfg["IDABD_IMG_DIR"], cfg["IDABD_MASK_DIR"],
                                     gt_split=cfg["GT_SPLIT"], debug=bool(cfg["PAIR_DEBUG"]))
    if not idabd_all:
        raise FileNotFoundError("[TGT] no IDA-BD triplets built.")
    tgt_train, tgt_val, tgt_test = split_list(idabd_all, cfg["VAL_RATIO"], cfg["TEST_RATIO"], seed=cfg["SEED"])
    print(f"[TGT  ] idabd total={len(idabd_all)} | train(unlabeled)={len(tgt_train)} val={len(tgt_val)} test={len(tgt_test)}")

    # Loaders
    src_ds = LabeledDamageDataset(src_triplets, crop_size=cfg["CROP"], is_train=True,
                                  fusion_p=cfg["SRC_FUSION_P"], seed=cfg["SEED"])
    tgt_u_ds = UnlabeledDamageDataset(tgt_train, crop_size=cfg["CROP"],
                                      fusion_p=cfg["TGT_FUSION_P"], seed=cfg["SEED"])
    val_ds  = LabeledDamageDataset(tgt_val,  crop_size=0, is_train=False, fusion_p=0.0, seed=cfg["SEED"])
    test_ds = LabeledDamageDataset(tgt_test, crop_size=0, is_train=False, fusion_p=0.0, seed=cfg["SEED"])

    src_ld = make_loader(src_ds, cfg["BATCH"], True,  cfg["WORKERS"], cfg["PIN_MEMORY"], drop_last=True)
    tgt_ld = make_loader(tgt_u_ds, cfg["BATCH"], True, cfg["WORKERS"], cfg["PIN_MEMORY"], drop_last=True)
    val_ld = make_loader(val_ds,  1, False, max(0, cfg["WORKERS"]), cfg["PIN_MEMORY"], drop_last=False)
    test_ld= make_loader(test_ds, 1, False, max(0, cfg["WORKERS"]), cfg["PIN_MEMORY"], drop_last=False)

    # Student + teacher
    model = build_damage_model_from_weight(init_weight).to(device)
    teacher = copy.deepcopy(model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WD"])
    ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL, reduction="mean")

    amp_on = bool(cfg["AMP"]) and (device.type == "cuda")
    if device.type == "cuda" and hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_on)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_on) if device.type == "cuda" else torch.cuda.amp.GradScaler(enabled=False)

    if amp_on and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        def amp_ctx():
            return torch.amp.autocast(device_type=device.type, enabled=True)
    else:
        def amp_ctx():
            return torch.cuda.amp.autocast(enabled=amp_on)

    # Baseline only
    if int(cfg["EPOCHS"]) <= 0:
        print("[INFO ] EPOCHS=0 -> baseline: evaluating pretrained XBD weights on IDA-BD TEST only.")
        acc5, macroB, f1sD, locP, locR, locF = eval_loader_pipeline(
            model, test_ld, device, loc_model, loc_a, loc_b, cfg["LOC_THRESH"], amp=amp_on
        )
        print("\n==================== IDA-BD TEST (PIPELINE) ====================")
        print(f"pipeline_acc(0..4)={acc5:.6f}")
        print(f"F1 Localization={locF:.6f} (P={locP:.6f}, R={locR:.6f})")
        print(f"macroF1(Damage 1..4)={macroB:.6f}")
        for i, name in enumerate(CLASS_NAMES_4):
            print(f"F1 {name:>9s}: {f1sD[i]:.6f}")
        print("===============================================================\n")
        return

    # UDA training
    best_val = 1e9
    src_iter = cycle(src_ld)
    tgt_iter = cycle(tgt_ld)
    steps = max(len(src_ld), len(tgt_ld))

    accum_steps = max(1, int(cfg.get("ACCUM_STEPS", 1)))
    print(f"[ACCUM] BATCH={cfg['BATCH']} ACCUM_STEPS={accum_steps} (effective batch={cfg['BATCH']*accum_steps})")

    for epoch in range(1, int(cfg["EPOCHS"]) + 1):
        model.train()
        teacher.eval()

        ramp = 1.0
        if int(cfg["UDA_RAMP_EPOCHS"]) > 0:
            ramp = min(1.0, epoch / float(cfg["UDA_RAMP_EPOCHS"]))
        uda_w = float(cfg["UDA_LAMBDA"]) * ramp
        src_w = float(cfg["SRC_LAMBDA"])

        losses = []
        uda_skipped = 0

        opt.zero_grad(set_to_none=True)

        for step_idx in range(steps):
            x_src, y_src, _ = next(src_iter)
            x_w, x_s, _ = next(tgt_iter)

            x_src = x_src.to(device, non_blocking=True)
            y_src = y_src.to(device, non_blocking=True)
            x_w   = x_w.to(device, non_blocking=True)
            x_s   = x_s.to(device, non_blocking=True)

            # ---- (A) supervised source loss BACKWARD FIRST (frees graph earlier)
            with amp_ctx():
                logits_src = model(x_src)
                if isinstance(logits_src, (tuple, list)):
                    logits_src = logits_src[0]
                loss_sup = masked_ce_loss_stage2(logits_src, y_src, ce_loss)
                if loss_sup is None:
                    # IMPORTANT: make a zero loss that still has a grad_fn
                    loss_sup = logits_src.sum() * 0.0

                loss_sup_scaled = (src_w * loss_sup) / float(accum_steps)

                if amp_on:
                    scaler.scale(loss_sup_scaled).backward()
                else:
                    loss_sup_scaled.backward()


            del logits_src

            # ---- (B) teacher pseudo labels on weak target (no grad)
            with torch.no_grad():
                build_mask = predict_build_mask_from_x6(loc_model, x_w, loc_a, loc_b, cfg["LOC_THRESH"])
                with amp_ctx():
                    t_logits = teacher(x_w)
                    if isinstance(t_logits, (tuple, list)):
                        t_logits = t_logits[0]

            # ---- (C) student on strong target + UDA loss BACKWARD
            with amp_ctx():
                s_logits = model(x_s)
                if isinstance(s_logits, (tuple, list)):
                    s_logits = s_logits[0]

                loss_uda = uda_pseudo_label_loss(
                s_logits, t_logits, build_mask,
                conf_thresh=float(cfg["PL_CONF"]),
                temp=float(cfg["PL_TEMP"]),
                conf_thresh_destroyed=(float(cfg["PL_CONF_DESTROYED"]) if float(cfg["PL_CONF_DESTROYED"]) > 0 else None),
                use_kl=bool(cfg["UDA_USE_KL"]),
            )

            if loss_uda is None:
                # IMPORTANT: zero loss connected to student graph
                loss_uda = s_logits.sum() * 0.0
                uda_skipped += 1

            loss_uda_scaled = (uda_w * loss_uda) / float(accum_steps)

            if amp_on:
                scaler.scale(loss_uda_scaled).backward()
            else:
                loss_uda_scaled.backward()



            del s_logits, t_logits, build_mask

            # track unscaled total for logging
            losses.append(float((src_w * loss_sup + uda_w * loss_uda).detach().cpu().item()))

            # ---- optimizer step each ACCUM_STEPS
            if (step_idx + 1) % accum_steps == 0:
                if amp_on:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()

                opt.zero_grad(set_to_none=True)
                ema_update(teacher, model, decay=float(cfg["EMA_DECAY"]))

        train_loss = float(np.mean(losses)) if losses else 1e9
        val_loss, _ = eval_val_loss(model, val_ld, device, ce_loss, amp=amp_on)

        print(f"[EPOCH {epoch:02d}/{cfg['EPOCHS']}] "
              f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f} "
              f"| src_w={src_w:.3f} uda_w={uda_w:.3f} | uda_skipped_steps={uda_skipped}")

        # save last
        torch.save({
            "epoch": epoch,
            "init_weight": init_weight,
            "state_dict": model.state_dict(),
            "config": cfg,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, last_ckpt)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "init_weight": init_weight,
                "state_dict": model.state_dict(),
                "config": cfg,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, best_ckpt)
            print(f"[SAVE ] Best -> {best_ckpt}")

        # curves eval
        curve_on = str(cfg["CURVE_EVAL_SPLIT"]).lower()
        if curve_on == "test":
            curve_loader = test_ld
            curve_split_name = "test_curve"
        else:
            curve_loader = val_ld
            curve_split_name = "val_curve"

        acc5, macroB, f1sD, locP, locR, locF = eval_loader_pipeline(
            model, curve_loader, device, loc_model, loc_a, loc_b, cfg["LOC_THRESH"], amp=amp_on
        )

        append_csv_row(metrics_csv, METRICS_FIELDS, {
            "run_tag": run_tag,
            "init_weight": init_weight,
            "loc_weight": loc_weight,
            "seed": cfg["SEED"],
            "epoch": epoch,
            "split": curve_split_name,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "pipeline_acc": acc5,
            "pipeline_loc_precision": locP,
            "pipeline_loc_recall": locR,
            "pipeline_loc_f1": locF,
            "pipeline_macroF1_damage": macroB,
            "pipeline_f1_no_damage": f1sD[0],
            "pipeline_f1_minor": f1sD[1],
            "pipeline_f1_major": f1sD[2],
            "pipeline_f1_destroyed": f1sD[3],
        })
        print(f"[CSV  ] epoch metrics -> {metrics_csv}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # final test on best
    ck = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    best_epoch = int(ck.get("epoch", cfg["EPOCHS"]))
    print(f"[LOAD ] Best for FINAL TEST: {best_ckpt} (epoch={best_epoch})")

    acc5, macroB, f1sD, locP, locR, locF = eval_loader_pipeline(
        model, test_ld, device, loc_model, loc_a, loc_b, cfg["LOC_THRESH"], amp=amp_on
    )

    print("\n==================== IDA-BD TEST (PIPELINE) ====================")
    print(f"pipeline_acc(0..4)={acc5:.6f}")
    print(f"F1 Localization={locF:.6f} (P={locP:.6f}, R={locR:.6f})")
    print(f"macroF1(Damage 1..4)={macroB:.6f}")
    for i, name in enumerate(CLASS_NAMES_4):
        print(f"F1 {name:>9s}: {f1sD[i]:.6f}")
    print("===============================================================\n")

    append_csv_row(metrics_csv, METRICS_FIELDS, {
        "run_tag": run_tag,
        "init_weight": init_weight,
        "loc_weight": loc_weight,
        "seed": cfg["SEED"],
        "epoch": best_epoch,
        "split": "test_final",
        "train_loss": "",
        "val_loss": "",
        "pipeline_acc": acc5,
        "pipeline_loc_precision": locP,
        "pipeline_loc_recall": locR,
        "pipeline_loc_f1": locF,
        "pipeline_macroF1_damage": macroB,
        "pipeline_f1_no_damage": f1sD[0],
        "pipeline_f1_minor": f1sD[1],
        "pipeline_f1_major": f1sD[2],
        "pipeline_f1_destroyed": f1sD[3],
    })

    title = f"F1 Curves ({run_tag}) split={str(cfg['CURVE_EVAL_SPLIT']).upper()}"
    curve_split = "test_curve" if str(cfg["CURVE_EVAL_SPLIT"]).lower() == "test" else "val_curve"
    plot_f1_curves_png(metrics_csv, curves_png, title=title, curve_split=curve_split)

# =============================================================================
# Entry
# =============================================================================
def main():
    cfg = CONFIG

    if float(cfg["VAL_RATIO"]) + float(cfg["TEST_RATIO"]) >= 1.0:
        raise ValueError("VAL_RATIO + TEST_RATIO must be < 1.0")

    weights_dir = cfg["WEIGHTS_DIR"]
    if not path.isdir(weights_dir):
        raise FileNotFoundError(f"WEIGHTS_DIR not found: {weights_dir}")

    loc_weight = str(cfg["LOC_WEIGHT"]).strip()
    if not loc_weight:
        loc_weight = auto_pick_loc_weight(weights_dir)
        print(f"[AUTO ] LOC_WEIGHT -> {loc_weight}")
    if not loc_weight or not path.exists(loc_weight):
        raise FileNotFoundError("LOC_WEIGHT not found. Put loc weights in ./weights or set CONFIG['LOC_WEIGHT'].")

    if bool(cfg["TRAIN_ALL"]):
        weight_files = list_all_stage2_inits(weights_dir)
        if not weight_files:
            raise FileNotFoundError(f"No stage2 cls weights found in {weights_dir}")
        print("[WEIGHTS] Stage-2 init weights:")
        for w in weight_files:
            print("  -", w)
        for init_w in weight_files:
            if path.exists(init_w):
                train_and_eval_one(cfg, init_w, loc_weight)
    else:
        init_weight = str(cfg["INIT_WEIGHT"]).strip()
        if not init_weight:
            init_weight = auto_pick_init_weight(weights_dir)
            print(f"[AUTO ] INIT_WEIGHT -> {init_weight}")
        if not init_weight or not path.exists(init_weight):
            raise FileNotFoundError("INIT_WEIGHT not found. Put cls weights in ./weights or set CONFIG['INIT_WEIGHT'].")
        train_and_eval_one(cfg, init_weight, loc_weight)

if __name__ == "__main__":
    main()
