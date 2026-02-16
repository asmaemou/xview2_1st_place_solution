#!/usr/bin/env python3
"""
================================================================================
STAGE-2 DAMAGE CLASSIFICATION (IDABD) + SUPERVISED FT + FUSION AUGMENTATION
+ DEFENSIBLE "Destroyed" (Class-4) HANDLING
+ Multiclass calibration on VAL logits (Vector Scaling) with CLASS-BALANCED sampling

PIPELINE-ONLY EVALUATION (END-TO-END) USING STAGE-1 LOCALIZATION ✅
--------------------------------------------------------------------------------
Key upgrades (for a defensible Destroyed F1):

1) Destroyed-aware crop sampling (Stage-2 TRAIN):
   - With probability focus_destroyed_p, force crops to contain class-4 pixels
   - Combats extreme class imbalance (model otherwise learns to never predict 4)

2) Stronger class-4 weighting (optional):
   - Multiply computed class weight for Destroyed by destroyed_weight_mul

3) Calibration (Vector Scaling) using CLASS-BALANCED pixels from VAL:
   - collect_val_logits_balanced() samples up to N pixels PER CLASS (1..4)
   - Ensures class-4 participates in calibration when available

4) Best checkpoint selection based on VAL loss (more defensible)

5) AMP-safe calibrated inference:
   - Vector scaling matmul is forced to FP32 under AMP (prevents Half vs Float crash)

6) Train-all produces 12 models by default:
   - Excludes the extra "se154_*_idabd_finetune_*" init weight unless
     --include_idabd_finetune is specified.

7) ✅ EARLY STOPPING (NEW):
   - Monitors VAL loss (lower is better)
   - Stops training when VAL loss does not improve for `early_stop_patience` epochs
   - Optional warmup epochs where early stopping is not applied

--------------------------------------------------------------------------------
Defaults changed (requested):
  - --epochs default is now 100
  - Early stopping enabled by default via --early_stop_patience (default 15)
    (set --early_stop_patience 0 to disable)

--------------------------------------------------------------------------------
Pipeline reported:
  Stage-1 (localization) predicts building mask on PRE image:
      p(building) = sigmoid(a*logit + b)    (Platt scaling optional)
      build_mask  = p(building) >= loc_thresh

  Stage-2 (damage) predicts damage class on building pixels only:
      damage_pred in {1..4}

  Final prediction map (0..4):
      pred_final = 0 if NOT building else damage_pred

Reported (5-class result you want):
  - F1 Localization (binary): Building vs Background
  - F1 No-Damage
  - F1 Minor
  - F1 Major
  - F1 Destroyed

Mask assumptions (POST masks)
-----------------------------
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
# Defaults
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
# CSV (PIPELINE-ONLY) — exact columns requested
# ---------------------------------------------------------------------
CSV_FIELDS = [
    "seed",
    "init_weight",
    "loc_weight",
    "loc_platt",
    "loc_thresh",

    "pipeline_acc_uncal",
    "pipeline_loc_precision_uncal",
    "pipeline_loc_recall_uncal",
    "pipeline_loc_f1_uncal",
    "pipeline_macroF1_damage_uncal",
    "pipeline_f1_no_damage_uncal",
    "pipeline_f1_minor_uncal",
    "pipeline_f1_major_uncal",
    "pipeline_f1_destroyed_uncal",

    "pipeline_acc_cal",
    "pipeline_loc_precision_cal",
    "pipeline_loc_recall_cal",
    "pipeline_loc_f1_cal",
    "pipeline_macroF1_damage_cal",
    "pipeline_f1_no_damage_cal",
    "pipeline_f1_minor_cal",
    "pipeline_f1_major_cal",
    "pipeline_f1_destroyed_cal",
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

        # try to crop with constraint
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

        # fallback: relax constraint
        for _ in range(self.crop_attempts):
            y0 = self.rng.randint(0, h - size)
            x0 = self.rng.randint(0, w - size)
            m = mask[y0:y0+size, x0:x0+size]
            build_px = int(np.isin(m, [1, 2, 3, 4]).sum())
            if build_px >= max(1, self.min_build_px // 2):
                return pre[y0:y0+size, x0:x0+size], post[y0:y0+size, x0:x0+size], m

        # final fallback random crop
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
# Stage-2 prediction helpers (AMP-safe calibrated vector scaling)
# =============================================================================
@torch.no_grad()
def predict_damage_1to4_from_logits(logits, calib=None):
    # logits: Bx5xHxW (0=background channel, 1..4 = damage)
    if calib is None:
        return torch.argmax(logits[:, 1:5, :, :], dim=1) + 1

    W, b = calib  # W: 4x4, b: 4 (float32)

    z = logits[:, 1:5, :, :]  # Bx4xHxW (might be float16 under AMP)

    # critical: do vector scaling in FP32 even if AMP is enabled
    with torch.cuda.amp.autocast(enabled=False):
        z = z.float()
        W = W.float()
        b = b.float()
        z = z.permute(0, 2, 3, 1).contiguous()        # BxHxWx4
        z = torch.matmul(z, W.t()) + b                # BxHxWx4
        z = z.permute(0, 3, 1, 2).contiguous()        # Bx4xHxW

    return torch.argmax(z, dim=1) + 1


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
# PIPELINE evaluation (Stage-1 loc -> Stage-2 damage)
# =============================================================================
@torch.no_grad()
def eval_loader_pipeline(model, loader, device, loc_model, loc_a, loc_b, loc_thresh, calib=None, amp=False):
    model.eval()
    loc_model.eval()

    conf5 = np.zeros((5, 5), dtype=np.int64)
    loc_tp = 0.0
    loc_fp = 0.0
    loc_fn = 0.0

    build_tensor = BUILD_TENSOR_CPU.to(device)

    # amp ctx (modern if available)
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

        pred_damage = predict_damage_1to4_from_logits(logits, calib=calib)

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
# VAL loss (for best checkpoint selection + early stopping)
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
# Calibration: Vector Scaling on VAL building logits (CLASS-BALANCED pixels)
# =============================================================================
@torch.no_grad()
def collect_val_logits_balanced(model, loader, device, per_class_pixels=50_000, seed=0):
    """
    Collect up to per_class_pixels from each class in {1,2,3,4} on VALID pixels.
    Returns:
      X: Nx4 logits features (float32)
      Y: N labels in {0,1,2,3}  (raw-1)
      counts: collected per class (length 4)
    """
    rng = np.random.RandomState(seed)
    model.eval()

    Xs = [[] for _ in range(4)]
    Ys = [[] for _ in range(4)]
    remaining = [int(per_class_pixels)] * 4
    collected = [0, 0, 0, 0]

    for x, raw, _ in loader:
        if all(r <= 0 for r in remaining):
            break

        x = x.to(device)
        raw = raw.to(device)

        # collect in FP32 for stability
        with torch.cuda.amp.autocast(enabled=False):
            logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            logits = logits.float()

        feat = logits[:, 1:5, :, :]  # Bx4xHxW
        feat = feat.permute(0, 2, 3, 1).contiguous().view(-1, 4)  # Nx4
        rawf = raw.view(-1)

        valid = (rawf != IGNORE_LABEL)
        rawv = rawf[valid]
        if rawv.numel() == 0:
            continue
        featv = feat[valid]

        for c_raw in [1, 2, 3, 4]:
            c = c_raw - 1
            need = remaining[c]
            if need <= 0:
                continue

            idx = torch.where(rawv == c_raw)[0]
            if idx.numel() == 0:
                continue

            take = min(need, int(idx.numel()))
            if int(idx.numel()) > take:
                pick = rng.choice(idx.detach().cpu().numpy(), size=take, replace=False)
                pick = torch.from_numpy(pick).to(device)
            else:
                pick = idx

            Xs[c].append(featv[pick].detach().cpu().numpy().astype(np.float32))
            Ys[c].append(np.full((take,), c, dtype=np.int64))
            remaining[c] -= take
            collected[c] += take

    X_list = []
    Y_list = []
    for c in range(4):
        if len(Xs[c]) == 0:
            continue
        X_list.append(np.concatenate(Xs[c], axis=0))
        Y_list.append(np.concatenate(Ys[c], axis=0))

    if not X_list:
        raise RuntimeError("Calibration collection failed: no pixels collected from VAL.")

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)

    perm = rng.permutation(X.shape[0])
    X = X[perm]
    Y = Y[perm]
    return X, Y, np.array(collected, dtype=np.int64)


def fit_vector_scaling(X: np.ndarray, Y: np.ndarray, device, lr=5e-2, epochs=10, batch_size=8192, wd=0.0):
    X_t = torch.from_numpy(X).to(device)
    Y_t = torch.from_numpy(Y).to(device)

    W = torch.eye(4, device=device, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(4, device=device, dtype=torch.float32, requires_grad=True)

    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss()

    n = X_t.shape[0]
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        steps = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_t[idx]
            yb = Y_t[idx]

            logits = xb @ W.t() + b
            loss = ce(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            steps += 1

        avg = total_loss / max(1, steps)
        print(f"[CALIB] vector-scaling epoch {ep:02d}/{epochs} loss={avg:.6f}")

    return W.detach().cpu().numpy().astype(np.float32), b.detach().cpu().numpy().astype(np.float32)


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
        raise ValueError("This pipeline-only script requires --loc_weight.")
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
        print(f"[LOC  ] platt:   {args.loc_platt}  (a={loc_a:.6f}, b={loc_b:.6f})")
    else:
        print("[LOC  ] platt:   (none) -> using a=1, b=0")

    print(f"[LOC  ] thresh:  {args.loc_thresh}")

    # -------------------------
    # Build IDABD triplets + split
    # -------------------------
    triplets = build_stage2_triplets(args.img_dir, args.mask_dir, gt_split=args.gt_split)
    if not triplets:
        raise FileNotFoundError("No (pre, post, mask) triplets found. Check paths.")

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

    # Ensure VAL has some destroyed tiles (for calibration stability)
    if args.min_val_destroyed_tiles > 0:
        val_d = [t for t in val_tr if has_destroyed(t[2])]
        if len(val_d) < args.min_val_destroyed_tiles:
            train_d = [t for t in train_tr if has_destroyed(t[2])]
            need = args.min_val_destroyed_tiles - len(val_d)
            moved = 0
            for t in train_d:
                if moved >= need:
                    break
                if len(val_tr) == 0:
                    break
                val_swap = val_tr[0]
                val_tr.remove(val_swap)
                train_tr.append(val_swap)

                if t in train_tr:
                    train_tr.remove(t)
                    val_tr.append(t)
                    moved += 1
            print(f"[SPLIT] ensured VAL destroyed tiles: {len([t for t in val_tr if has_destroyed(t[2])])} (moved={moved})")

    # Ensure TRAIN has some destroyed tiles
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

    # Modern GradScaler (removes deprecation warning)
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
    calib_path     = path.join(args.out_dir, f"{base}_idabd_stage2_calib_vector_scaling.npz")

    best_val = 1e9
    best_epoch = -1

    # early stopping state
    patience = int(args.early_stop_patience)
    warmup = int(args.early_stop_warmup)
    min_delta = float(args.early_stop_min_delta)
    no_improve = 0

    # AMP ctx
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

        # VAL loss for best selection + early stopping
        val_loss, val_skipped = eval_val_loss(model, val_ld, device, ce_loss, amp=args.amp)

        print(f"[EPOCH {epoch:03d}/{args.epochs}] "
              f"train_loss={mean_loss:.5f} (skipped={skipped}) | "
              f"val_loss={val_loss:.5f} (skipped={val_skipped})")

        # Always save "last"
        torch.save({
            "epoch": epoch,
            "init_weight": init_weight,
            "state_dict": model.state_dict(),
            "args": vars(args),
            "train_loss": mean_loss,
            "val_loss": val_loss,
        }, last_ckpt_path)

        # Best checkpoint
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
            print(f"[SAVE ] Best (val) -> {best_ckpt_path}  (best_val={best_val:.6f})")
        else:
            # Early stopping counter only after warmup
            if epoch >= warmup:
                no_improve += 1

        # Early stopping check
        if patience > 0 and epoch >= warmup and no_improve >= patience:
            print(f"[EARLY STOP] No VAL improvement for {patience} epochs "
                  f"(min_delta={min_delta}). Stopping at epoch {epoch}. "
                  f"Best was epoch {best_epoch} with val_loss={best_val:.6f}.")
            break

    # -------------------------
    # Load best, fit calibration (vector scaling) with BALANCED sampling
    # -------------------------
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["state_dict"], strict=True)
    model.eval()
    print(f"\n[LOAD ] BEST checkpoint for calibration/test: {best_ckpt_path}")

    print("[CALIB] Collecting VAL logits (CLASS-BALANCED pixels per class 1..4)...")
    X, Y, got = collect_val_logits_balanced(
        model, val_ld, device,
        per_class_pixels=args.calib_pixels_per_class,
        seed=args.seed
    )
    print(f"[CALIB] Collected per-class pixels [No,Minor,Major,Destroyed] = {got.tolist()} | total={int(X.shape[0])}")

    print(f"[CALIB] Fitting vector scaling on N={X.shape[0]} building pixels...")
    W_np, b_np = fit_vector_scaling(
        X, Y, device=device,
        lr=args.calib_lr, epochs=args.calib_epochs, batch_size=args.calib_batch, wd=args.calib_wd
    )
    np.savez(calib_path, W=W_np, b=b_np, note="vector scaling on 4-dim building logits (class-balanced sampling)")
    print(f"[CALIB] Saved -> {calib_path}")

    W_t = torch.from_numpy(W_np).to(device)
    b_t = torch.from_numpy(b_np).to(device)
    calib_damage = (W_t, b_t)

    # -------------------------
    # PIPELINE evaluation (ONLY metrics kept)
    # -------------------------
    def print_pipeline(title, acc5, macroB, f1sD, locP, locR, locF):
        print(f"\n==================== {title} ====================")
        print("END-TO-END: Stage-1 loc gating -> Stage-2 damage (VALID pixels only; mask!=255)")
        print(f"pipeline_acc(0..4)={acc5:.6f}")
        print(f"F1 Localization (Building vs Background)={locF:.6f}  (P={locP:.6f}, R={locR:.6f})")
        print(f"macroF1(Damage 1..4, end-to-end conf)={macroB:.6f}")
        print("Per-damage F1 (end-to-end conf):")
        for i in range(4):
            print(f"F1 {CLASS_NAMES_4[i]:>9s}: {f1sD[i]:.6f}")
        print("==================================================\n")

    acc_u, macroB_u, f1sD_u, locP_u, locR_u, locF_u = eval_loader_pipeline(
        model, test_ld, device,
        loc_model=loc_model, loc_a=loc_a, loc_b=loc_b, loc_thresh=args.loc_thresh,
        calib=None,
        amp=args.amp,
    )
    acc_c, macroB_c, f1sD_c, locP_c, locR_c, locF_c = eval_loader_pipeline(
        model, test_ld, device,
        loc_model=loc_model, loc_a=loc_a, loc_b=loc_b, loc_thresh=args.loc_thresh,
        calib=calib_damage,
        amp=args.amp,
    )

    print_pipeline("PIPELINE TEST (UNCALIBRATED DAMAGE + STAGE-1 LOC)",
                   acc_u, macroB_u, f1sD_u, locP_u, locR_u, locF_u)
    print_pipeline("PIPELINE TEST (CALIBRATED DAMAGE + STAGE-1 LOC)",
                   acc_c, macroB_c, f1sD_c, locP_c, locR_c, locF_c)

    # -------------------------
    # CSV append (ONE row per init_weight)
    # -------------------------
    row = {
        "seed": args.seed,
        "init_weight": init_weight,
        "loc_weight": args.loc_weight,
        "loc_platt": args.loc_platt,
        "loc_thresh": args.loc_thresh,

        "pipeline_acc_uncal": acc_u,
        "pipeline_loc_precision_uncal": locP_u,
        "pipeline_loc_recall_uncal": locR_u,
        "pipeline_loc_f1_uncal": locF_u,
        "pipeline_macroF1_damage_uncal": macroB_u,
        "pipeline_f1_no_damage_uncal": f1sD_u[0],
        "pipeline_f1_minor_uncal": f1sD_u[1],
        "pipeline_f1_major_uncal": f1sD_u[2],
        "pipeline_f1_destroyed_uncal": f1sD_u[3],

        "pipeline_acc_cal": acc_c,
        "pipeline_loc_precision_cal": locP_c,
        "pipeline_loc_recall_cal": locR_c,
        "pipeline_loc_f1_cal": locF_c,
        "pipeline_macroF1_damage_cal": macroB_c,
        "pipeline_f1_no_damage_cal": f1sD_c[0],
        "pipeline_f1_minor_cal": f1sD_c[1],
        "pipeline_f1_major_cal": f1sD_c[2],
        "pipeline_f1_destroyed_cal": f1sD_c[3],
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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--img_dir", default=IMG_DIR)
    ap.add_argument("--mask_dir", default=MASK_DIR)
    ap.add_argument("--weights_dir", default=WEIGHTS_DIR)

    ap.add_argument("--gt_split", choices=["pre", "post"], default="post")

    # train/val/test (defaults give 80/10/10)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)

    # training (UPDATED default epochs=100)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")

    # early stopping (NEW)
    ap.add_argument("--early_stop_patience", type=int, default=15,
                    help="Stop if VAL loss doesn't improve for this many epochs. Set 0 to disable.")
    ap.add_argument("--early_stop_warmup", type=int, default=5,
                    help="Do not count early-stop patience before this epoch number.")
    ap.add_argument("--early_stop_min_delta", type=float, default=0.0,
                    help="Required improvement in VAL loss to reset patience (best_val - val_loss > min_delta).")

    # crop + fusion
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--fusion_p", type=float, default=0.75)
    ap.add_argument("--min_build_px", type=int, default=50)

    # destroyed-aware sampling
    ap.add_argument("--focus_destroyed_p", type=float, default=0.6,
                    help="Probability to force crops that include Destroyed pixels (class 4).")
    ap.add_argument("--min_destroyed_px", type=int, default=50,
                    help="Minimum Destroyed pixels (class 4) required in a forced crop.")
    ap.add_argument("--crop_attempts", type=int, default=20,
                    help="How many tries to find a crop satisfying constraints.")

    # class weights
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--weight_count_images", type=int, default=200)
    ap.add_argument("--destroyed_weight_mul", type=float, default=3.0,
                    help="Multiplier for class-4 (Destroyed) weight when --use_class_weights is set.")

    # calibration (stage-2 vector scaling) — class-balanced
    ap.add_argument("--calib_pixels_per_class", type=int, default=50_000,
                    help="Max number of VAL pixels to sample PER class for vector scaling.")
    ap.add_argument("--calib_lr", type=float, default=5e-2)
    ap.add_argument("--calib_epochs", type=int, default=10)
    ap.add_argument("--calib_batch", type=int, default=8192)
    ap.add_argument("--calib_wd", type=float, default=0.0)

    # stage-2 init weights
    ap.add_argument("--init_weight", default="")
    ap.add_argument("--train_all", action="store_true")

    # by default, keep Stage-2 at 12 models (exclude idabd_finetune)
    ap.add_argument(
        "--include_idabd_finetune",
        action="store_true",
        help="If set, includes se154_*_idabd_finetune_* init weights in --train_all. "
             "Default excludes them so Stage-2 is 12 models (4 backbones x 3 seeds)."
    )

    ap.add_argument("--out_dir", default="idabd_stage2_damage_ft_checkpoints_retrained_destroyed")

    # enforce destroyed presence in splits
    ap.add_argument("--min_val_destroyed_tiles", type=int, default=1,
                    help="Ensure VAL contains at least this many tiles with class-4 pixels (for calibration stability).")
    ap.add_argument("--min_train_destroyed_tiles", type=int, default=1,
                    help="Ensure TRAIN contains at least this many tiles with class-4 pixels.")

    # Stage-1 localization for pipeline evaluation (MANDATORY)
    ap.add_argument("--loc_weight", type=str, default="",
                    help="Path to stage-1 localization weight: *_idabd_ft_best.pth (required)")
    ap.add_argument("--loc_platt", type=str, default="",
                    help="Path to stage-1 Platt params: *_idabd_platt.npz (optional)")
    ap.add_argument("--loc_thresh", type=float, default=0.5,
                    help="Threshold for stage-1 building prob (gate)")

    # CSV
    ap.add_argument("--csv_path", default="idabd_stage2_pipeline_results_DESTROYED_DEFENSIBLE.csv",
                    help="CSV path (created immediately; appended per init_weight).")
    ap.add_argument("--overwrite_csv", action="store_true",
                    help="If set, deletes csv_path at start and recreates it.")

    args = ap.parse_args()

    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    if not args.loc_weight:
        raise ValueError("This pipeline-only script requires --loc_weight.")

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ensure_csv(args.csv_path, overwrite=args.overwrite_csv)
    print(f"[CSV  ] Ready: {args.csv_path}")

    if args.train_all:
        patterns = [
            "dpn92_cls*.pth",
            "res34_cls*.pth",
            "res34_cls2*.pth",
            "res50_cls*.pth",
            "se154_cls*.pth",
        ]
        weight_files = []
        for ptn in patterns:
            weight_files += glob.glob(path.join(args.weights_dir, ptn))
        weight_files = sorted(dedup_preserve_order(weight_files))

        if not args.include_idabd_finetune:
            before = len(weight_files)
            weight_files = [w for w in weight_files if "idabd_finetune" not in path.basename(w).lower()]
            removed = before - len(weight_files)
            if removed > 0:
                print(f"[WEIGHTS] excluded idabd_finetune init weights: removed={removed}")

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


"""
Example (runs up to 100 epochs but will early-stop by default):



python train_idabd_stage2_pipeline_Fusion_earlyStop.py --train_all --val_ratio 0.1 --test_ratio 0.1 --epochs 100 --batch 2 --lr 1e-4 --wd 1e-4 --workers 2 --seed 0 --amp --use_class_weights --destroyed_weight_mul 3.0 --focus_destroyed_p 0.6 --min_destroyed_px 50 --crop_attempts 20 --calib_pixels_per_class 50000 --calib_epochs 10 --loc_weight idabd_stage1_loc_ft_checkpoints/dpn92_loc_0_tuned_best_idabd_ft_best.pth --loc_platt idabd_stage1_loc_ft_checkpoints/dpn92_loc_0_tuned_best_idabd_platt.npz --loc_thresh 0.5 --csv_path idabd_stage2_pipeline_results_DESTROYED_DEFENSIBLE.csv --overwrite_csv --early_stop_patience 10 --early_stop_warmup 5 --early_stop_min_delta 1e-4

train for 100 epoch and 
patience 10 
Disable early stopping:
  add: --early_stop_patience 0

If you WANT to include the extra se154 idabd_finetune init weight (making 13):
  add: --include_idabd_finetune
"""
