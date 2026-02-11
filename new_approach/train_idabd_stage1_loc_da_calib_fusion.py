#!/usr/bin/env python3
"""

Below is a single, self-contained training script for Stage-1 localization only on IDABD, using:

Fusion augmentation (Canny edges + HistEq + Unsharp + CLAHE, fused into the input)

Domain adaptation via supervised probability calibration (Platt scaling / logistic regression on IDABD val pixels)

Fine-tuning from xBD pretrained localization weights in your weights/ folder

Saving your own new checkpoints (fine-tuned model + calibration params)
================================================================================
STAGE-1 LOCALIZATION (IDABD) + DOMAIN ADAPTATION
Supervised Probability Calibration (Platt Scaling) + Fusion Augmentation
================================================================================

Goal
----
- Stage-1 = building localization (binary mask).
- Fine-tune xBD (source) pretrained localization checkpoints on IDABD (target).
- Add robustness with Fusion Augmentation (Canny + HistEq + Unsharp + CLAHE).
- Reduce domain shift with Supervised Probability Calibration:
    Fit a 1D Logistic Regression on IDABD validation pixels:
        p_cal = sigmoid(a * logit + b)
    Save calibration params alongside the fine-tuned checkpoint.

Important IDABD labeling detail
------------------------------
- IDABD PRE masks contain only {0,255} -> NOT building-labeled.
- IDABD POST masks contain {0,1,2,3,...} -> real labels.
So by default this script:
  - uses PRE images as input (cleaner)
  - uses POST masks as ground truth (building = mask>0, ignore=255)

Outputs
-------
- Fine-tuned weights saved to:  <out_dir>/<checkpoint_name>_idabd_ft_best.pth
- Calibration params saved to:  <out_dir>/<checkpoint_name>_idabd_platt.npz

Run (recommended)
-----------------
# Fine-tune ALL localization checkpoints in ./weights
python train_idabd_stage1_loc_da_calib_fusion.py --train_all --epochs 5 --batch 2 --lr 2e-4

# Fine-tune ONE checkpoint
python train_idabd_stage1_loc_da_calib_fusion.py --init_weight weights/dpn92_loc_0_tuned_best.pth --epochs 8

# Evaluate with calibrated probabilities on val set (always done at end)
# Optionally sweep thresholds:
python train_idabd_stage1_loc_da_calib_fusion.py --init_weight weights/res34_loc_0_1_best.pth --thresh_sweep

================================================================================
"""

import os
from os import path
import sys
from pathlib import Path
import glob
import argparse
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For supervised probability calibration (Platt scaling)
from sklearn.linear_model import LogisticRegression


# -----------------------------------------------------------------------------
# Make sure "zoo" is importable even when running from /new_approach
# -----------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
for cand in [THIS_DIR, THIS_DIR.parent, THIS_DIR.parent.parent]:
    if (cand / "zoo").is_dir():
        sys.path.insert(0, str(cand))
        break


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
IMG_DIR = "../idabd/images"
MASK_DIR = "../idabd/masks"
WEIGHTS_DIR = "weights"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =============================================================================
# Utility: pairing PRE images with POST masks (by tile_id)
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
    # expected: <tile_id>_<gt_split>_disaster.png
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        cand = path.join(mask_dir, f"{tile_id}_{gt_split}_disaster{ext}")
        if path.exists(cand):
            return cand
    return None

def build_pairs(img_dir: str, mask_dir: str, input_split: str, gt_split: str):
    imgs = list_split_files(img_dir, input_split)
    pairs = []
    for img_path in imgs:
        tid = tile_id_from_name(img_path)
        m = find_mask(mask_dir, tid, gt_split)
        if m is not None:
            pairs.append((img_path, m))
    return pairs


# =============================================================================
# Fusion augmentation (Canny + HistEq + Unsharp + CLAHE)
# =============================================================================
def canny_edges_rgb(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges3

def hist_equalize_rgb(img_bgr: np.ndarray) -> np.ndarray:
    # Equalize Y channel in YCrCb (better than equalizing each channel)
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
    """
    Fuse original + enhanced variants into one image.
    You can adjust weights if you want.
    """
    e = canny_edges_rgb(img_bgr)
    h = hist_equalize_rgb(img_bgr)
    u = unsharp_mask_rgb(img_bgr, amount=1.0, radius=3)
    c = clahe_rgb(img_bgr, clip=2.0, tile=(8, 8))

    # Weighted fusion (simple + stable)
    # Keep original dominant; add enhancements lightly
    fused = (
        0.60 * img_bgr.astype(np.float32) +
        0.10 * e.astype(np.float32) +
        0.10 * h.astype(np.float32) +
        0.10 * u.astype(np.float32) +
        0.10 * c.astype(np.float32)
    )
    return np.clip(fused, 0, 255).astype(np.uint8)


# =============================================================================
# Preprocess + mask loading
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

def unpad(arr2d, pad_hw):
    pad_h, pad_w = pad_hw
    if pad_h == 0 and pad_w == 0:
        return arr2d
    h, w = arr2d.shape[:2]
    return arr2d[: h - pad_h, : w - pad_w]

def preprocess_rgb(img_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))  # CHW
    return torch.from_numpy(img)  # 3xHxW

def load_mask_binary(mask_path: str):
    """
    Building GT = (mask > 0) & (mask != 255)
    Ignore = (mask == 255)
    """
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(mask_path)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = m.astype(np.int32)
    valid = (m != 255)
    gt = (m > 0) & valid
    return gt.astype(np.float32), valid.astype(np.float32), m


# =============================================================================
# Dataset (train: random crop + fusion aug; val: full image, no aug)
# =============================================================================
class IdaBDLocDataset(Dataset):
    def __init__(self, pairs, crop_size=512, is_train=True, fusion_p=0.5, seed=0):
        self.pairs = pairs
        self.crop_size = int(crop_size) if crop_size else 0
        self.is_train = is_train
        self.fusion_p = float(fusion_p)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.pairs)

    def _random_crop(self, img, gt, valid, size):
        h, w = img.shape[:2]
        if h < size or w < size:
            # pad first if too small
            pad_h = max(0, size - h)
            pad_w = max(0, size - w)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            gt = np.pad(gt, ((0, pad_h), (0, pad_w)), mode="edge")
            valid = np.pad(valid, ((0, pad_h), (0, pad_w)), mode="edge")
            h, w = img.shape[:2]

        y0 = self.rng.randint(0, h - size)
        x0 = self.rng.randint(0, w - size)
        img_c = img[y0:y0+size, x0:x0+size]
        gt_c = gt[y0:y0+size, x0:x0+size]
        v_c  = valid[y0:y0+size, x0:x0+size]
        return img_c, gt_c, v_c

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)

        gt, valid, _raw = load_mask_binary(mask_path)

        # Train-time fusion augmentation
        if self.is_train and (self.rng.random() < self.fusion_p):
            img = fusion_augment(img)

        # Random crop for train
        if self.is_train and self.crop_size > 0:
            img, gt, valid = self._random_crop(img, gt, valid, self.crop_size)

        # Pad to factor 32 (safe for all backbones)
        img, pad_hw = pad_to_factor(img, 32)
        gt = np.pad(gt, ((0, pad_hw[0]), (0, pad_hw[1])), mode="edge") if (pad_hw != (0, 0)) else gt
        valid = np.pad(valid, ((0, pad_hw[0]), (0, pad_hw[1])), mode="edge") if (pad_hw != (0, 0)) else valid

        x = preprocess_rgb(img)  # 3xHxW
        y = torch.from_numpy(gt).unsqueeze(0)      # 1xHxW
        v = torch.from_numpy(valid).unsqueeze(0)   # 1xHxW

        return x.float(), y.float(), v.float(), path.basename(img_path)


# =============================================================================
# Model loading (zoo.models localization variants)
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
    # PyTorch 2.6+: weights_only defaults to True -> may fail with old ckpt
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

    # IMPORTANT: pass pretrained=None/False to avoid downloads
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
# Loss + metrics
# =============================================================================
def masked_bce_with_logits(logits, target, valid, pos_weight=None):
    """
    logits: Bx1xHxW
    target: Bx1xHxW float {0,1}
    valid : Bx1xHxW float {0,1}
    """
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    loss_map = bce(logits, target)
    loss = (loss_map * valid).sum() / (valid.sum() + 1e-9)
    return loss

@torch.no_grad()
def prf_counts(logits, target, valid, thresh=0.5, calib=None):
    """
    calib: (a,b) for Platt scaling on logits (optional)
    """
    if calib is not None:
        a, b = calib
        logits = a * logits + b

    prob = torch.sigmoid(logits)
    pred = (prob >= thresh).float()

    pred = pred * valid
    tgt  = target * valid

    tp = (pred * tgt).sum().item()
    fp = (pred * (1 - tgt)).sum().item()
    fn = ((1 - pred) * tgt).sum().item()
    return tp, fp, fn

def prf_from_counts(tp, fp, fn):
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return p, r, f


# =============================================================================
# Supervised probability calibration (Platt scaling) on IDABD val pixels
# =============================================================================
@torch.no_grad()
def fit_platt_scaling(model, loader, device, max_pixels=500_000, seed=0):
    """
    Collect (logit, label) pairs from validation set and fit LogisticRegression:
        p = sigmoid(a*logit + b)
    """
    rng = np.random.RandomState(seed)
    xs = []
    ys = []

    collected = 0
    model.eval()

    for x, y, v, _name in loader:
        x = x.to(device)
        y = y.to(device)
        v = v.to(device)

        logits = model(x)  # Bx1xHxW
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.ndim == 4 and logits.shape[1] > 1:
            logits = logits[:, 0:1, :, :]

        # Flatten valid pixels
        log_flat = logits.detach().cpu().numpy().reshape(-1)
        y_flat   = y.detach().cpu().numpy().reshape(-1)
        v_flat   = v.detach().cpu().numpy().reshape(-1)

        idx = np.where(v_flat > 0.5)[0]
        if idx.size == 0:
            continue

        # Sample a subset from this batch
        need = max_pixels - collected
        if need <= 0:
            break

        take = min(need, idx.size)
        pick = rng.choice(idx, size=take, replace=False) if idx.size > take else idx

        xs.append(log_flat[pick])
        ys.append(y_flat[pick])
        collected += take

        if collected >= max_pixels:
            break

    if collected < 10_000:
        print(f"[CALIB] Warning: only collected {collected} pixels for calibration (still fitting).")

    X = np.concatenate(xs).reshape(-1, 1).astype(np.float32)
    Y = np.concatenate(ys).astype(np.int32)

    # Logistic regression = Platt scaling
    lr = LogisticRegression(
        solver="lbfgs",
        max_iter=200,
        class_weight="balanced"
    )
    lr.fit(X, Y)

    a = float(lr.coef_.ravel()[0])
    b = float(lr.intercept_.ravel()[0])

    return a, b


# =============================================================================
# Threshold sweep utility (optional)
# =============================================================================
@torch.no_grad()
def eval_loader(model, loader, device, thresh=0.5, calib=None):
    model.eval()
    tp = fp = fn = 0.0
    for x, y, v, _n in loader:
        x = x.to(device)
        y = y.to(device)
        v = v.to(device)
        logits = model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.ndim == 4 and logits.shape[1] > 1:
            logits = logits[:, 0:1, :, :]

        a, b = prf_counts(logits, y, v, thresh=thresh, calib=calib)
        tp += a; fp += b; fn += (0.0)  # wrong, careful
    # Above line is wrong: prf_counts returns tp,fp,fn; fix:
    return None

@torch.no_grad()
def eval_loader_counts(model, loader, device, thresh=0.5, calib=None):
    model.eval()
    tp = fp = fn = 0.0
    for x, y, v, _n in loader:
        x = x.to(device)
        y = y.to(device)
        v = v.to(device)
        logits = model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.ndim == 4 and logits.shape[1] > 1:
            logits = logits[:, 0:1, :, :]
        a, b, c = prf_counts(logits, y, v, thresh=thresh, calib=calib)
        tp += a; fp += b; fn += c
    return tp, fp, fn

def sweep_thresholds(model, loader, device, thresholds, calib=None):
    best = (-1.0, None, None)  # f1, t, (p,r)
    for t in thresholds:
        tp, fp, fn = eval_loader_counts(model, loader, device, thresh=t, calib=calib)
        p, r, f = prf_from_counts(tp, fp, fn)
        if f > best[0]:
            best = (f, t, (p, r))
    return best


# =============================================================================
# Training loop (single checkpoint)
# =============================================================================
def train_one_checkpoint(args, init_weight):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] {device}")
    print(f"[INIT]  {init_weight}")

    # Build pairs (pre images + post masks by default)
    pairs = build_pairs(args.img_dir, args.mask_dir, args.input_split, args.gt_split)
    if not pairs:
        raise FileNotFoundError("No (image, mask) pairs found. Check paths/splits.")

    # Train/val split
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_val = max(1, int(n * args.val_ratio))
    val_pairs = pairs[:n_val]
    tr_pairs  = pairs[n_val:]

    print(f"[DATA] pairs total={n} | train={len(tr_pairs)} | val={len(val_pairs)}")
    print(f"[MODE] input_split={args.input_split} | gt_split={args.gt_split}")

    # Datasets/loaders
    train_ds = IdaBDLocDataset(
        tr_pairs,
        crop_size=args.crop,
        is_train=True,
        fusion_p=args.fusion_p,
        seed=args.seed
    )
    val_ds = IdaBDLocDataset(
        val_pairs,
        crop_size=0,          # evaluate on full image
        is_train=False,
        fusion_p=0.0,
        seed=args.seed
    )

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=max(0, args.workers // 2), pin_memory=True)

    # Model
    model = build_loc_model_from_weight(init_weight)
    model.to(device)

    # Optional: freeze encoder for first epoch? (off by default)
    if args.freeze_encoder:
        for name, p in model.named_parameters():
            # crude heuristic: freeze early encoder blocks
            if name.startswith(("conv1", "conv2", "conv3", "conv4", "conv5")):
                p.requires_grad = False
        print("[TRAIN] Encoder frozen (conv1..conv5)")

    # Pos weight (helps recall when buildings are rare)
    # If you want, keep it off by default (pos_weight=None).
    pos_weight = None
    if args.use_pos_weight:
        # approximate using a few train batches
        pos = 0.0
        neg = 0.0
        with torch.no_grad():
            for i, (_x, _y, _v, _n) in enumerate(train_ld):
                y = _y.numpy().reshape(-1)
                v = _v.numpy().reshape(-1)
                idx = v > 0.5
                yy = y[idx]
                pos += float((yy > 0.5).sum())
                neg += float((yy <= 0.5).sum())
                if i >= 5:
                    break
        if pos > 0:
            pw = neg / (pos + 1e-9)
            pos_weight = torch.tensor([pw], dtype=torch.float32, device=device)
            print(f"[LOSS] Using pos_weight={pw:.4f}")
        else:
            print("[LOSS] pos_weight requested but pos pixels not found in sampled batches.")

    # Optimizer
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.wd)

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))

    # Output paths
    os.makedirs(args.out_dir, exist_ok=True)
    base = path.splitext(path.basename(init_weight))[0]
    best_ckpt_path = path.join(args.out_dir, f"{base}_idabd_ft_best.pth")
    last_ckpt_path = path.join(args.out_dir, f"{base}_idabd_ft_last.pth")
    calib_path = path.join(args.out_dir, f"{base}_idabd_platt.npz")

    # Train
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for x, y, v, _name in train_ld:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            v = v.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp)):
                logits = model(x)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                if logits.ndim == 4 and logits.shape[1] > 1:
                    logits = logits[:, 0:1, :, :]
                loss = masked_bce_with_logits(logits, y, v, pos_weight=pos_weight)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.item()))

        # Validate (uncalibrated)
        tp, fp, fn = eval_loader_counts(model, val_ld, device, thresh=args.thresh, calib=None)
        p, r, f1 = prf_from_counts(tp, fp, fn)

        print(f"[EPOCH {epoch:02d}/{args.epochs}] "
              f"loss={np.mean(losses):.5f} | val P={p:.4f} R={r:.4f} F1={f1:.4f} (t={args.thresh})")

        # Save last
        torch.save({
            "epoch": epoch,
            "init_weight": init_weight,
            "state_dict": model.state_dict(),
            "args": vars(args),
            "val_f1": f1,
            "val_p": p,
            "val_r": r,
        }, last_ckpt_path)

        # Save best
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "epoch": epoch,
                "init_weight": init_weight,
                "state_dict": model.state_dict(),
                "args": vars(args),
                "val_f1": f1,
                "val_p": p,
                "val_r": r,
            }, best_ckpt_path)
            print(f"[SAVE] Best checkpoint -> {best_ckpt_path}")

    # -----------------------------------------------------------------------------
    # Supervised probability calibration on IDABD val (Platt scaling)
    # -----------------------------------------------------------------------------
    print("\n[CALIB] Fitting supervised probability calibration (Platt scaling) on IDABD val pixels...")
    a, b = fit_platt_scaling(model, val_ld, device, max_pixels=args.calib_pixels, seed=args.seed)
    np.savez(calib_path, a=a, b=b, note="p_cal = sigmoid(a*logit + b)")
    print(f"[CALIB] Saved -> {calib_path}  (a={a:.6f}, b={b:.6f})")

    # Evaluate calibrated
    tp, fp, fn = eval_loader_counts(model, val_ld, device, thresh=args.thresh, calib=(a, b))
    p, r, f1 = prf_from_counts(tp, fp, fn)
    print(f"[CALIB] val P={p:.4f} R={r:.4f} F1={f1:.4f} (t={args.thresh})")

    if args.thresh_sweep:
        thresholds = [0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
        best_f1, best_t, (best_p, best_r) = sweep_thresholds(model, val_ld, device, thresholds, calib=(a, b))
        print("========== THRESHOLD SWEEP (CALIBRATED) ==========")
        print(f"Best F1: {best_f1:.6f} at threshold {best_t}")
        print(f"Precision: {best_p:.6f} | Recall: {best_r:.6f}")
        print("Tried thresholds:", thresholds)
        print("=================================================\n")

    return best_ckpt_path, calib_path


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--img_dir", default=IMG_DIR)
    ap.add_argument("--mask_dir", default=MASK_DIR)
    ap.add_argument("--weights_dir", default=WEIGHTS_DIR)

    # Data splits for IDABD
    ap.add_argument("--input_split", choices=["pre", "post"], default="pre",
                    help="Which images to train on. Recommended: pre")
    ap.add_argument("--gt_split", choices=["pre", "post"], default="post",
                    help="Which masks provide the labels. Recommended: post")

    # Training
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_ratio", type=float, default=0.2)

    # Patch/crop (train only)
    ap.add_argument("--crop", type=int, default=512,
                    help="Train crop size. Use 0 to train on full 1024x1024.")

    # Fusion augmentation controls
    ap.add_argument("--fusion_p", type=float, default=0.5,
                    help="Probability of applying fusion augmentation per sample during training.")

    # Loss/optimization options
    ap.add_argument("--use_pos_weight", action="store_true",
                    help="Use approximate pos_weight to boost recall (often helps).")
    ap.add_argument("--freeze_encoder", action="store_true",
                    help="Freeze encoder conv1..conv5 (optional).")
    ap.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")

    # Eval
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--thresh_sweep", action="store_true")

    # Calibration
    ap.add_argument("--calib_pixels", type=int, default=500_000,
                    help="Max val pixels to sample for Platt scaling.")

    # Which weights to train
    ap.add_argument("--init_weight", default="",
                    help="Path to ONE localization checkpoint to fine-tune.")
    ap.add_argument("--train_all", action="store_true",
                    help="Fine-tune ALL localization checkpoints found in weights_dir.")

    ap.add_argument("--out_dir", default="idabd_stage1_loc_ft_checkpoints")

    args = ap.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Find weight files
    if args.train_all:
        patterns = ["dpn92_loc*.pth", "res34_loc*.pth", "res50_loc*.pth", "se154_loc*.pth"]
        weight_files = []
        for ptn in patterns:
            weight_files += glob.glob(path.join(args.weights_dir, ptn))
        weight_files = sorted(weight_files)
        if not weight_files:
            raise FileNotFoundError(f"No localization weights found in {args.weights_dir}")
        print("[WEIGHTS] Will fine-tune ALL localization checkpoints:")
        for w in weight_files:
            print("  -", w)

        for w in weight_files:
            train_one_checkpoint(args, w)

    else:
        if not args.init_weight:
            raise ValueError("Provide --init_weight <path> OR use --train_all")
        if not path.exists(args.init_weight):
            raise FileNotFoundError(args.init_weight)
        train_one_checkpoint(args, args.init_weight)


if __name__ == "__main__":
    main()
