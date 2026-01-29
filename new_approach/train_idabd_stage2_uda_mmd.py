#!/usr/bin/env python3
"""
================================================================================
STAGE-2 DAMAGE CLASSIFICATION with UNSUPERVISED DOMAIN ADAPTATION (MMD)

Classic UDA setup:
  - Source domain: xBD (labeled)  -> supervised CE loss
  - Target domain: IDA-BD (unlabeled for UDA) -> MMD alignment loss

Total loss:
  L = CE_source + lambda_mmd * MMD( feat_source , feat_target )

Notes:
- We apply MMD on pooled logits[:,1:5,:,:] (damage channels only) to avoid
  architecture-dependent hooks.
- Target labels are NOT used for training loss; only for validation selection (optional).
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------
# Make sure "zoo" is importable
# ---------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
for cand in [THIS_DIR, THIS_DIR.parent, THIS_DIR.parent.parent]:
    if (cand / "zoo").is_dir():
        sys.path.insert(0, str(cand))
        break


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IGNORE_LABEL = 255
BUILD_TENSOR_CPU = torch.tensor([1, 2, 3, 4], dtype=torch.long)


# =============================================================================
# Pairing helpers (assumes *_pre_disaster / *_post_disaster naming)
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

def build_triplets(img_dir: str, mask_dir: str, gt_split: str = "post"):
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


# =============================================================================
# Datasets
# =============================================================================
class SourceLabeledStage2(Dataset):
    """Source: returns x6 + mask (for CE loss)."""
    def __init__(self, triplets, crop=0, seed=0):
        self.triplets = triplets
        self.crop = int(crop)
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
        return (pre[y0:y0+size, x0:x0+size],
                post[y0:y0+size, x0:x0+size],
                mask[y0:y0+size, x0:x0+size])

    def __getitem__(self, idx):
        pre_path, post_path, mask_path = self.triplets[idx]
        pre  = cv2.imread(pre_path, cv2.IMREAD_COLOR)
        post = cv2.imread(post_path, cv2.IMREAD_COLOR)
        if pre is None:  raise FileNotFoundError(pre_path)
        if post is None: raise FileNotFoundError(post_path)
        m = load_mask_raw(mask_path)

        if self.crop and self.crop > 0:
            pre, post, m = self._random_crop(pre, post, m, self.crop)

        pre, pad_hw = pad_to_factor(pre, 32)
        post, _ = pad_to_factor(post, 32)
        if pad_hw != (0, 0):
            m = np.pad(m, ((0, pad_hw[0]), (0, pad_hw[1])), mode="edge")

        x6 = preprocess_6ch(pre, post).float()
        y  = torch.from_numpy(m).long()
        return x6, y

class TargetUnlabeledStage2(Dataset):
    """Target: returns x6 only (mask is loaded but not returned for training)."""
    def __init__(self, triplets, crop=0, seed=0):
        self.triplets = triplets
        self.crop = int(crop)
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
        return (pre[y0:y0+size, x0:x0+size],
                post[y0:y0+size, x0:x0+size])

    def __getitem__(self, idx):
        pre_path, post_path, mask_path = self.triplets[idx]
        pre  = cv2.imread(pre_path, cv2.IMREAD_COLOR)
        post = cv2.imread(post_path, cv2.IMREAD_COLOR)
        if pre is None:  raise FileNotFoundError(pre_path)
        if post is None: raise FileNotFoundError(post_path)

        # mask only to keep same padding sizes (not used for loss)
        m = load_mask_raw(mask_path)

        if self.crop and self.crop > 0:
            pre, post = self._random_crop(pre, post, m, self.crop)

        pre, _ = pad_to_factor(pre, 32)
        post, _ = pad_to_factor(post, 32)

        x6 = preprocess_6ch(pre, post).float()
        return x6

class TargetValLabeled(Dataset):
    """Target validation: x6 + mask (only for model selection metrics)."""
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        pre_path, post_path, mask_path = self.triplets[idx]
        pre  = cv2.imread(pre_path, cv2.IMREAD_COLOR)
        post = cv2.imread(post_path, cv2.IMREAD_COLOR)
        if pre is None:  raise FileNotFoundError(pre_path)
        if post is None: raise FileNotFoundError(post_path)
        m = load_mask_raw(mask_path)

        pre, pad_hw = pad_to_factor(pre, 32)
        post, _ = pad_to_factor(post, 32)
        if pad_hw != (0, 0):
            m = np.pad(m, ((0, pad_hw[0]), (0, pad_hw[1])), mode="edge")

        x6 = preprocess_6ch(pre, post).float()
        y  = torch.from_numpy(m).long()
        return x6, y


# =============================================================================
# Checkpoint loading + model builders (same as your scripts)
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
        raise ValueError(f"Unrecognized cls weight prefix: {fname}")

    sd = load_state_dict_any(weight_path)
    model.load_state_dict(sd, strict=True)
    return model


# =============================================================================
# Loss helpers
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
# MMD (RBF kernel) on sampled features
# =============================================================================
def rbf_kernel(x, y, sigma):
    # x: NxD, y: MxD
    x2 = (x * x).sum(dim=1, keepdim=True)  # Nx1
    y2 = (y * y).sum(dim=1, keepdim=True).t()  # 1xM
    dist2 = x2 + y2 - 2.0 * (x @ y.t())
    return torch.exp(-dist2 / (2.0 * sigma * sigma + 1e-9))

def mmd_rbf(x, y, sigmas=(1.0, 2.0, 4.0, 8.0)):
    # unbiased-ish: use mean of kernels
    mmd = 0.0
    for s in sigmas:
        Kxx = rbf_kernel(x, x, s)
        Kyy = rbf_kernel(y, y, s)
        Kxy = rbf_kernel(x, y, s)
        mmd += Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
    return mmd / float(len(sigmas))

def sample_feats_from_logits(logits_5ch, max_pix=4096):
    """
    logits_5ch: Bx5xHxW
    We use damage channels 1..4 and sample pixel-wise vectors -> Nx4
    """
    z = logits_5ch[:, 1:5, :, :]              # Bx4xHxW
    z = z.permute(0, 2, 3, 1).contiguous()    # BxHxWx4
    z = z.view(-1, 4)                         # (B*H*W)x4
    n = z.shape[0]
    if n <= max_pix:
        return z
    idx = torch.randperm(n, device=z.device)[:max_pix]
    return z[idx]


# =============================================================================
# VAL loss on target labeled set (for checkpoint selection only)
# =============================================================================
@torch.no_grad()
def eval_target_val_loss(model, loader, device, ce_loss, amp=False):
    model.eval()
    losses = []
    for x6, raw in loader:
        x6 = x6.to(device)
        raw = raw.to(device)
        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            logits = model(x6)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            loss = masked_ce_loss(logits, raw, ce_loss)
        if loss is not None:
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 1e9


# =============================================================================
# Main training loop (one init weight)
# =============================================================================
def train_one(args, init_weight):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    print(f"[INIT  ] {init_weight}")

    # Source triplets (xBD)
    src_triplets = build_triplets(args.src_img_dir, args.src_mask_dir, gt_split=args.gt_split)
    if not src_triplets:
        raise FileNotFoundError("No SOURCE triplets found. Check --src_img_dir / --src_mask_dir naming.")
    # Target triplets (IDA-BD)
    tgt_triplets = build_triplets(args.tgt_img_dir, args.tgt_mask_dir, gt_split=args.gt_split)
    if not tgt_triplets:
        raise FileNotFoundError("No TARGET triplets found. Check --tgt_img_dir / --tgt_mask_dir naming.")

    # Split target into train/val (val uses labels only for selection)
    rng = random.Random(args.seed)
    rng.shuffle(tgt_triplets)
    n = len(tgt_triplets)
    n_val = max(1, int(n * args.val_ratio))
    val_tr = tgt_triplets[:n_val]
    train_tr = tgt_triplets[n_val:]
    print(f"[TARGET] total={n} | train(unlabeled)={len(train_tr)} | val(labeled)={len(val_tr)}")
    print(f"[SOURCE] total={len(src_triplets)}")

    # Datasets
    src_ds = SourceLabeledStage2(src_triplets, crop=args.crop, seed=args.seed)
    tgt_ds = TargetUnlabeledStage2(train_tr, crop=args.crop, seed=args.seed)
    val_ds = TargetValLabeled(val_tr)

    src_ld = DataLoader(src_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True)
    tgt_ld = DataLoader(tgt_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=max(0, args.workers//2),
                        pin_memory=True)

    # Loss weights (optional)
    if args.use_class_weights:
        # simple fixed weights (you can keep your previous counting-based scheme if you want)
        weight5 = torch.ones(5, dtype=torch.float32, device=device)
        weight5[4] *= args.destroyed_weight_mul
        ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL, weight=weight5)
    else:
        ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    # Model
    model = build_damage_model_from_weight(init_weight).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    os.makedirs(args.out_dir, exist_ok=True)
    base = path.splitext(path.basename(init_weight))[0]
    best_path = path.join(args.out_dir, f"{base}_idabd_stage2_mmd_best.pth")
    last_path = path.join(args.out_dir, f"{base}_idabd_stage2_mmd_last.pth")

    best_val = 1e9

    # Iterators (cycle shorter loader)
    from itertools import cycle
    tgt_iter = cycle(tgt_ld)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses_ce = []
        losses_mmd = []

        for x_src, y_src in src_ld:
            x_tgt = next(tgt_iter)

            x_src = x_src.to(device, non_blocking=True)
            y_src = y_src.to(device, non_blocking=True)
            x_tgt = x_tgt.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                logits_src = model(x_src)
                logits_tgt = model(x_tgt)
                if isinstance(logits_src, (tuple, list)): logits_src = logits_src[0]
                if isinstance(logits_tgt, (tuple, list)): logits_tgt = logits_tgt[0]

                loss_ce = masked_ce_loss(logits_src, y_src, ce_loss)
                if loss_ce is None:
                    continue

                f_src = sample_feats_from_logits(logits_src, max_pix=args.mmd_max_pixels)
                f_tgt = sample_feats_from_logits(logits_tgt, max_pix=args.mmd_max_pixels)
                loss_mmd = mmd_rbf(f_src, f_tgt, sigmas=args.mmd_sigmas)

                loss = loss_ce + args.lambda_mmd * loss_mmd

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses_ce.append(float(loss_ce.item()))
            losses_mmd.append(float(loss_mmd.item()))

        mean_ce = float(np.mean(losses_ce)) if losses_ce else 0.0
        mean_mmd = float(np.mean(losses_mmd)) if losses_mmd else 0.0

        # target val loss (for best checkpoint selection)
        val_loss = eval_target_val_loss(model, val_ld, device, ce_loss, amp=args.amp)

        print(f"[EPOCH {epoch:03d}/{args.epochs}] "
              f"CE_src={mean_ce:.6f} | MMD={mean_mmd:.6f} (λ={args.lambda_mmd}) | "
              f"VAL_target_loss={val_loss:.6f}")

        torch.save({"epoch": epoch, "init_weight": init_weight, "state_dict": model.state_dict(),
                    "args": vars(args)}, last_path)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch, "init_weight": init_weight, "state_dict": model.state_dict(),
                        "args": vars(args)}, best_path)
            print(f"[SAVE] best -> {best_path}")

    print(f"[DONE] best checkpoint: {best_path}")


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

    # Source (xBD)
    ap.add_argument("--src_img_dir", required=True)
    ap.add_argument("--src_mask_dir", required=True)

    # Target (IDA-BD)
    ap.add_argument("--tgt_img_dir", required=True)
    ap.add_argument("--tgt_mask_dir", required=True)

    ap.add_argument("--gt_split", choices=["pre", "post"], default="post")

    # Target split for val selection
    ap.add_argument("--val_ratio", type=float, default=0.1)

    # Training
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")

    # Crop
    ap.add_argument("--crop", type=int, default=512)

    # Class weights (optional)
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--destroyed_weight_mul", type=float, default=3.0)

    # MMD
    ap.add_argument("--lambda_mmd", type=float, default=0.1)
    ap.add_argument("--mmd_max_pixels", type=int, default=4096)
    ap.add_argument("--mmd_sigmas", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])

    # Init weights
    ap.add_argument("--weights_dir", default="weights")
    ap.add_argument("--init_weight", default="")
    ap.add_argument("--train_all", action="store_true")

    ap.add_argument("--out_dir", default="idabd_stage2_mmd_uda_checkpoints")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
        if not weight_files:
            raise FileNotFoundError(f"No cls weights found in {args.weights_dir}")
        for w in weight_files:
            train_one(args, w)
    else:
        if not args.init_weight:
            raise ValueError("Provide --init_weight <path> OR use --train_all")
        if not path.exists(args.init_weight):
            raise FileNotFoundError(args.init_weight)
        train_one(args, args.init_weight)


if __name__ == "__main__":
    main()
"""
python train_idabd_stage2_uda_mmd.py --src_img_dir ./xbd_source/images --src_mask_dir ./xbd_source/masks --tgt_img_dir ./idabd/images --tgt_mask_dir ./idabd/masks --gt_split post --val_ratio 0.1 --epochs 20 --batch 2 --lr 1e-4 --amp --lambda_mmd 0.1 --mmd_max_pixels 4096 --train_all --weights_dir ./weights --out_dir ./idabd_stage2_mmd_uda_checkpoints
"""