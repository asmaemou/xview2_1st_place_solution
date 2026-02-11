#!/usr/bin/env python3
"""
Single-result evaluation for IDABD Stage-2 models.

Default behavior (no args):
- Loads all *_idabd_stage2_ft_best.pth from OUT_DIR
- Runs an ENSEMBLE by averaging logits across models
- Applies calibration per-model if matching npz exists (optional; enabled by default)
- Uses GT building gating by default (damage-only), so loc metrics = NaN
- Writes ONE row to CSV.

Options:
- --mode ensemble (default) or --mode best
- --best_metric macroF1_damage_cal | f1_destroyed_cal | acc_cal | macroF1_damage_uncal | ...
- --disable_calib to evaluate uncalibrated only
- --loc_weight to do full pipeline gating (Stage-1 loc -> Stage-2 damage), otherwise GT building gating
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
for cand in [THIS_DIR, THIS_DIR.parent, THIS_DIR.parent.parent, THIS_DIR.parent.parent.parent]:
    if (cand / "zoo").is_dir():
        sys.path.insert(0, str(cand))
        break


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
IMG_DIR = "../idabd/images"
MASK_DIR = "../idabd/masks"
OUT_DIR = "../idabd_stage2_damage_ft_checkpoints_retrained_destroyed"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IGNORE_LABEL = 255
CLASS_NAMES_4 = ["No-Damage", "Minor", "Major", "Destroyed"]
BUILD_TENSOR_CPU = torch.tensor([1, 2, 3, 4], dtype=torch.long)

CSV_FIELDS_SINGLE = [
    "mode",
    "num_models",
    "out_dir",
    "loc_weight",
    "loc_platt",
    "loc_thresh",
    "used_calib",

    "acc",
    "loc_precision",
    "loc_recall",
    "loc_f1",
    "macroF1_damage",
    "f1_no_damage",
    "f1_minor",
    "f1_major",
    "f1_destroyed",
]


# =============================================================================
# Pairing: (pre, post, mask) by tile_id
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
# Dataset (EVAL): no augmentation, no cropping
# =============================================================================
class IdaBDEval(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        pre_path, post_path, mask_path = self.triplets[idx]

        pre  = cv2.imread(pre_path, cv2.IMREAD_COLOR)
        post = cv2.imread(post_path, cv2.IMREAD_COLOR)
        if pre is None:
            raise FileNotFoundError(pre_path)
        if post is None:
            raise FileNotFoundError(post_path)

        m = load_mask_raw(mask_path)

        pre, pad_hw = pad_to_factor(pre, 32)
        post, _ = pad_to_factor(post, 32)
        if pad_hw != (0, 0):
            m = np.pad(m, ((0, pad_hw[0]), (0, pad_hw[1])), mode="edge")

        x = preprocess_6ch(pre, post)
        y = torch.from_numpy(m).long()
        return x.float(), y.long(), path.basename(post_path)


# =============================================================================
# Checkpoint loading helpers
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
    return strip_prefixes(sd), ckpt


def infer_backbone_from_name(p: str) -> str:
    b = path.basename(p).lower()
    for key in ["dpn92", "res34", "res50", "se154"]:
        if b.startswith(key) or (f"_{key}" in b) or (key in b):
            return key
    return ""


def build_damage_model(backbone_key: str):
    import zoo.models as zm
    if backbone_key == "dpn92":
        return zm.Dpn92_Unet_Double(pretrained=None)
    if backbone_key == "res34":
        return zm.Res34_Unet_Double(pretrained=False)
    if backbone_key == "res50":
        return zm.SeResNext50_Unet_Double(pretrained=None)
    if backbone_key == "se154":
        return zm.SeNet154_Unet_Double(pretrained=None)
    raise ValueError(f"Cannot infer backbone. Got: {backbone_key}")


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

    sd, _ = load_state_dict_any(weight_path)
    model.load_state_dict(sd, strict=True)
    return model


# =============================================================================
# Calibration (vector scaling)
# =============================================================================
def load_vector_scaling_npz(npz_path: str, device):
    d = np.load(npz_path)
    W = torch.from_numpy(d["W"]).to(device)
    b = torch.from_numpy(d["b"]).to(device)
    return (W, b)


def apply_vector_scaling_4ch(logits_5ch, calib):
    """
    logits_5ch: Bx5xHxW where channels 1..4 are damage logits
    returns: scaled damage logits Bx4xHxW
    """
    W, b = calib
    z = logits_5ch[:, 1:5, :, :]  # Bx4xHxW
    with torch.cuda.amp.autocast(enabled=False):
        z = z.float()
        W = W.float()
        b = b.float()
        z = z.permute(0, 2, 3, 1).contiguous()        # BxHxWx4
        z = torch.matmul(z, W.t()) + b                # BxHxWx4
        z = z.permute(0, 3, 1, 2).contiguous()        # Bx4xHxW
    return z


# =============================================================================
# Stage-1 loc helper
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
# Metrics
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
# Evaluate one model (returns confusion + loc counts)
# =============================================================================
@torch.no_grad()
def eval_model_collect_conf(model, loader, device, loc_model=None, loc_a=1.0, loc_b=0.0, loc_thresh=0.5,
                            calib=None, use_calib=False, amp=False):
    model.eval()
    if loc_model is not None:
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
        valid = (raw != IGNORE_LABEL)

        if loc_model is None:
            build_mask = valid & torch.isin(raw, build_tensor)  # GT building mask
        else:
            build_mask = predict_build_mask_from_x6(loc_model, x, loc_a, loc_b, loc_thresh)

        with amp_ctx():
            logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

        # damage prediction
        if use_calib and calib is not None:
            dmg_logits = apply_vector_scaling_4ch(logits, calib=calib)  # Bx4xHxW
            pred_damage = torch.argmax(dmg_logits, dim=1) + 1
        else:
            pred_damage = torch.argmax(logits[:, 1:5, :, :], dim=1) + 1

        pred_final = torch.zeros_like(pred_damage)
        pred_final[build_mask] = pred_damage[build_mask]

        # loc metrics only if loc_model exists
        gt_build = valid & torch.isin(raw, build_tensor)
        pr_build = valid & build_mask
        if loc_model is not None:
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

    return conf5, loc_tp, loc_fp, loc_fn


# =============================================================================
# Ensemble eval (mean logits, with optional per-model calibration)
# =============================================================================
@torch.no_grad()
def eval_ensemble(loader, device, models, calibs, use_calib=True,
                  loc_model=None, loc_a=1.0, loc_b=0.0, loc_thresh=0.5, amp=False):
    for m in models:
        m.eval()
    if loc_model is not None:
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
        valid = (raw != IGNORE_LABEL)

        if loc_model is None:
            build_mask = valid & torch.isin(raw, build_tensor)
        else:
            build_mask = predict_build_mask_from_x6(loc_model, x, loc_a, loc_b, loc_thresh)

        # accumulate ensemble damage logits (4ch)
        sum_logits4 = None

        for i, model in enumerate(models):
            with amp_ctx():
                logits5 = model(x)
                if isinstance(logits5, (tuple, list)):
                    logits5 = logits5[0]

            if use_calib and calibs[i] is not None:
                logits4 = apply_vector_scaling_4ch(logits5, calibs[i])  # Bx4xHxW
            else:
                logits4 = logits5[:, 1:5, :, :].float()

            if sum_logits4 is None:
                sum_logits4 = logits4
            else:
                sum_logits4 = sum_logits4 + logits4

        mean_logits4 = sum_logits4 / float(len(models))
        pred_damage = torch.argmax(mean_logits4, dim=1) + 1

        pred_final = torch.zeros_like(pred_damage)
        pred_final[build_mask] = pred_damage[build_mask]

        # loc metrics only if loc_model exists
        gt_build = valid & torch.isin(raw, build_tensor)
        pr_build = valid & build_mask
        if loc_model is not None:
            loc_tp += float((pr_build & gt_build).sum().item())
            loc_fp += float((pr_build & (~gt_build)).sum().item())
            loc_fn += float(((~pr_build) & gt_build).sum().item())

        gt_np = raw.detach().cpu().numpy().astype(np.int64)
        pr_np = pred_final.detach().cpu().numpy().astype(np.int64)

        for b in range(gt_np.shape[0]):
            g = gt_np[b]
            p = pr_np[b]
            v = (g != IGNORE_LABEL)
            if v.sum() == 0:
                continue
            g_valid = g[v]
            p_valid = np.clip(p[v], 0, 4)
            _confusion_add(conf5, g_valid, p_valid, 5)

    return conf5, loc_tp, loc_fp, loc_fn


def conf_to_metrics(conf5, loc_tp, loc_fp, loc_fn, has_loc):
    acc5 = acc_from_conf(conf5)
    f1s5 = f1s_from_conf(conf5)
    f1s_damage = [f1s5[c] for c in [1, 2, 3, 4]]
    macro_damage = float(np.mean(f1s_damage))

    if has_loc:
        loc_p, loc_r, loc_f1 = prf_from_counts(loc_tp, loc_fp, loc_fn)
    else:
        loc_p, loc_r, loc_f1 = float("nan"), float("nan"), float("nan")

    return acc5, loc_p, loc_r, loc_f1, macro_damage, f1s_damage


def write_single_csv(csv_path: str, row: dict):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS_SINGLE, extrasaction="ignore")
        w.writeheader()
        w.writerow(row)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default=IMG_DIR)
    ap.add_argument("--mask_dir", default=MASK_DIR)
    ap.add_argument("--gt_split", choices=["pre", "post"], default="post")

    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--csv_out", default="idabd_stage2_EVAL_SINGLE.csv")

    ap.add_argument("--mode", choices=["ensemble", "best"], default="ensemble",
                    help="ensemble = mean logits across all models; best = pick best single checkpoint by --best_metric.")
    ap.add_argument("--best_metric", default="macroF1_damage",
                    help="Used only if --mode best. One of: macroF1_damage, f1_destroyed, acc")
    ap.add_argument("--disable_calib", action="store_true",
                    help="If set, ignores vector scaling npz and evaluates uncalibrated only.")

    # optional stage-1 loc
    ap.add_argument("--loc_weight", type=str, default="")
    ap.add_argument("--loc_platt", type=str, default="")
    ap.add_argument("--loc_thresh", type=float, default=0.5)

    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    # Split like training
    triplets = build_stage2_triplets(args.img_dir, args.mask_dir, gt_split=args.gt_split)
    if not triplets:
        raise FileNotFoundError("No (pre, post, mask) triplets found. Check --img_dir and --mask_dir.")

    rng = random.Random(args.seed)
    rng.shuffle(triplets)
    n = len(triplets)

    n_test = max(1, int(n * args.test_ratio))
    n_val  = max(1, int(n * args.val_ratio))
    if n_test + n_val >= n:
        n_test = max(1, min(n_test, n - 2))
        n_val  = max(1, min(n_val,  n - 1 - n_test))

    test_tr = triplets[:n_test]
    ds = IdaBDEval(test_tr)
    ld = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # Optional stage-1 localization
    loc_model = None
    loc_a, loc_b = 1.0, 0.0
    if args.loc_weight:
        if not path.exists(args.loc_weight):
            raise FileNotFoundError(f"--loc_weight not found: {args.loc_weight}")
        loc_model = build_loc_model_from_weight(args.loc_weight).to(device).eval()
        for p in loc_model.parameters():
            p.requires_grad = False

        if args.loc_platt:
            if not path.exists(args.loc_platt):
                raise FileNotFoundError(f"--loc_platt not found: {args.loc_platt}")
            d = np.load(args.loc_platt)
            loc_a = float(d["a"])
            loc_b = float(d["b"])
            print(f"[LOC] enabled | platt a={loc_a:.6f}, b={loc_b:.6f} | thresh={args.loc_thresh}")
        else:
            print(f"[LOC] enabled | no platt | thresh={args.loc_thresh}")
    else:
        print("[LOC] disabled -> damage-only eval (GT building mask gating)")

    # Find checkpoints
    ckpts = sorted(glob.glob(path.join(args.out_dir, "*_idabd_stage2_ft_best.pth")))
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoints found in {args.out_dir}. Expected pattern *_idabd_stage2_ft_best.pth"
        )
    print(f"[CKPT] found {len(ckpts)} checkpoints.")

    # Load all models + optional calibs
    models = []
    calibs = []
    used_calib_any = False

    for ckpt_path in ckpts:
        sd, ckpt_obj = load_state_dict_any(ckpt_path)
        backbone = infer_backbone_from_name(ckpt_path)
        if not backbone and isinstance(ckpt_obj, dict) and "init_weight" in ckpt_obj:
            backbone = infer_backbone_from_name(str(ckpt_obj["init_weight"]))
        if not backbone:
            raise ValueError(f"Could not infer backbone from: {ckpt_path}")

        m = build_damage_model(backbone).to(device)
        m.load_state_dict(sd, strict=True)
        models.append(m)

        base = ckpt_path.replace("_idabd_stage2_ft_best.pth", "")
        calib_npz = base + "_idabd_stage2_calib_vector_scaling.npz"
        if (not args.disable_calib) and path.exists(calib_npz):
            calibs.append(load_vector_scaling_npz(calib_npz, device=device))
            used_calib_any = True
        else:
            calibs.append(None)

    # ---- mode: ensemble ----
    if args.mode == "ensemble":
        conf5, loc_tp, loc_fp, loc_fn = eval_ensemble(
            ld, device, models, calibs,
            use_calib=(not args.disable_calib),
            loc_model=loc_model, loc_a=loc_a, loc_b=loc_b, loc_thresh=args.loc_thresh,
            amp=args.amp
        )
        acc, locP, locR, locF, macroF1, f1s_damage = conf_to_metrics(
            conf5, loc_tp, loc_fp, loc_fn, has_loc=(loc_model is not None)
        )

        row = {
            "mode": "ensemble",
            "num_models": len(models),
            "out_dir": args.out_dir,
            "loc_weight": args.loc_weight or "",
            "loc_platt": args.loc_platt or "",
            "loc_thresh": args.loc_thresh,
            "used_calib": (used_calib_any and (not args.disable_calib)),

            "acc": acc,
            "loc_precision": locP,
            "loc_recall": locR,
            "loc_f1": locF,
            "macroF1_damage": macroF1,
            "f1_no_damage": f1s_damage[0],
            "f1_minor": f1s_damage[1],
            "f1_major": f1s_damage[2],
            "f1_destroyed": f1s_damage[3],
        }

        print("\n===== SINGLE RESULT (ENSEMBLE) =====")
        print(f"acc(0..4):          {acc:.6f}")
        print(f"macroF1(damage1..4):{macroF1:.6f}")
        print(f"F1 No/Minor/Major/Destroyed: {', '.join(f'{x:.4f}' for x in f1s_damage)}")
        print("===================================\n")

        write_single_csv(args.csv_out, row)
        print(f"[CSV] wrote: {args.csv_out}")
        return

    # ---- mode: best single checkpoint ----
    # Evaluate each model, pick best by metric
    best_idx = -1
    best_val = -1e18
    best_metrics = None

    for i, ckpt_path in enumerate(ckpts):
        conf5, loc_tp, loc_fp, loc_fn = eval_model_collect_conf(
            models[i], ld, device,
            loc_model=loc_model, loc_a=loc_a, loc_b=loc_b, loc_thresh=args.loc_thresh,
            calib=calibs[i], use_calib=(calibs[i] is not None and (not args.disable_calib)),
            amp=args.amp
        )
        acc, locP, locR, locF, macroF1, f1s_damage = conf_to_metrics(
            conf5, loc_tp, loc_fp, loc_fn, has_loc=(loc_model is not None)
        )

        if args.best_metric == "macroF1_damage":
            val = macroF1
        elif args.best_metric == "f1_destroyed":
            val = f1s_damage[3]
        elif args.best_metric == "acc":
            val = acc
        else:
            raise ValueError("--best_metric must be one of: macroF1_damage, f1_destroyed, acc")

        if val > best_val:
            best_val = val
            best_idx = i
            best_metrics = (acc, locP, locR, locF, macroF1, f1s_damage, ckpt_path)

    acc, locP, locR, locF, macroF1, f1s_damage, best_ckpt = best_metrics

    row = {
        "mode": f"best({args.best_metric})",
        "num_models": 1,
        "out_dir": args.out_dir,
        "loc_weight": args.loc_weight or "",
        "loc_platt": args.loc_platt or "",
        "loc_thresh": args.loc_thresh,
        "used_calib": (calibs[best_idx] is not None and (not args.disable_calib)),

        "acc": acc,
        "loc_precision": locP,
        "loc_recall": locR,
        "loc_f1": locF,
        "macroF1_damage": macroF1,
        "f1_no_damage": f1s_damage[0],
        "f1_minor": f1s_damage[1],
        "f1_major": f1s_damage[2],
        "f1_destroyed": f1s_damage[3],
    }

    print("\n===== SINGLE RESULT (BEST MODEL) =====")
    print(f"best_ckpt:          {best_ckpt}")
    print(f"acc(0..4):          {acc:.6f}")
    print(f"macroF1(damage1..4):{macroF1:.6f}")
    print(f"F1 No/Minor/Major/Destroyed: {', '.join(f'{x:.4f}' for x in f1s_damage)}")
    print("=====================================\n")

    write_single_csv(args.csv_out, row)
    print(f"[CSV] wrote: {args.csv_out}")


if __name__ == "__main__":
    main()
