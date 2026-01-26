#!/usr/bin/env python3
"""
================================================================================
EVAL: ENSEMBLE STAGE-1 LOCALIZATION + ENSEMBLE STAGE-2 DAMAGE (CALIBRATED ONLY)
IDABD end-to-end pipeline

Stage-1 ensemble (localization):
  - Load multiple loc models
  - For each model: p(building) = sigmoid(a*logit + b)  (Platt scaling if available)
  - Ensemble by averaging probabilities across loc models
  - build_mask = mean_prob >= loc_thresh

Stage-2 ensemble (damage):
  - Load multiple stage-2 fine-tuned damage models
  - Apply per-model Vector Scaling calibration (W,b) on logits (classes 1..4)
  - Convert to softmax probs over 4 damage classes
  - Ensemble by averaging probabilities across models
  - damage_pred = argmax(mean_probs) + 1

Final prediction:
  pred_final = 0 if not building else damage_pred

Reports + CSV (CALIBRATED ONLY):
  - pipeline_acc_cal
  - pipeline_loc_precision_cal / recall / f1
  - pipeline_macroF1_damage_cal
  - pipeline_f1_{no_damage, minor, major, destroyed}_cal

Mask assumptions (POST masks):
  0=background, 1=NoDamage, 2=Minor, 3=Major, 4=Destroyed, 255=ignore

UPDATES INCLUDED (requested):
  ✅ add --stage2_glob (so you can control which stage-2 models are loaded)
  ✅ FP32-safe calibration math under AMP (prevents Half vs Float crashes)
  ✅ optional: enforce stage2_max_models (default 12) to avoid accidental 13th model
  ✅ default split aligned with your training script (80/10/10): test_ratio=0.1

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


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
IMG_DIR = "../idabd/images"
MASK_DIR = "../idabd/masks"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IGNORE_LABEL = 255
CLASS_NAMES_4 = ["No-Damage", "Minor", "Major", "Destroyed"]
BUILD_TENSOR_CPU = torch.tensor([1, 2, 3, 4], dtype=torch.long)


# ---------------------------------------------------------------------
# CSV (CALIBRATED ONLY)
# ---------------------------------------------------------------------
CSV_FIELDS = [
    "seed",
    "loc_dir",
    "loc_models_used",
    "stage2_dir",
    "stage2_models_used",
    "loc_glob",
    "stage2_glob",
    "loc_thresh",
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
# Dataset (no augmentation; evaluation only)
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
# Build models from weights (prefix-based)
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
        raise ValueError(f"Unrecognized stage-2 weight prefix: {fname}")

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
        raise ValueError(f"Unrecognized stage-1 weight prefix: {fname}")

    sd = load_state_dict_any(weight_path)
    model.load_state_dict(sd, strict=True)
    return model


# =============================================================================
# Metrics helpers
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
# Stage-1 ensemble: average building probabilities -> ONE gate mask
# =============================================================================
@torch.no_grad()
def predict_build_mask_ensemble(loc_pack, x6, thresh, use_amp: bool):
    pre = x6[:, 0:3, :, :]  # normalized pre

    prob_sum = None
    for (m, a, b) in loc_pack:
        with torch.cuda.amp.autocast(enabled=(use_amp and pre.is_cuda)):
            logit = m(pre)
            if isinstance(logit, (tuple, list)):
                logit = logit[0]
            if logit.ndim == 3:
                logit = logit.unsqueeze(1)
            if logit.ndim == 4 and logit.shape[1] > 1:
                logit = logit[:, 0:1, :, :]
            logit = a * logit + b
            prob = torch.sigmoid(logit)[:, 0, :, :]  # BxHxW

        prob_sum = prob if prob_sum is None else (prob_sum + prob)

    prob_mean = prob_sum / float(len(loc_pack))
    return prob_mean >= thresh


# =============================================================================
# Stage-2 ensemble (calibrated): average softmax probs over 4 classes
# IMPORTANT: calibration math is forced to FP32 to avoid AMP dtype crashes
# =============================================================================
@torch.no_grad()
def predict_damage_ensemble_cal(stage2_pack, x6, use_amp: bool):
    prob_sum = None

    for (m, W, b) in stage2_pack:
        # 1) forward can be AMP
        with torch.cuda.amp.autocast(enabled=(use_amp and x6.is_cuda)):
            logits = m(x6)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            z = logits[:, 1:5, :, :]  # Bx4xHxW (may be fp16)

        # 2) calibration + softmax in FP32 (stable + defensible)
        with torch.cuda.amp.autocast(enabled=False):
            z = z.float()
            Wf = W.float()
            bf = b.float()

            z2 = z.permute(0, 2, 3, 1).contiguous()          # BxHxWx4
            z2 = torch.matmul(z2, Wf.t()) + bf               # BxHxWx4
            z2 = z2.permute(0, 3, 1, 2).contiguous()         # Bx4xHxW
            probs = F.softmax(z2, dim=1)                     # FP32

        prob_sum = probs if prob_sum is None else (prob_sum + probs)

    prob_mean = prob_sum / float(len(stage2_pack))
    pred = torch.argmax(prob_mean, dim=1) + 1  # 1..4
    return pred


# =============================================================================
# End-to-end evaluation: Stage-1 ensemble gate + Stage-2 calibrated ensemble
# =============================================================================
@torch.no_grad()
def eval_pipeline_ensemble_cal(loader, device, loc_pack, stage2_pack, loc_thresh, use_amp: bool, debug_destroyed: bool):
    conf5 = np.zeros((5, 5), dtype=np.int64)

    loc_tp = 0.0
    loc_fp = 0.0
    loc_fn = 0.0
    build_tensor = BUILD_TENSOR_CPU.to(device)

    for x6, raw, _ in loader:
        x6 = x6.to(device, non_blocking=True)
        raw = raw.to(device, non_blocking=True)

        build_mask = predict_build_mask_ensemble(loc_pack, x6, loc_thresh, use_amp=use_amp)
        pred_damage = predict_damage_ensemble_cal(stage2_pack, x6, use_amp=use_amp)

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

    f1s_damage = [f1s5[c] for c in [1, 2, 3, 4]]
    macro_damage = float(np.mean(f1s_damage))
    loc_p, loc_r, loc_f1 = prf_from_counts(loc_tp, loc_fp, loc_fn)

    if debug_destroyed:
        gt = int(conf5[4, :].sum())
        pred = int(conf5[:, 4].sum())
        tp = int(conf5[4, 4])
        fp = pred - tp
        fn = gt - tp
        print(f"[DESTROYED DEBUG] GT={gt}  PRED={pred}  TP={tp}  FP={fp}  FN={fn}")

    return acc5, loc_p, loc_r, loc_f1, macro_damage, f1s_damage


# =============================================================================
# Load packs (loc + stage2)
# =============================================================================
def load_loc_pack(loc_dir: str, device, loc_glob="*_idabd_ft_best.pth"):
    weights = sorted(glob.glob(path.join(loc_dir, loc_glob)))
    if not weights:
        raise FileNotFoundError(f"No Stage-1 loc weights found in {loc_dir} with {loc_glob}")

    pack = []
    for w in weights:
        base = path.basename(w)
        stem = base[:-len("_idabd_ft_best.pth")] if base.endswith("_idabd_ft_best.pth") else path.splitext(base)[0]
        platt = path.join(loc_dir, f"{stem}_idabd_platt.npz")

        model = build_loc_model_from_weight(w).to(device).eval()
        for p in model.parameters():
            p.requires_grad = False

        a, b = 1.0, 0.0
        if path.exists(platt):
            d = np.load(platt)
            a = float(d["a"])
            b = float(d["b"])

        pack.append((model, a, b))

    return pack, weights

def load_stage2_pack(stage2_dir: str, device, stage2_glob="*_idabd_stage2_ft_best.pth", stage2_max_models: int = 12):
    ckpts = sorted(glob.glob(path.join(stage2_dir, stage2_glob)))
    if not ckpts:
        raise FileNotFoundError(f"No Stage-2 best checkpoints found in {stage2_dir} ({stage2_glob})")

    # optional: keep only first N (useful if an extra 13th appears)
    if stage2_max_models > 0 and len(ckpts) > stage2_max_models:
        print(f"[STAGE2] WARNING: found {len(ckpts)} models; keeping first {stage2_max_models} (sorted).")
        ckpts = ckpts[:stage2_max_models]

    pack = []
    for ck in ckpts:
        base = path.basename(ck)
        stem = base[:-len("_idabd_stage2_ft_best.pth")]
        calib = path.join(stage2_dir, f"{stem}_idabd_stage2_calib_vector_scaling.npz")
        if not path.exists(calib):
            raise FileNotFoundError(f"Missing calibration for {ck} -> expected {calib}")

        model = build_damage_model_from_weight(ck).to(device).eval()
        for p in model.parameters():
            p.requires_grad = False

        d = np.load(calib)
        W = torch.from_numpy(d["W"].astype(np.float32)).to(device)
        b = torch.from_numpy(d["b"].astype(np.float32)).to(device)

        pack.append((model, W, b))

    return pack, ckpts


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default=IMG_DIR)
    ap.add_argument("--mask_dir", default=MASK_DIR)
    ap.add_argument("--gt_split", choices=["pre", "post"], default="post")

    # IMPORTANT: align defaults with your training script (80/10/10)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)

    # ensemble inputs
    ap.add_argument("--loc_dir", required=True, help="Folder containing *_idabd_ft_best.pth and optional *_idabd_platt.npz")
    ap.add_argument("--loc_glob", default="*_idabd_ft_best.pth", help="Pattern to select Stage-1 loc weights in loc_dir")

    ap.add_argument("--stage2_dir", required=True, help="Folder containing *_idabd_stage2_ft_best.pth and *_calib_vector_scaling.npz")
    ap.add_argument("--stage2_glob", default="*_idabd_stage2_ft_best.pth", help="Pattern to select Stage-2 best weights in stage2_dir")

    # optional safety: avoid accidental 13th model
    ap.add_argument("--stage2_max_models", type=int, default=12,
                    help="If >0, cap how many Stage-2 models are loaded (after sorting). Use 0 to disable cap.")

    ap.add_argument("--loc_thresh", type=float, default=0.5)

    # loader
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true")

    # csv
    ap.add_argument("--csv_path", default="idabd_pipeline_ENSEMBLE_LOC+DMG_CAL_ONLY.csv")
    ap.add_argument("--overwrite_csv", action="store_true")

    ap.add_argument("--debug_destroyed", action="store_true")

    args = ap.parse_args()
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ensure_csv(args.csv_path, overwrite=args.overwrite_csv)
    print(f"[CSV   ] Ready: {args.csv_path}")

    # Build triplets + split (same style as training scripts)
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
    print(f"[DATA  ] total={n} | train={len(train_tr)} | val={len(val_tr)} | test={len(test_tr)}")

    test_ds = IdaBDEval(test_tr)
    test_ld = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=max(0, args.workers), pin_memory=True
    )

    # Load ensembles
    loc_pack, loc_used = load_loc_pack(args.loc_dir, device=device, loc_glob=args.loc_glob)
    print(f"[LOC   ] models: {len(loc_used)}")
    for w in loc_used:
        print("         -", w)

    stage2_pack, stage2_used = load_stage2_pack(
        args.stage2_dir, device=device,
        stage2_glob=args.stage2_glob,
        stage2_max_models=args.stage2_max_models
    )
    print(f"[STAGE2] models: {len(stage2_used)}")
    for w in stage2_used:
        print("         -", w)

    # Eval
    acc5, locP, locR, locF, macroD, f1sD = eval_pipeline_ensemble_cal(
        test_ld, device,
        loc_pack=loc_pack,
        stage2_pack=stage2_pack,
        loc_thresh=args.loc_thresh,
        use_amp=args.amp,
        debug_destroyed=args.debug_destroyed
    )

    print("\n==================== ENSEMBLED PIPELINE (CALIBRATED ONLY) ====================")
    print("END-TO-END: Stage-1 ensemble gate -> Stage-2 calibrated ensemble damage")
    print(f"pipeline_acc(0..4)={acc5:.6f}")
    print(f"F1 Localization (Building vs Background)={locF:.6f}  (P={locP:.6f}, R={locR:.6f})")
    print(f"macroF1(Damage 1..4, end-to-end conf)={macroD:.6f}")
    for i in range(4):
        print(f"F1 {CLASS_NAMES_4[i]:>9s}: {f1sD[i]:.6f}")
    print("=============================================================================\n")

    row = {
        "seed": args.seed,
        "loc_dir": args.loc_dir,
        "loc_models_used": len(loc_used),
        "stage2_dir": args.stage2_dir,
        "stage2_models_used": len(stage2_used),
        "loc_glob": args.loc_glob,
        "stage2_glob": args.stage2_glob,
        "loc_thresh": args.loc_thresh,

        "pipeline_acc_cal": acc5,
        "pipeline_loc_precision_cal": locP,
        "pipeline_loc_recall_cal": locR,
        "pipeline_loc_f1_cal": locF,
        "pipeline_macroF1_damage_cal": macroD,
        "pipeline_f1_no_damage_cal": f1sD[0],
        "pipeline_f1_minor_cal": f1sD[1],
        "pipeline_f1_major_cal": f1sD[2],
        "pipeline_f1_destroyed_cal": f1sD[3],
    }
    append_csv_row(args.csv_path, row)
    print(f"[CSV   ] Appended -> {args.csv_path}")


if __name__ == "__main__":
    main()


"""
Example (recommended, matches 80/10/10 split and caps stage2 at 12 models):

python eval_idabd_pipeline_ensemble_loc_and_damage_cal_only.py --stage2_dir idabd_stage2_damage_ft_checkpoints_retrained_destroyed --loc_dir idabd_stage1_loc_ft_checkpoints --loc_thresh 0.5 --val_ratio 0.1 --test_ratio 0.1 --seed 0 --stage2_glob "*_idabd_stage2_ft_best.pth" --stage2_max_models 12 --csv_path idabd_pipeline_ENSEMBLE_LOC+DMG_CAL_ONLY.csv --overwrite_csv --amp --debug_destroyed

If you want ONLY dpn92 stage2:
  --stage2_glob "dpn92*_idabd_stage2_ft_best.pth"
"""
