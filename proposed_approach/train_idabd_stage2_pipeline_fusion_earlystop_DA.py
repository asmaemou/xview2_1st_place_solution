"""
Zero-args runnable:
  cd proposed_approach/scripts
  python train_idabd_stage2_pipeline_fusion_earlystop.py

What it does (default behavior):
- Auto-finds:
  - IDABD img_dir / mask_dir
  - weights_dir (stage-2 init weights)
  - stage-1 localization checkpoint (loc_weight)
  - optional platt params (loc_platt)
- If --init_weight is NOT provided -> runs ALL found stage-2 init weights (12 by default)
- Fine-tunes stage-2 on IDABD with:
  - fusion augmentation (train only)
  - destroyed-aware crops (train only)
  - early stopping on VAL loss
- Fits vector scaling calibration on VAL logits (class-balanced pixels)
- Reports and logs pipeline results (uncal + cal) on TEST split
"""
from __future__ import annotations

import os
from os import path
import sys
import glob
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = (SCRIPT_DIR / ".." / "..").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure "zoo" is importable (xView2 solution package)
for cand in [REPO_ROOT, REPO_ROOT.parent, SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent]:
    if (cand / "zoo").is_dir():
        if str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
        break

# ------------------------------------------------------------
# Imports from your modular code
# ------------------------------------------------------------
from proposed_approach.models.xview2_builders import (
    build_damage_model_from_weight,
    build_loc_model_from_weight,
)
from proposed_approach.train.dataset_stage2_damage_fusion import (
    build_stage2_triplets,
    IdaBDStage2DamageFusion,
)
from proposed_approach.train.split_destroyed import ensure_destroyed_in_splits
from proposed_approach.train.class_weights_stage2 import compute_class_weights_from_triplets
from proposed_approach.train.fit_stage2_earlystop import train_stage2_earlystop
from proposed_approach.train.vector_scaling import collect_val_logits_balanced, fit_vector_scaling
from proposed_approach.train.pipeline_eval_stage2 import eval_loader_pipeline

IGNORE_LABEL = 255

CLASS_NAMES_4 = ["No-Damage", "Minor", "Major", "Destroyed"]

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
# ------------------------------------------------------------
# Minimal CSV helpers (so script always runs)
# ------------------------------------------------------------
def ensure_csv(csv_path: str, fields: list[str], overwrite: bool = False):
    if not csv_path:
        return
    out_dir = path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if overwrite and path.exists(csv_path):
        os.remove(csv_path)
    if not path.exists(csv_path):
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()

def append_csv_row(csv_path: str, fields: list[str], row: dict):
    if not csv_path:
        return
    import csv
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writerow(row)


# ------------------------------------------------------------
# Auto-find helpers (same spirit as your previous scripts)
# ------------------------------------------------------------
def dedup_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _candidate_dirs(*rel_paths: str) -> list[str]:
    roots = [
        SCRIPT_DIR,
        SCRIPT_DIR.parent,
        SCRIPT_DIR.parent.parent,
        REPO_ROOT,
        REPO_ROOT.parent,
        Path.cwd(),
        Path.cwd().parent,
    ]
    out = []
    for r in roots:
        for rel in rel_paths:
            d = (r / rel).resolve()
            if d.is_dir():
                out.append(str(d))
    return dedup_preserve_order(out)


def auto_defaults_for_dataset():
    img_dirs = _candidate_dirs(
        "idabd/images",
        "../idabd/images",
        "../../idabd/images",
        "data/idabd/images",
        "dataset/idabd/images",
    )
    mask_dirs = _candidate_dirs(
        "idabd/masks",
        "../idabd/masks",
        "../../idabd/masks",
        "data/idabd/masks",
        "dataset/idabd/masks",
    )
    img = img_dirs[0] if img_dirs else ""
    msk = mask_dirs[0] if mask_dirs else ""
    return img, msk


def auto_defaults_for_weights_dir():
    wdirs = _candidate_dirs(
        "weights",
        "../weights",
        "../../weights",
        "checkpoints/weights",
        "models/weights",
    )
    return wdirs[0] if wdirs else ""


def auto_find_first_file(dir_candidates: list[str], patterns: list[str]) -> str:
    for d in dir_candidates:
        for ptn in patterns:
            hits = sorted(glob.glob(path.join(d, ptn)))
            if hits:
                return str(Path(hits[0]).resolve())
    return ""


def auto_find_all_files(dir_candidates: list[str], patterns: list[str]) -> list[str]:
    files = []
    for d in dir_candidates:
        for ptn in patterns:
            files += glob.glob(path.join(d, ptn))
    files = [str(Path(f).resolve()) for f in files]
    return sorted(dedup_preserve_order(files))


def auto_find_stage1_loc_weight() -> str:
    loc_dirs = _candidate_dirs(
        "idabd_stage1_loc_ft_checkpoints",
        "../idabd_stage1_loc_ft_checkpoints",
        "../../idabd_stage1_loc_ft_checkpoints",
        "two_stage_pipeline/idabd_stage1_loc_ft_checkpoints",
        "../two_stage_pipeline/idabd_stage1_loc_ft_checkpoints",
        "checkpoints/idabd_stage1_loc_ft_checkpoints",
        "stage1_loc_checkpoints",
        "../stage1_loc_checkpoints",
    )
    patterns = ["*_idabd_ft_best.pth", "*ft_best*.pth", "*best*.pth", "*.pth"]
    return auto_find_first_file(loc_dirs, patterns)


def auto_find_stage1_platt(loc_weight_path: str) -> str:
    if not loc_weight_path:
        return ""
    d = str(Path(loc_weight_path).resolve().parent)
    base = Path(loc_weight_path).name

    repls = [
        base.replace("_idabd_ft_best.pth", "_idabd_platt.npz"),
        base.replace("_ft_best.pth", "_platt.npz"),
        base.replace(".pth", "_platt.npz"),
    ]
    for r in repls:
        cand = path.join(d, r)
        if path.exists(cand):
            return str(Path(cand).resolve())

    hits = sorted(glob.glob(path.join(d, "*platt*.npz")))
    return str(Path(hits[0]).resolve()) if hits else ""


def auto_find_stage2_init_weights(weights_dir: str, include_idabd_finetune: bool) -> list[str]:
    patterns = ["dpn92_cls*.pth", "res34_cls*.pth", "res34_cls2*.pth", "res50_cls*.pth", "se154_cls*.pth"]
    files = auto_find_all_files([weights_dir], patterns)
    if not include_idabd_finetune:
        files = [w for w in files if "idabd_finetune" not in path.basename(w).lower()]
    return files


# ------------------------------------------------------------
# Main train/eval for one init weight
# ------------------------------------------------------------
def train_eval_one(args, init_weight: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] {device}")
    print(f"[INIT ] {init_weight}")

    # ---- Stage-1 Loc
    loc_model = build_loc_model_from_weight(args.loc_weight).to(device).eval()
    for p in loc_model.parameters():
        p.requires_grad = False

    loc_a, loc_b = 1.0, 0.0
    if args.loc_platt and path.exists(args.loc_platt):
        d = np.load(args.loc_platt)
        loc_a, loc_b = float(d["a"]), float(d["b"])
        print(f"[LOC  ] platt: {args.loc_platt} (a={loc_a:.6f}, b={loc_b:.6f})")
    else:
        print("[LOC  ] platt: (none) -> using a=1, b=0")

    # ---- Triplets + split
    triplets = build_stage2_triplets(args.img_dir, args.mask_dir, gt_split=args.gt_split)
    if not triplets:
        raise FileNotFoundError("No (pre, post, mask) triplets found. Check img_dir/mask_dir.")

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

    if args.min_val_destroyed_tiles > 0 or args.min_train_destroyed_tiles > 0:
        train_tr, val_tr, train_d, val_d = ensure_destroyed_in_splits(
            train_tr, val_tr,
            min_train_destroyed=args.min_train_destroyed_tiles,
            min_val_destroyed=args.min_val_destroyed_tiles,
            seed=args.seed,
        )
        print(f"[SPLIT] destroyed tiles train={train_d} val={val_d}")

    print(f"[DATA ] total={n} | train={len(train_tr)} | val={len(val_tr)} | test={len(test_tr)}")

    # ---- Loss (optional weights)
    if args.use_class_weights:
        w4, counts = compute_class_weights_from_triplets(train_tr, max_images=args.weight_count_images, seed=args.seed)
        w4[3] *= float(args.destroyed_weight_mul)
        print(f"[WGT  ] counts(1..4)={counts.tolist()}")
        print(f"[WGT  ] w(1..4)={w4.tolist()}  destroyed_mul={args.destroyed_weight_mul}")
        weight5 = np.ones(5, dtype=np.float32)
        weight5[1:5] = w4
        ce_loss = nn.CrossEntropyLoss(
            ignore_index=IGNORE_LABEL,
            reduction="mean",
            weight=torch.from_numpy(weight5).to(device)
        )
    else:
        ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL, reduction="mean")

    # ---- Loaders
    train_ds = IdaBDStage2DamageFusion(
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
    val_ds = IdaBDStage2DamageFusion(val_tr, crop_size=0, is_train=False, fusion_p=0.0, min_build_px=0, seed=args.seed)
    test_ds = IdaBDStage2DamageFusion(test_tr, crop_size=0, is_train=False, fusion_p=0.0, min_build_px=0, seed=args.seed)

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=max(0, args.workers // 2), pin_memory=True)
    test_ld  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=max(0, args.workers // 2), pin_memory=True)

    # ---- Stage-2 model train (early stop)
    model = build_damage_model_from_weight(init_weight).to(device)

    os.makedirs(args.out_dir, exist_ok=True)
    base = path.splitext(path.basename(init_weight))[0]

    best_ckpt_path, _ = train_stage2_earlystop(
        model=model,
        train_ld=train_ld,
        val_ld=val_ld,
        device=device,
        ce_loss=ce_loss,
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
        amp=args.amp,
        out_dir=args.out_dir,
        base_name=base,
        early_stop_patience=args.early_stop_patience,
        early_stop_warmup=args.early_stop_warmup,
        early_stop_min_delta=args.early_stop_min_delta,
    )

    # ---- Load best
    best = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"], strict=True)
    model.eval()
    print(f"[LOAD ] BEST for calib/test: {best_ckpt_path}")

    # ---- Vector scaling calibration
    calib_path = path.join(args.out_dir, f"{base}_idabd_stage2_calib_vector_scaling.npz")
    print("[CALIB] Collecting balanced VAL logits...")
    X, Y, got = collect_val_logits_balanced(
        model, val_ld, device,
        per_class_pixels=args.calib_pixels_per_class,
        seed=args.seed
    )
    print(f"[CALIB] Collected per-class [No,Minor,Major,Destroyed]={got.tolist()} total={int(X.shape[0])}")

    print("[CALIB] Fitting vector scaling...")
    W_np, b_np = fit_vector_scaling(
        X, Y, device=device,
        lr=args.calib_lr, epochs=args.calib_epochs, batch_size=args.calib_batch, wd=args.calib_wd
    )
    np.savez(calib_path, W=W_np, b=b_np, note="vector scaling on 4-dim building logits (balanced sampling)")
    print(f"[CALIB] Saved -> {calib_path}")

    calib = (torch.from_numpy(W_np).to(device), torch.from_numpy(b_np).to(device))

    # ---- Pipeline eval
    acc_u, macro_u, f1_u, locP_u, locR_u, locF_u = eval_loader_pipeline(
        model, test_ld, device,
        loc_model=loc_model, loc_a=loc_a, loc_b=loc_b, loc_thresh=args.loc_thresh,
        calib=None, amp=args.amp
    )
    acc_c, macro_c, f1_c, locP_c, locR_c, locF_c = eval_loader_pipeline(
        model, test_ld, device,
        loc_model=loc_model, loc_a=loc_a, loc_b=loc_b, loc_thresh=args.loc_thresh,
        calib=calib, amp=args.amp
    )

    def _print(title, acc5, macro, f1s, locP, locR, locF):
        print(f"\n==================== {title} ====================")
        print(f"pipeline_acc(0..4)={acc5:.6f}")
        print(f"F1 Localization={locF:.6f} (P={locP:.6f}, R={locR:.6f})")
        print(f"macroF1(Damage 1..4)={macro:.6f}")
        for i in range(4):
            print(f"F1 {CLASS_NAMES_4[i]:>9s}: {f1s[i]:.6f}")
        print("==================================================\n")

    _print("PIPELINE TEST (UNCAL)", acc_u, macro_u, f1_u, locP_u, locR_u, locF_u)
    _print("PIPELINE TEST (CAL)",   acc_c, macro_c, f1_c, locP_c, locR_c, locF_c)

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
        "pipeline_macroF1_damage_uncal": macro_u,
        "pipeline_f1_no_damage_uncal": f1_u[0],
        "pipeline_f1_minor_uncal": f1_u[1],
        "pipeline_f1_major_uncal": f1_u[2],
        "pipeline_f1_destroyed_uncal": f1_u[3],

        "pipeline_acc_cal": acc_c,
        "pipeline_loc_precision_cal": locP_c,
        "pipeline_loc_recall_cal": locR_c,
        "pipeline_loc_f1_cal": locF_c,
        "pipeline_macroF1_damage_cal": macro_c,
        "pipeline_f1_no_damage_cal": f1_c[0],
        "pipeline_f1_minor_cal": f1_c[1],
        "pipeline_f1_major_cal": f1_c[2],
        "pipeline_f1_destroyed_cal": f1_c[3],
    }
    append_csv_row(args.csv_path, CSV_FIELDS, row)
    print(f"[CSV  ] Appended -> {args.csv_path}")


def main():
    ap = argparse.ArgumentParser()

    # All optional now (auto-filled if missing)
    ap.add_argument("--img_dir", default="")
    ap.add_argument("--mask_dir", default="")
    ap.add_argument("--weights_dir", default="")
    ap.add_argument("--gt_split", choices=["pre", "post"], default="post")

    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--early_stop_patience", type=int, default=15)
    ap.add_argument("--early_stop_warmup", type=int, default=5)
    ap.add_argument("--early_stop_min_delta", type=float, default=0.0)

    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--fusion_p", type=float, default=0.75)
    ap.add_argument("--min_build_px", type=int, default=50)

    ap.add_argument("--focus_destroyed_p", type=float, default=0.6)
    ap.add_argument("--min_destroyed_px", type=int, default=50)
    ap.add_argument("--crop_attempts", type=int, default=20)

    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--weight_count_images", type=int, default=200)
    ap.add_argument("--destroyed_weight_mul", type=float, default=3.0)

    ap.add_argument("--calib_pixels_per_class", type=int, default=50_000)
    ap.add_argument("--calib_lr", type=float, default=5e-2)
    ap.add_argument("--calib_epochs", type=int, default=10)
    ap.add_argument("--calib_batch", type=int, default=8192)
    ap.add_argument("--calib_wd", type=float, default=0.0)

    ap.add_argument("--init_weight", default="")
    ap.add_argument("--include_idabd_finetune", action="store_true")

    ap.add_argument("--out_dir", default="idabd_stage2_damage_ft_checkpoints_retrained_destroyed")

    ap.add_argument("--min_val_destroyed_tiles", type=int, default=1)
    ap.add_argument("--min_train_destroyed_tiles", type=int, default=1)

    # Now optional (auto-find)
    ap.add_argument("--loc_weight", type=str, default="")
    ap.add_argument("--loc_platt", type=str, default="")
    ap.add_argument("--loc_thresh", type=float, default=0.5)

    ap.add_argument("--csv_path", default="idabd_stage2_pipeline_results_DESTROYED_DEFENSIBLE.csv")
    ap.add_argument("--overwrite_csv", action="store_true")

    args = ap.parse_args()

    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    # Repro
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---------------- AUTO CONFIG ----------------
    img_auto, mask_auto = auto_defaults_for_dataset()
    if not args.img_dir or not path.isdir(args.img_dir):
        args.img_dir = img_auto
    if not args.mask_dir or not path.isdir(args.mask_dir):
        args.mask_dir = mask_auto

    if not args.weights_dir or not path.isdir(args.weights_dir):
        args.weights_dir = auto_defaults_for_weights_dir()

    if not args.loc_weight or not path.exists(args.loc_weight):
        args.loc_weight = auto_find_stage1_loc_weight()

    if (not args.loc_platt) and args.loc_weight:
        args.loc_platt = auto_find_stage1_platt(args.loc_weight)

    print("\n[AUTO CONFIG]")
    print(f"  img_dir     : {args.img_dir if args.img_dir else '(NOT FOUND)'}")
    print(f"  mask_dir    : {args.mask_dir if args.mask_dir else '(NOT FOUND)'}")
    print(f"  weights_dir : {args.weights_dir if args.weights_dir else '(NOT FOUND)'}")
    print(f"  loc_weight  : {args.loc_weight if args.loc_weight else '(NOT FOUND)'}")
    print(f"  loc_platt   : {args.loc_platt if args.loc_platt else '(none)'}")
    print(f"  loc_thresh  : {args.loc_thresh}")

    # Hard checks (for true zero-args run)
    if not args.img_dir or not path.isdir(args.img_dir):
        raise FileNotFoundError("Could not auto-find img_dir. Pass --img_dir explicitly.")
    if not args.mask_dir or not path.isdir(args.mask_dir):
        raise FileNotFoundError("Could not auto-find mask_dir. Pass --mask_dir explicitly.")
    if not args.weights_dir or not path.isdir(args.weights_dir):
        raise FileNotFoundError("Could not auto-find weights_dir. Pass --weights_dir explicitly.")
    if not args.loc_weight or not path.exists(args.loc_weight):
        raise FileNotFoundError("Could not auto-find Stage-1 loc checkpoint. Pass --loc_weight explicitly.")

    ensure_csv(args.csv_path, CSV_FIELDS, overwrite=args.overwrite_csv)
    print(f"[CSV  ] Ready: {args.csv_path}")

    # If init_weight given: run one
    if args.init_weight:
        if not path.exists(args.init_weight):
            raise FileNotFoundError(args.init_weight)
        train_eval_one(args, args.init_weight)
        return
    # Else: auto-run all init weights (your previous script behavior)
    weight_files = auto_find_stage2_init_weights(args.weights_dir, include_idabd_finetune=args.include_idabd_finetune)
    if not weight_files:
        raise FileNotFoundError(f"No Stage-2 init weights found in: {args.weights_dir}")
    print("[WEIGHTS] Stage-2 init checkpoints to run:")
    for w in weight_files:
        print("  -", w)

    for w in weight_files:
        train_eval_one(args, w)
if __name__ == "__main__":
    main()