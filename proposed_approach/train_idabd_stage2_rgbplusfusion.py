#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE-2 (IDABD) — RGB + FULL FUSION AUGMENTATION (TRAIN-ONLY)
- Robust Windows spawn-safe
- Robust Stage-1 checkpoint discovery (optional)
- Filters broken triplets (cv2.imread None) to avoid NoneType .shape crashes
- Keeps your pipeline eval: Stage-1 localization gating + (optional) Platt + Stage-2 vector scaling
"""

import os
from os import path
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import multiprocessing as mp

import cv2  # for validation of readable images

from idabd_stage2.utils.paths import ensure_zoo_importable
ensure_zoo_importable(__file__)

from idabd_stage2.constants import (
    IMG_DIR_DEFAULT, MASK_DIR_DEFAULT, WEIGHTS_DIR_DEFAULT, STAGE1_DIR_DEFAULT,
    IGNORE_LABEL, CLASS_NAMES_4
)
from idabd_stage2.utils.auto import (
    auto_find_stage1_loc_weight, auto_find_stage1_platt, auto_find_stage2_init_weights
)
from idabd_stage2.data.pairing import build_stage2_triplets
from idabd_stage2.models.loaders import build_damage_model_from_weight, build_loc_model_from_weight
from idabd_stage2.train.dataset6ch import IdaBDStage2Dataset6CH
from idabd_stage2.train.losses import compute_class_weights_from_triplets
from idabd_stage2.train.fit_loop import train_with_checkpoints
from idabd_stage2.train.pipeline_eval import eval_loader_pipeline
from idabd_stage2.train.calibration import collect_val_logits_balanced, fit_vector_scaling
from idabd_stage2.utils.csv_utils import ensure_csv, append_csv_row
from idabd_stage2.aug.fusion_full import fusion_augment
from idabd_stage2.train.split import split_triplets

CSV_FIELDS = [
    "seed","init_weight","loc_weight","loc_platt","loc_thresh",
    "amp","workers","fusion_p",
    "pipeline_acc_uncal","pipeline_loc_precision_uncal","pipeline_loc_recall_uncal","pipeline_loc_f1_uncal",
    "pipeline_macroF1_damage_uncal","pipeline_f1_no_damage_uncal","pipeline_f1_minor_uncal",
    "pipeline_f1_major_uncal","pipeline_f1_destroyed_uncal",
    "pipeline_acc_cal","pipeline_loc_precision_cal","pipeline_loc_recall_cal","pipeline_loc_f1_cal",
    "pipeline_macroF1_damage_cal","pipeline_f1_no_damage_cal","pipeline_f1_minor_cal",
    "pipeline_f1_major_cal","pipeline_f1_destroyed_cal",
]


def filter_valid_triplets(triplets, max_report=12):
    """
    Prevent dataset crashes like:
      AttributeError: 'NoneType' object has no attribute 'shape'
    which happens when cv2.imread(...) returns None (unreadable/missing image).
    """
    kept, bad = [], []
    for pre_p, post_p, mask_p in triplets:
        pre_p, post_p, mask_p = str(pre_p), str(post_p), str(mask_p)

        if (not path.exists(pre_p)) or (not path.exists(post_p)) or (not path.exists(mask_p)):
            bad.append((pre_p, post_p, mask_p, "missing_file"))
            continue

        im = cv2.imread(post_p, cv2.IMREAD_COLOR)
        if im is None:
            bad.append((pre_p, post_p, mask_p, "cv2_imread_post_none"))
            continue

        kept.append((pre_p, post_p, mask_p))

    if bad:
        print(f"[WARN] Dropped {len(bad)} broken triplets. Showing up to {max_report}:")
        for i, (a, b, c, why) in enumerate(bad[:max_report]):
            print(f"  {i+1:02d}) why={why}\n      pre ={a}\n      post={b}\n      mask={c}")

    print(f"[INFO] Triplets: total={len(triplets)} kept={len(kept)} dropped={len(bad)}")
    return kept


def try_find_stage1_loc_in_dir(stage1_dir: str):
    """Pick a likely Stage-1 localization checkpoint from a directory (if provided)."""
    if not stage1_dir or not path.isdir(stage1_dir):
        return ""
    cand = list(Path(stage1_dir).rglob("*.pth"))
    if not cand:
        return ""

    def score(p: Path):
        n = p.name.lower()
        s = 0
        if "loc" in n: s += 5
        if "stage1" in n: s += 3
        if "best" in n: s += 2
        if "calib" in n or "platt" in n: s -= 5
        if "stage2" in n or "damage" in n: s -= 5
        return s

    cand.sort(key=lambda p: (score(p), p.stat().st_mtime), reverse=True)
    return str(cand[0])


def try_find_platt_in_dir(stage1_dir: str):
    """Pick a likely Platt npz from a directory (if provided)."""
    if not stage1_dir or not path.isdir(stage1_dir):
        return ""
    cand = list(Path(stage1_dir).rglob("*.npz"))
    if not cand:
        return ""

    def score(p: Path):
        n = p.name.lower()
        s = 0
        if "platt" in n: s += 10
        if "stage1" in n or "loc" in n: s += 2
        if "stage2" in n or "damage" in n: s -= 5
        return s

    cand.sort(key=lambda p: (score(p), p.stat().st_mtime), reverse=True)
    best = cand[0]
    return str(best) if score(best) > 0 else ""


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--img_dir", default=IMG_DIR_DEFAULT)
    ap.add_argument("--mask_dir", default=MASK_DIR_DEFAULT)
    ap.add_argument("--weights_dir", default=WEIGHTS_DIR_DEFAULT)
    ap.add_argument("--stage1_dir", default=STAGE1_DIR_DEFAULT)
    ap.add_argument("--gt_split", choices=["pre","post"], default="post")

    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)

    # AMP OFF by default, but you can enable with --amp
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

    # Stage-2 init weight: if not provided, auto-find (so script is easier to run)
    ap.add_argument("--init_weight", default="")

    ap.add_argument("--out_dir", default="idabd_stage2_damage_ft_checkpoints_RGBplusFUSION_6ch")

    # Stage-1 loc: if not provided, auto-find
    ap.add_argument("--loc_weight", default="")
    ap.add_argument("--loc_platt", default="")
    ap.add_argument("--loc_thresh", type=float, default=0.5)

    ap.add_argument("--csv_path", default="idabd_stage2_pipeline_results_RGBplusFUSION_6ch.csv")
    ap.add_argument("--overwrite_csv", action="store_true")

    args = ap.parse_args()
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    ensure_csv(args.csv_path, CSV_FIELDS, overwrite=args.overwrite_csv)

    # seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device} workers={args.workers} amp={bool(args.amp)} fusion_p={args.fusion_p}")

    # -----------------------
    # Stage-1 loc checkpoint auto-find (fixes the Path/str bug)
    # -----------------------
    if not args.loc_weight:
        args.loc_weight = try_find_stage1_loc_in_dir(args.stage1_dir)
    if not args.loc_weight:
        # IMPORTANT: auto_find_stage1_loc_weight expects a Path to a script file (not a string dir)
        args.loc_weight = auto_find_stage1_loc_weight(Path(__file__))

    if not args.loc_platt:
        args.loc_platt = try_find_platt_in_dir(args.stage1_dir)
    if not args.loc_platt:
        # keep optional; if auto finder fails, proceed without platt
        try:
            args.loc_platt = auto_find_stage1_platt(args.loc_weight)
        except Exception:
            args.loc_platt = ""

    if not args.loc_weight or not path.exists(args.loc_weight):
        raise FileNotFoundError(
            f"Stage-1 loc checkpoint not found.\n"
            f"  stage1_dir={args.stage1_dir}\n"
            f"  Pass --loc_weight explicitly if needed."
        )

    # -----------------------
    # Stage-2 init weight auto-find
    # -----------------------
    if not args.init_weight:
        out = auto_find_stage2_init_weights(args.weights_dir, include_idabd_finetune=True)
        if isinstance(out, tuple):
            init_list = out[0]
        else:
            init_list = out
        if not init_list:
            raise FileNotFoundError("No stage-2 init weights found. Pass --init_weight or check --weights_dir.")
        args.init_weight = init_list[0]

    if not path.exists(args.init_weight):
        raise FileNotFoundError(f"Stage-2 init weight not found: {args.init_weight}")

    # -----------------------
    # Models
    # -----------------------
    loc_model = build_loc_model_from_weight(args.loc_weight).to(device).eval()
    for p in loc_model.parameters():
        p.requires_grad = False

    loc_a, loc_b = 1.0, 0.0
    if args.loc_platt and path.exists(args.loc_platt):
        d = np.load(args.loc_platt)
        loc_a, loc_b = float(d["a"]), float(d["b"])

    print(f"[INFO] Stage-1 loc_weight={args.loc_weight}")
    print(f"[INFO] Stage-1 platt={args.loc_platt} (a={loc_a:.6f}, b={loc_b:.6f})")
    print(f"[INFO] Stage-2 init_weight={args.init_weight}")

    # -----------------------
    # Data
    # -----------------------
    triplets = build_stage2_triplets(args.img_dir, args.mask_dir, gt_split=args.gt_split)
    if not triplets:
        raise FileNotFoundError("No (pre, post, mask) triplets found.")

    triplets = filter_valid_triplets(triplets)
    if not triplets:
        raise FileNotFoundError("All triplets are invalid (missing/unreadable post images).")

    train_tr, val_tr, test_tr = split_triplets(triplets, args.seed, args.val_ratio, args.test_ratio)

    # -----------------------
    # Loss
    # -----------------------
    if args.use_class_weights:
        w4, _ = compute_class_weights_from_triplets(train_tr, max_images=args.weight_count_images, seed=args.seed)
        w4[3] *= float(args.destroyed_weight_mul)
        weight5 = np.ones(5, dtype=np.float32)
        weight5[1:5] = w4
        ce_loss = nn.CrossEntropyLoss(
            ignore_index=IGNORE_LABEL, reduction="mean",
            weight=torch.from_numpy(weight5).to(device)
        )
    else:
        ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL, reduction="mean")

    # -----------------------
    # Datasets / loaders
    # TRAIN: fusion on post with prob fusion_p ; VAL/TEST: raw
    # -----------------------
    train_ds = IdaBDStage2Dataset6CH(
        train_tr, crop_size=args.crop, is_train=True, seed=args.seed,
        min_build_px=args.min_build_px,
        focus_destroyed_p=args.focus_destroyed_p,
        min_destroyed_px=args.min_destroyed_px,
        crop_attempts=args.crop_attempts,
        post_p_train=args.fusion_p,
        post_transform_train=fusion_augment,   # top-level function (picklable)
        post_transform_eval=None,
    )
    val_ds = IdaBDStage2Dataset6CH(val_tr, crop_size=0, is_train=False, seed=args.seed)
    test_ds = IdaBDStage2Dataset6CH(test_tr, crop_size=0, is_train=False, seed=args.seed)

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

    # -----------------------
    # Train
    # -----------------------
    model = build_damage_model_from_weight(args.init_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    os.makedirs(args.out_dir, exist_ok=True)
    base = path.splitext(path.basename(args.init_weight))[0]
    best_ckpt_path = path.join(args.out_dir, f"{base}_idabd_stage2_ft_best_6ch.pth")
    last_ckpt_path = path.join(args.out_dir, f"{base}_idabd_stage2_ft_last_6ch.pth")
    calib_path = path.join(args.out_dir, f"{base}_idabd_stage2_calib_vector_scaling_6ch.npz")

    train_with_checkpoints(
        model, train_ld, val_ld, device, ce_loss, opt,
        epochs=args.epochs, amp=args.amp,
        last_ckpt_path=last_ckpt_path, best_ckpt_path=best_ckpt_path,
        ckpt_extra={"init_weight": args.init_weight, "args": vars(args), "stage2_in_channels": 6},
        early_stop_patience=args.early_stop_patience,
        early_stop_warmup=args.early_stop_warmup,
        early_stop_min_delta=args.early_stop_min_delta,
    )

    best = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"], strict=False)
    model.eval()

    # -----------------------
    # Calibration (vector scaling) on VAL logits
    # -----------------------
    X, Y, _ = collect_val_logits_balanced(
        model, val_ld, device,
        per_class_pixels=args.calib_pixels_per_class, seed=args.seed
    )
    W_np, b_np = fit_vector_scaling(
        X, Y, device,
        lr=args.calib_lr, epochs=args.calib_epochs,
        batch_size=args.calib_batch, wd=args.calib_wd
    )
    np.savez(calib_path, W=W_np, b=b_np)
    calib = (torch.from_numpy(W_np).to(device), torch.from_numpy(b_np).to(device))

    # -----------------------
    # Pipeline eval
    # -----------------------
    acc_u, macro_u, f1_u, locP_u, locR_u, locF_u = eval_loader_pipeline(
        model, test_ld, device, loc_model, loc_a, loc_b, args.loc_thresh, calib=None, amp=args.amp
    )
    acc_c, macro_c, f1_c, locP_c, locR_c, locF_c = eval_loader_pipeline(
        model, test_ld, device, loc_model, loc_a, loc_b, args.loc_thresh, calib=calib, amp=args.amp
    )

    row = {
        "seed": args.seed,
        "init_weight": args.init_weight,
        "loc_weight": args.loc_weight,
        "loc_platt": args.loc_platt,
        "loc_thresh": args.loc_thresh,
        "amp": bool(args.amp),
        "workers": int(args.workers),
        "fusion_p": float(args.fusion_p),

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

    print("\nPer-damage F1 (calibrated):")
    for i in range(4):
        print(f"  F1 {CLASS_NAMES_4[i]:>9s}: {f1_c[i]:.6f}")


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    main()