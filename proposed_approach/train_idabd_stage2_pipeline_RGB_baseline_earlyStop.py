#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE-2 DAMAGE (IDABD) — RGB-only baseline (NO augmentation)
+ Destroyed-aware crops
+ Vector Scaling calibration on VAL (class-balanced)
+ Early stopping
+ Pipeline-only eval using Stage-1 localization
"""

import os
from os import path
import sys
from pathlib import Path
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------
# Make sure "zoo" is importable
# ---------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
for cand in [THIS_DIR, THIS_DIR.parent, THIS_DIR.parent.parent]:
    if (cand / "zoo").is_dir():
        sys.path.insert(0, str(cand))
        break

# ---------------------------------------------------------------------
# Shared modules (modular codebase)
# ---------------------------------------------------------------------
from idabd_stage2.constants import IGNORE_LABEL, CLASS_NAMES_4
from idabd_stage2.data.pairing import build_stage2_triplets
from idabd_stage2.data.dataset_stage2_damage import IdaBDStage2Damage, has_destroyed
from idabd_stage2.models.build_from_weights import build_damage_model_from_weight, build_loc_model_from_weight
from idabd_stage2.train.losses import compute_class_weights_from_triplets, masked_ce_loss
from idabd_stage2.eval.pipeline_only import eval_loader_pipeline, eval_val_loss
from idabd_stage2.calib.vector_scaling import collect_val_logits_balanced, fit_vector_scaling
from idabd_stage2.utils.csv_logger import ensure_csv, append_csv_row
from idabd_stage2.utils.auto_config import (
    auto_defaults_for_dataset,
    auto_defaults_for_weights_dir,
    auto_find_stage1_loc_weight,
    auto_find_stage1_platt,
    auto_find_stage2_init_weights,
)

IMG_DIR = "../idabd/images"
MASK_DIR = "../idabd/masks"
WEIGHTS_DIR = "weights"

CSV_FIELDS_BASELINE = [
    "seed", "init_weight", "loc_weight", "loc_platt", "loc_thresh",
    "pipeline_acc_uncal", "pipeline_loc_precision_uncal", "pipeline_loc_recall_uncal", "pipeline_loc_f1_uncal",
    "pipeline_macroF1_damage_uncal", "pipeline_f1_no_damage_uncal", "pipeline_f1_minor_uncal",
    "pipeline_f1_major_uncal", "pipeline_f1_destroyed_uncal",
    "pipeline_acc_cal", "pipeline_loc_precision_cal", "pipeline_loc_recall_cal", "pipeline_loc_f1_cal",
    "pipeline_macroF1_damage_cal", "pipeline_f1_no_damage_cal", "pipeline_f1_minor_cal",
    "pipeline_f1_major_cal", "pipeline_f1_destroyed_cal",
]


@torch.no_grad()
def train_and_eval_pipeline_one(args, init_weight: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] {device}")
    print(f"[INIT  ] {init_weight}")

    # -------------------------
    # Stage-1 loc (mandatory)
    # -------------------------
    if not args.loc_weight or not path.exists(args.loc_weight):
        raise FileNotFoundError(f"loc_weight not found: {args.loc_weight}")

    loc_model = build_loc_model_from_weight(args.loc_weight).to(device).eval()
    for p in loc_model.parameters():
        p.requires_grad = False
    print(f"[LOC   ] weights: {args.loc_weight}")

    loc_a, loc_b = 1.0, 0.0
    if args.loc_platt:
        d = np.load(args.loc_platt)
        loc_a = float(d["a"])
        loc_b = float(d["b"])
        print(f"[LOC   ] platt: {args.loc_platt} (a={loc_a:.6f}, b={loc_b:.6f})")
    else:
        print("[LOC   ] platt: (none) -> using a=1, b=0")
    print(f"[LOC   ] thresh: {args.loc_thresh}")

    # -------------------------
    # Triplets + split
    # -------------------------
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
    val_tr   = triplets[n_test:n_test + n_val]
    train_tr = triplets[n_test + n_val:]

    # Ensure destroyed tiles exist where needed
    if args.min_val_destroyed_tiles > 0:
        val_d = [t for t in val_tr if has_destroyed(t[2])]
        if len(val_d) < args.min_val_destroyed_tiles:
            train_d = [t for t in train_tr if has_destroyed(t[2])]
            need = args.min_val_destroyed_tiles - len(val_d)
            moved = 0
            for t in train_d:
                if moved >= need or len(val_tr) == 0:
                    break
                val_swap = val_tr[0]
                val_tr.remove(val_swap)
                train_tr.append(val_swap)
                if t in train_tr:
                    train_tr.remove(t)
                    val_tr.append(t)
                    moved += 1
            print(f"[SPLIT] ensured VAL destroyed tiles: {len([t for t in val_tr if has_destroyed(t[2])])} (moved={moved})")

    if args.min_train_destroyed_tiles > 0:
        train_d = [t for t in train_tr if has_destroyed(t[2])]
        if len(train_d) < args.min_train_destroyed_tiles:
            val_d = [t for t in val_tr if has_destroyed(t[2])]
            need = args.min_train_destroyed_tiles - len(train_d)
            moved = 0
            for t in val_d:
                if moved >= need or len(train_tr) == 0:
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
    # Loss
    # -------------------------
    if args.use_class_weights:
        w4, counts = compute_class_weights_from_triplets(train_tr, max_images=args.weight_count_images, seed=args.seed)
        w4[3] *= float(args.destroyed_weight_mul)
        print(f"[WGT  ] counts: {counts.tolist()}")
        print(f"[WGT  ] weights(1..4) after destroyed_mul={args.destroyed_weight_mul}: {w4.tolist()}")
        weight5 = np.ones(5, dtype=np.float32)
        weight5[1:5] = w4
        ce_loss = nn.CrossEntropyLoss(
            ignore_index=IGNORE_LABEL,
            reduction="mean",
            weight=torch.from_numpy(weight5).to(device),
        )
    else:
        ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL, reduction="mean")

    # -------------------------
    # Loaders (BASELINE = NO AUGMENTATION)
    # -------------------------
    train_ds = IdaBDStage2Damage(
        train_tr,
        crop_size=args.crop,
        is_train=True,
        min_build_px=args.min_build_px,
        seed=args.seed,
        focus_destroyed_p=args.focus_destroyed_p,
        min_destroyed_px=args.min_destroyed_px,
        crop_attempts=args.crop_attempts,
        aug_p=0.0,          # ✅ baseline
        aug_pair_fn=None,   # ✅ baseline
    )
    val_ds  = IdaBDStage2Damage(val_tr,  crop_size=0, is_train=False, min_build_px=0, seed=args.seed)
    test_ds = IdaBDStage2Damage(test_tr, crop_size=0, is_train=False, min_build_px=0, seed=args.seed)

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val_ds, batch_size=1, shuffle=False,
                          num_workers=max(0, args.workers // 2), pin_memory=True)
    test_ld  = DataLoader(test_ds, batch_size=1, shuffle=False,
                          num_workers=max(0, args.workers // 2), pin_memory=True)

    # -------------------------
    # Model + optimizer
    # -------------------------
    model = build_damage_model_from_weight(init_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=args.amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    os.makedirs(args.out_dir, exist_ok=True)
    base = path.splitext(path.basename(init_weight))[0]
    best_ckpt_path = path.join(args.out_dir, f"{base}_idabd_stage2_ft_best.pth")
    last_ckpt_path = path.join(args.out_dir, f"{base}_idabd_stage2_ft_last.pth")
    calib_path     = path.join(args.out_dir, f"{base}_idabd_stage2_calib_vector_scaling.npz")

    best_val = 1e9
    best_epoch = -1
    no_improve = 0

    # AMP ctx
    if args.amp and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        def amp_ctx():
            return torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"))
    else:
        def amp_ctx():
            return torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp))

    # -------------------------
    # Train + early stop on VAL loss
    # -------------------------
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

        val_loss, val_skipped = eval_val_loss(
            model, val_ld, device,
            ce_loss=ce_loss,
            masked_ce_loss_fn=masked_ce_loss,
            amp=args.amp
        )

        print(f"[EPOCH {epoch:03d}/{args.epochs}] train_loss={mean_loss:.5f} (skipped={skipped}) | "
              f"val_loss={val_loss:.5f} (skipped={val_skipped})")

        torch.save({"epoch": epoch, "init_weight": init_weight, "state_dict": model.state_dict(),
                    "args": vars(args), "train_loss": mean_loss, "val_loss": val_loss}, last_ckpt_path)

        improved = (val_loss < (best_val - args.early_stop_min_delta))
        if improved:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save({"epoch": epoch, "init_weight": init_weight, "state_dict": model.state_dict(),
                        "args": vars(args), "train_loss": mean_loss, "val_loss": val_loss}, best_ckpt_path)
            print(f"[SAVE ] Best -> {best_ckpt_path} (best_val={best_val:.6f})")
        else:
            if epoch >= args.early_stop_warmup:
                no_improve += 1

        if args.early_stop_patience > 0 and epoch >= args.early_stop_warmup and no_improve >= args.early_stop_patience:
            print(f"[EARLY STOP] stopped at epoch {epoch}. best_epoch={best_epoch}, best_val={best_val:.6f}")
            break

    # -------------------------
    # Load best + calibrate
    # -------------------------
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["state_dict"], strict=True)
    model.eval()
    print(f"\n[LOAD ] BEST for calib/test: {best_ckpt_path}")

    X, Y, got = collect_val_logits_balanced(model, val_ld, device,
                                            per_class_pixels=args.calib_pixels_per_class, seed=args.seed)
    print(f"[CALIB] collected [No,Minor,Major,Destroyed]={got.tolist()} total={int(X.shape[0])}")

    W_np, b_np = fit_vector_scaling(X, Y, device=device,
                                    lr=args.calib_lr, epochs=args.calib_epochs,
                                    batch_size=args.calib_batch, wd=args.calib_wd)
    np.savez(calib_path, W=W_np, b=b_np)
    print(f"[CALIB] saved -> {calib_path}")

    calib_damage = (torch.from_numpy(W_np).to(device), torch.from_numpy(b_np).to(device))

    # -------------------------
    # Pipeline eval
    # -------------------------
    acc_u, macro_u, f1s_u, locP_u, locR_u, locF_u = eval_loader_pipeline(
        model, test_ld, device,
        loc_model=loc_model, loc_a=loc_a, loc_b=loc_b, loc_thresh=args.loc_thresh,
        calib=None, amp=args.amp
    )
    acc_c, macro_c, f1s_c, locP_c, locR_c, locF_c = eval_loader_pipeline(
        model, test_ld, device,
        loc_model=loc_model, loc_a=loc_a, loc_b=loc_b, loc_thresh=args.loc_thresh,
        calib=calib_damage, amp=args.amp
    )

    def show(title, acc5, macroB, f1sD, locP, locR, locF):
        print(f"\n==== {title} ====")
        print(f"acc(0..4)={acc5:.6f} | locF1={locF:.6f} (P={locP:.6f}, R={locR:.6f}) | macroF1(1..4)={macroB:.6f}")
        for i, name in enumerate(CLASS_NAMES_4):
            print(f"F1 {name:>9s}: {f1sD[i]:.6f}")

    show("PIPELINE TEST (UNCAL)", acc_u, macro_u, f1s_u, locP_u, locR_u, locF_u)
    show("PIPELINE TEST (CAL)",   acc_c, macro_c, f1s_c, locP_c, locR_c, locF_c)

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
        "pipeline_f1_no_damage_uncal": f1s_u[0],
        "pipeline_f1_minor_uncal": f1s_u[1],
        "pipeline_f1_major_uncal": f1s_u[2],
        "pipeline_f1_destroyed_uncal": f1s_u[3],

        "pipeline_acc_cal": acc_c,
        "pipeline_loc_precision_cal": locP_c,
        "pipeline_loc_recall_cal": locR_c,
        "pipeline_loc_f1_cal": locF_c,
        "pipeline_macroF1_damage_cal": macro_c,
        "pipeline_f1_no_damage_cal": f1s_c[0],
        "pipeline_f1_minor_cal": f1s_c[1],
        "pipeline_f1_major_cal": f1s_c[2],
        "pipeline_f1_destroyed_cal": f1s_c[3],
    }
    append_csv_row(args.csv_path, CSV_FIELDS_BASELINE, row)
    print(f"[CSV  ] appended -> {args.csv_path}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--img_dir", default="")
    ap.add_argument("--mask_dir", default="")
    ap.add_argument("--weights_dir", default="")
    ap.add_argument("--gt_split", choices=["pre", "post"], default="post")

    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=10)
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

    ap.add_argument("--out_dir", default="idabd_stage2_damage_ft_checkpoints_RGB_ONLY_BASELINE")
    ap.add_argument("--min_val_destroyed_tiles", type=int, default=1)
    ap.add_argument("--min_train_destroyed_tiles", type=int, default=1)

    ap.add_argument("--loc_weight", default="")
    ap.add_argument("--loc_platt", default="")
    ap.add_argument("--loc_thresh", type=float, default=0.5)

    ap.add_argument("--csv_path", default="idabd_stage2_pipeline_results_RGB_ONLY_BASELINE.csv")
    ap.add_argument("--overwrite_csv", action="store_true")

    args = ap.parse_args()

    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    # AUTO config (shared)
    img_auto, mask_auto, img_tried, mask_tried = auto_defaults_for_dataset(THIS_DIR, IMG_DIR, MASK_DIR)
    if not args.img_dir or not path.isdir(args.img_dir):
        args.img_dir = img_auto
    if not args.mask_dir or not path.isdir(args.mask_dir):
        args.mask_dir = mask_auto

    w_auto, w_tried = auto_defaults_for_weights_dir(THIS_DIR, WEIGHTS_DIR)
    if not args.weights_dir or not path.isdir(args.weights_dir):
        args.weights_dir = w_auto

    if not args.loc_weight:
        args.loc_weight, _ = auto_find_stage1_loc_weight(THIS_DIR)
    if not args.loc_platt and args.loc_weight:
        args.loc_platt, _ = auto_find_stage1_platt(args.loc_weight)

    print("\n[AUTO CONFIG]")
    print(f"  img_dir     : {args.img_dir}")
    print(f"  mask_dir    : {args.mask_dir}")
    print(f"  weights_dir : {args.weights_dir}")
    print(f"  loc_weight  : {args.loc_weight if args.loc_weight else '(NOT FOUND)'}")
    print(f"  loc_platt   : {args.loc_platt if args.loc_platt else '(none)'}")
    print(f"  out_dir     : {args.out_dir}")

    if not path.isdir(args.img_dir):
        print("\n[AUTO DEBUG] Tried img_dir candidates:")
        for d in img_tried: print("  -", d)
        raise FileNotFoundError("Could not find img_dir. Pass --img_dir explicitly.")
    if not path.isdir(args.mask_dir):
        print("\n[AUTO DEBUG] Tried mask_dir candidates:")
        for d in mask_tried: print("  -", d)
        raise FileNotFoundError("Could not find mask_dir. Pass --mask_dir explicitly.")
    if not args.loc_weight or not path.exists(args.loc_weight):
        raise FileNotFoundError("Could not find Stage-1 loc checkpoint. Pass --loc_weight explicitly.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ensure_csv(args.csv_path, CSV_FIELDS_BASELINE, overwrite=args.overwrite_csv)
    print(f"[CSV  ] Ready: {args.csv_path}")

    if args.init_weight:
        if not path.exists(args.init_weight):
            raise FileNotFoundError(args.init_weight)
        train_and_eval_pipeline_one(args, args.init_weight)
        return

    weight_files, patterns = auto_find_stage2_init_weights(args.weights_dir)
    if not args.include_idabd_finetune:
        weight_files = [w for w in weight_files if "idabd_finetune" not in path.basename(w).lower()]

    if not weight_files:
        raise FileNotFoundError(f"No Stage-2 init weights found. Expected patterns: {patterns}")

    print("[WEIGHTS] Stage-2 init checkpoints to run:")
    for w in weight_files:
        print("  -", w)

    for w in weight_files:
        train_and_eval_pipeline_one(args, w)


if __name__ == "__main__":
    main()