#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE-2 DAMAGE (IDABD) — RGB + CONTRAST + EDGE ONLY (POST-AUG TRAIN ONLY)
- Destroyed-aware crops
- Guarantee destroyed tiles in VAL and TRAIN
- AMP ON by default (disable with --no_amp)
- Class weights ON by default (disable with --no_class_weights)
- Vector Scaling calibration on VAL logits (class-balanced pixels)
- Pipeline-only evaluation using Stage-1 localization gating
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
# Shared modules
# ---------------------------------------------------------------------
from idabd_stage2.constants import IGNORE_LABEL, CLASS_NAMES_4
from idabd_stage2.aug.contrast_edge_only import augment_pair_post_contrast_edge
from idabd_stage2.data.pairing import build_stage2_triplets
from idabd_stage2.data.dataset_stage2_damage import IdaBDStage2Damage, has_destroyed
from idabd_stage2.data.splits import ensure_destroyed_in_splits
from idabd_stage2.models.build_from_weights import build_damage_model_from_weight, build_loc_model_from_weight
from idabd_stage2.train.losses import compute_class_weights_from_triplets, masked_ce_loss
from idabd_stage2.eval.pipeline_only import eval_loader_pipeline, eval_val_loss
from idabd_stage2.calib.vector_scaling import collect_val_logits_balanced, fit_vector_scaling
from idabd_stage2.utils.csv_logger import (
    ensure_csv,
    append_csv_row,
    CSV_FIELDS_STAGE2_PIPELINE_CONTRAST_EDGE,
)
from idabd_stage2.utils.auto_config import (
    auto_defaults_for_dataset,
    auto_defaults_for_weights_dir,
    auto_find_stage1_loc_weight,
    auto_find_stage1_platt,
    auto_find_stage2_init_weights,
)

# Fallbacks if auto-detect fails
IMG_DIR = "../idabd/images"
MASK_DIR = "../idabd/masks"
WEIGHTS_DIR = "weights"


@torch.no_grad()
def train_and_eval_pipeline_one(args, init_weight: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] {device}")
    print(f"[INIT ] {init_weight}")

    # ---------------- Stage-1 localization ----------------
    if not args.loc_weight or not path.exists(args.loc_weight):
        raise FileNotFoundError("Missing Stage-1 localization checkpoint (loc_weight).")

    loc_model = build_loc_model_from_weight(args.loc_weight).to(device).eval()
    for p in loc_model.parameters():
        p.requires_grad = False
    print(f"[LOC  ] weights: {args.loc_weight}")

    loc_a, loc_b = 1.0, 0.0
    if args.loc_platt:
        if not path.exists(args.loc_platt):
            raise FileNotFoundError(args.loc_platt)
        d = np.load(args.loc_platt)
        loc_a = float(d["a"])
        loc_b = float(d["b"])
        print(f"[LOC  ] platt: {args.loc_platt} (a={loc_a:.6f}, b={loc_b:.6f})")
    else:
        print("[LOC  ] platt: (none) -> a=1, b=0")
    print(f"[LOC  ] thresh: {args.loc_thresh}")

    # ---------------- Triplets + split ----------------
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

    ensure_destroyed_in_splits(
        train_tr, val_tr,
        min_train=int(args.min_train_destroyed_tiles),
        min_val=int(args.min_val_destroyed_tiles),
    )

    print(f"[DATA ] total={n} | train={len(train_tr)} | val={len(val_tr)} | test={len(test_tr)}")
    print(f"[DATA ] destroyed tiles: train={sum(has_destroyed(t[2]) for t in train_tr)} | "
          f"val={sum(has_destroyed(t[2]) for t in val_tr)} | test={sum(has_destroyed(t[2]) for t in test_tr)}")

    # ---------------- Loss ----------------
    if args.use_class_weights:
        w4, counts = compute_class_weights_from_triplets(train_tr, max_images=args.weight_count_images, seed=args.seed)
        w4[3] *= float(args.destroyed_weight_mul)
        weight5 = np.ones(5, dtype=np.float32)
        weight5[1:5] = w4
        ce_loss = nn.CrossEntropyLoss(
            ignore_index=IGNORE_LABEL,
            reduction="mean",
            weight=torch.from_numpy(weight5).to(device),
        )
        print(f"[WGT  ] counts={counts.tolist()} weights(1..4)={w4.tolist()}")
    else:
        ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL, reduction="mean")
        print("[WGT  ] class weights: OFF")

    # ---------------- Dataset / loaders ----------------
    # TRAIN: apply POST contrast+edge with probability fusion_p
    train_ds = IdaBDStage2Damage(
        train_tr,
        crop_size=args.crop,
        is_train=True,
        min_build_px=args.min_build_px,
        seed=args.seed,
        focus_destroyed_p=args.focus_destroyed_p,
        min_destroyed_px=args.min_destroyed_px,
        crop_attempts=args.crop_attempts,
        aug_p=args.fusion_p,
        aug_pair_fn=lambda pre, post: augment_pair_post_contrast_edge(
            pre, post,
            alpha_contrast=args.alpha_contrast,
            alpha_edge=args.alpha_edge,
            clahe_clip=args.clahe_clip,
            clahe_grid=args.clahe_grid,
            canny_t1=args.canny_t1,
            canny_t2=args.canny_t2,
            edge_dilate=args.edge_dilate,
        ),
    )

    # VAL/TEST: raw RGB (NO fusion) for clean calibration/eval
    val_ds  = IdaBDStage2Damage(val_tr,  crop_size=0, is_train=False, min_build_px=0, seed=args.seed, aug_p=0.0, aug_pair_fn=None)
    test_ds = IdaBDStage2Damage(test_tr, crop_size=0, is_train=False, min_build_px=0, seed=args.seed, aug_p=0.0, aug_pair_fn=None)

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=max(0, args.workers // 2), pin_memory=True)
    test_ld = DataLoader(test_ds, batch_size=1, shuffle=False,
                         num_workers=max(0, args.workers // 2), pin_memory=True)

    # ---------------- Model/optim/scaler ----------------
    model = build_damage_model_from_weight(init_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if device.type == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    if args.amp and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        def amp_ctx():
            return torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"))
    else:
        def amp_ctx():
            return torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp))

    # ---------------- Checkpoint paths ----------------
    os.makedirs(args.out_dir, exist_ok=True)
    base = path.splitext(path.basename(init_weight))[0]
    best_ckpt_path = path.join(args.out_dir, f"{base}_idabd_stage2_ft_best.pth")
    last_ckpt_path = path.join(args.out_dir, f"{base}_idabd_stage2_ft_last.pth")
    calib_path     = path.join(args.out_dir, f"{base}_idabd_stage2_calib_vector_scaling.npz")

    # ---------------- Train loop (early stop) ----------------
    best_val = 1e9
    best_epoch = -1
    patience = int(args.early_stop_patience)
    warmup = int(args.early_stop_warmup)
    min_delta = float(args.early_stop_min_delta)
    no_improve = 0

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

        train_loss = float(np.mean(losses)) if losses else 0.0
        val_loss, val_skipped = eval_val_loss(model, val_ld, device, ce_loss, masked_ce_loss_fn=masked_ce_loss, amp=args.amp)

        print(f"[EPOCH {epoch:03d}/{args.epochs}] "
              f"train_loss={train_loss:.5f} (skipped={skipped}) | "
              f"val_loss={val_loss:.5f} (skipped={val_skipped})")

        torch.save({
            "epoch": epoch,
            "init_weight": init_weight,
            "state_dict": model.state_dict(),
            "args": vars(args),
            "train_loss": train_loss,
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
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, best_ckpt_path)
            print(f"[SAVE ] Best (val) -> {best_ckpt_path} (best_val={best_val:.6f})")
        else:
            if epoch >= warmup:
                no_improve += 1

        if patience > 0 and epoch >= warmup and no_improve >= patience:
            print(f"[EARLY STOP] No VAL improvement for {patience} epochs. "
                  f"Stopping at epoch {epoch}. Best was epoch {best_epoch} (val={best_val:.6f}).")
            break

    # ---------------- Load best + calibrate ----------------
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["state_dict"], strict=True)
    model.eval()
    print(f"\n[LOAD ] BEST checkpoint for calibration/test: {best_ckpt_path}")

    X, Y, got = collect_val_logits_balanced(
        model, val_ld, device,
        per_class_pixels=args.calib_pixels_per_class,
        seed=args.seed
    )
    print(f"[CALIB] Collected per-class pixels [No,Minor,Major,Destroyed]={got.tolist()} total={int(X.shape[0])}")

    W_np, b_np = fit_vector_scaling(
        X, Y, device=device,
        lr=args.calib_lr, epochs=args.calib_epochs, batch_size=args.calib_batch, wd=args.calib_wd
    )
    np.savez(calib_path, W=W_np, b=b_np,
             note="vector scaling on 4-dim building logits (class-balanced sampling)")
    print(f"[CALIB] Saved -> {calib_path}")

    calib_damage = (torch.from_numpy(W_np).to(device), torch.from_numpy(b_np).to(device))

    # ---------------- Pipeline eval (uncal + cal) ----------------
    def print_pipeline(title, acc5, macroB, f1sD, locP, locR, locF):
        print(f"\n==================== {title} ====================")
        print("END-TO-END: Stage-1 loc gating -> Stage-2 damage (mask!=255)")
        print(f"pipeline_acc(0..4)={acc5:.6f}")
        print(f"F1 Localization={locF:.6f} (P={locP:.6f}, R={locR:.6f})")
        print(f"macroF1(Damage 1..4)={macroB:.6f}")
        for i in range(4):
            print(f"F1 {CLASS_NAMES_4[i]:>9s}: {f1sD[i]:.6f}")
        print("==================================================\n")

    acc_u, macro_u, f1_u, locP_u, locR_u, locF_u = eval_loader_pipeline(
        model, test_ld, device,
        loc_model=loc_model, loc_a=loc_a, loc_b=loc_b, loc_thresh=args.loc_thresh,
        calib=None, amp=args.amp
    )
    acc_c, macro_c, f1_c, locP_c, locR_c, locF_c = eval_loader_pipeline(
        model, test_ld, device,
        loc_model=loc_model, loc_a=loc_a, loc_b=loc_b, loc_thresh=args.loc_thresh,
        calib=calib_damage, amp=args.amp
    )

    print_pipeline("PIPELINE TEST (UNCALIBRATED DAMAGE + STAGE-1 LOC)", acc_u, macro_u, f1_u, locP_u, locR_u, locF_u)
    print_pipeline("PIPELINE TEST (CALIBRATED DAMAGE + STAGE-1 LOC)",   acc_c, macro_c, f1_c, locP_c, locR_c, locF_c)

    # ---------------- CSV logging ----------------
    row = {
        "seed": args.seed,
        "init_weight": init_weight,
        "loc_weight": args.loc_weight,
        "loc_platt": args.loc_platt,
        "loc_thresh": args.loc_thresh,

        "fusion_p": args.fusion_p,
        "alpha_contrast": args.alpha_contrast,
        "alpha_edge": args.alpha_edge,
        "clahe_clip": args.clahe_clip,
        "clahe_grid": args.clahe_grid,
        "canny_t1": args.canny_t1,
        "canny_t2": args.canny_t2,
        "edge_dilate": args.edge_dilate,

        "min_val_destroyed_tiles": args.min_val_destroyed_tiles,
        "min_train_destroyed_tiles": args.min_train_destroyed_tiles,

        "use_class_weights": bool(args.use_class_weights),
        "destroyed_weight_mul": float(args.destroyed_weight_mul),

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
    append_csv_row(args.csv_path, CSV_FIELDS_STAGE2_PIPELINE_CONTRAST_EDGE, row)
    print(f"[CSV  ] Appended -> {args.csv_path}")


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

    # AMP ON by default (use --no_amp to disable)
    ap.add_argument("--no_amp", action="store_true", help="Disable AMP (AMP enabled by default).")

    ap.add_argument("--early_stop_patience", type=int, default=15)
    ap.add_argument("--early_stop_warmup", type=int, default=5)
    ap.add_argument("--early_stop_min_delta", type=float, default=0.0)

    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--min_build_px", type=int, default=50)

    # Destroyed-aware crops
    ap.add_argument("--focus_destroyed_p", type=float, default=0.7)
    ap.add_argument("--min_destroyed_px", type=int, default=80)
    ap.add_argument("--crop_attempts", type=int, default=25)

    # Class weights ON by default (use --no_class_weights to disable)
    ap.add_argument("--no_class_weights", action="store_true")
    ap.add_argument("--weight_count_images", type=int, default=200)
    ap.add_argument("--destroyed_weight_mul", type=float, default=4.0)

    # Calibration
    ap.add_argument("--calib_pixels_per_class", type=int, default=50_000)
    ap.add_argument("--calib_lr", type=float, default=5e-2)
    ap.add_argument("--calib_epochs", type=int, default=10)
    ap.add_argument("--calib_batch", type=int, default=8192)
    ap.add_argument("--calib_wd", type=float, default=0.0)

    # TRAIN-only fusion params (POST only)
    ap.add_argument("--fusion_p", type=float, default=0.75)
    ap.add_argument("--alpha_contrast", type=float, default=0.45)
    ap.add_argument("--alpha_edge", type=float, default=0.10)
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_grid", type=int, default=8)
    ap.add_argument("--canny_t1", type=int, default=80)
    ap.add_argument("--canny_t2", type=int, default=200)
    ap.add_argument("--edge_dilate", type=int, default=0)

    # Destroyed tiles guarantee
    ap.add_argument("--min_val_destroyed_tiles", type=int, default=1)
    ap.add_argument("--min_train_destroyed_tiles", type=int, default=1)

    # Weights selection
    ap.add_argument("--init_weight", default="")
    ap.add_argument("--include_idabd_finetune", action="store_true")

    # Outputs + Stage-1 loc
    ap.add_argument("--out_dir", default="idabd_stage2_damage_ft_checkpoints_RGB_CONTRAST_EDGE_ONLY")
    ap.add_argument("--loc_weight", type=str, default="")
    ap.add_argument("--loc_platt", type=str, default="")
    ap.add_argument("--loc_thresh", type=float, default=0.5)

    ap.add_argument("--csv_path", default="idabd_stage2_pipeline_results_RGB_CONTRAST_EDGE_ONLY.csv")
    ap.add_argument("--overwrite_csv", action="store_true")

    args = ap.parse_args()
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    args.amp = (not args.no_amp)
    args.use_class_weights = (not args.no_class_weights)

    # AUTO defaults
    img_auto, mask_auto, img_tried, mask_tried = auto_defaults_for_dataset(THIS_DIR, IMG_DIR, MASK_DIR)
    if not args.img_dir or not path.isdir(args.img_dir):
        args.img_dir = img_auto
    if not args.mask_dir or not path.isdir(args.mask_dir):
        args.mask_dir = mask_auto

    w_auto, _ = auto_defaults_for_weights_dir(THIS_DIR, WEIGHTS_DIR)
    if not args.weights_dir or not path.isdir(args.weights_dir):
        args.weights_dir = w_auto

    if not args.loc_weight:
        loc_w, _ = auto_find_stage1_loc_weight(THIS_DIR)
        args.loc_weight = loc_w
    if not args.loc_platt and args.loc_weight:
        args.loc_platt, _ = auto_find_stage1_platt(args.loc_weight)

    print("\n[AUTO CONFIG]")
    print(f"  img_dir      : {args.img_dir}")
    print(f"  mask_dir     : {args.mask_dir}")
    print(f"  weights_dir  : {args.weights_dir}")
    print(f"  loc_weight   : {args.loc_weight if args.loc_weight else '(NOT FOUND)'}")
    print(f"  loc_platt    : {args.loc_platt if args.loc_platt else '(none)'}")
    print(f"  amp          : {args.amp} (disable with --no_amp)")
    print(f"  fusion_p     : {args.fusion_p} (TRAIN aug only; VAL/TEST raw RGB)")
    print(f"  contrast     : alpha={args.alpha_contrast}, clahe_clip={args.clahe_clip}, clahe_grid={args.clahe_grid}")
    print(f"  edges        : alpha={args.alpha_edge}, t1={args.canny_t1}, t2={args.canny_t2}, dilate={args.edge_dilate}")
    print(f"  ensure tiles : min_val_destroyed_tiles={args.min_val_destroyed_tiles}, min_train_destroyed_tiles={args.min_train_destroyed_tiles}")
    print(f"  class weights: {args.use_class_weights} (destroyed_weight_mul={args.destroyed_weight_mul})")
    print(f"  destroyed crops: focus_p={args.focus_destroyed_p}, min_px={args.min_destroyed_px}, attempts={args.crop_attempts}")

    if not path.isdir(args.img_dir):
        print("\n[AUTO DEBUG] Tried img_dir candidates:")
        for d in img_tried:
            print("  -", d)
        raise FileNotFoundError("Could not find img_dir automatically. Pass --img_dir explicitly.")

    if not path.isdir(args.mask_dir):
        print("\n[AUTO DEBUG] Tried mask_dir candidates:")
        for d in mask_tried:
            print("  -", d)
        raise FileNotFoundError("Could not find mask_dir automatically. Pass --mask_dir explicitly.")

    if not args.loc_weight or not path.exists(args.loc_weight):
        raise FileNotFoundError(
            "Could not auto-find Stage-1 localization checkpoint (loc_weight).\n"
            "Put it in e.g. idabd_stage1_loc_ft_checkpoints/ or pass --loc_weight explicitly."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ensure_csv(args.csv_path, CSV_FIELDS_STAGE2_PIPELINE_CONTRAST_EDGE, overwrite=args.overwrite_csv)
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