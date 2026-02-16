#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE-2 DAMAGE (IDABD) — FUSION-ONLY (Edge+Contrast+Unsharp) + NO DA + EarlyStop
x6 = [pre_fusion_3ch, post_fusion_3ch]  (6ch for *_Unet_Double)
Pipeline eval uses Stage-1 localization gating (Platt optional).

Run:
  python proposed_approach/scripts/train_idabd_stage2_pipeline_FUSION_ONLY_ECUS_noDA_earlyStop.py
"""

from __future__ import annotations
import os
from os import path
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure repo root is on sys.path so "proposed_approach" imports work
THIS_FILE = path.abspath(__file__)
ROOT = path.abspath(path.join(path.dirname(THIS_FILE), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from proposed_approach.aug.fusion_ecus import FusionECUSConfig
from proposed_approach.models.xview2_builders import build_damage_model_from_weight, build_loc_model_from_weight
from proposed_approach.train.dataset_fusion_ecus import build_stage2_triplets, IdaBDStage2DamageFusionECUS
from proposed_approach.train.pipeline_eval_uncal import eval_loader_pipeline_uncal
from proposed_approach.utils.auto_idabd_stage2 import (
    auto_defaults_for_dataset,
    auto_defaults_for_weights_dir,
    auto_find_stage1_loc_weight,
    auto_find_stage1_platt,
    auto_find_stage2_init_weights,
)
from proposed_approach.utils.csv_simple import ensure_csv, append_csv_row

IGNORE_LABEL = 255
CLASS_NAMES_4 = ["No-Damage", "Minor", "Major", "Destroyed"]
BUILD_TENSOR_CPU = torch.tensor([1, 2, 3, 4], dtype=torch.long)


CSV_FIELDS = [
    "seed",
    "init_weight",
    "loc_weight",
    "loc_platt",
    "loc_thresh",
    "fusion_p",
    # fusion params
    "alpha_unsharp", "unsharp_sigma", "unsharp_amount",
    "alpha_contrast", "clahe_clip", "clahe_grid",
    "alpha_edge", "canny_t1", "canny_t2", "edge_dilate",
    # pipeline metrics (uncal)
    "pipeline_acc_uncal",
    "pipeline_loc_precision_uncal",
    "pipeline_loc_recall_uncal",
    "pipeline_loc_f1_uncal",
    "pipeline_macroF1_damage_uncal",
    "pipeline_f1_no_damage_uncal",
    "pipeline_f1_minor_uncal",
    "pipeline_f1_major_uncal",
    "pipeline_f1_destroyed_uncal",
]


def masked_ce_loss(logits, raw_mask, ce_loss: nn.Module):
    target = torch.full_like(raw_mask, IGNORE_LABEL)
    build_tensor = BUILD_TENSOR_CPU.to(raw_mask.device)
    valid = (raw_mask != IGNORE_LABEL)
    build = valid & torch.isin(raw_mask, build_tensor)
    target[build] = raw_mask[build]  # 1..4
    if build.sum().item() == 0:
        return None
    return ce_loss(logits, target)


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
    return (float(np.mean(losses)) if losses else 1e9), skipped


def train_and_eval_one(args, init_weight: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] {device}")
    print(f"[INIT ] {init_weight}")

    # Stage-1 loc
    loc_model = build_loc_model_from_weight(args.loc_weight).to(device).eval()
    for p in loc_model.parameters():
        p.requires_grad = False

    loc_a, loc_b = 1.0, 0.0
    if args.loc_platt:
        d = np.load(args.loc_platt)
        loc_a, loc_b = float(d["a"]), float(d["b"])

    # triplets + split
    triplets = build_stage2_triplets(args.img_dir, args.mask_dir, gt_split=args.gt_split)
    if not triplets:
        raise FileNotFoundError("No (pre, post, mask) triplets found.")

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
    print(f"[DATA ] total={n} | train={len(train_tr)} | val={len(val_tr)} | test={len(test_tr)}")

    cfg = FusionECUSConfig(
        alpha_unsharp=args.alpha_unsharp,
        unsharp_sigma=args.unsharp_sigma,
        unsharp_amount=args.unsharp_amount,
        alpha_contrast=args.alpha_contrast,
        clahe_clip=args.clahe_clip,
        clahe_grid=args.clahe_grid,
        alpha_edge=args.alpha_edge,
        canny_t1=args.canny_t1,
        canny_t2=args.canny_t2,
        edge_dilate=args.edge_dilate,
    )

    train_ds = IdaBDStage2DamageFusionECUS(
        train_tr, cfg=cfg, crop_size=args.crop, is_train=True,
        fusion_p=args.fusion_p, min_build_px=args.min_build_px, seed=args.seed,
        focus_destroyed_p=args.focus_destroyed_p, min_destroyed_px=args.min_destroyed_px,
        crop_attempts=args.crop_attempts,
    )
    val_ds = IdaBDStage2DamageFusionECUS(val_tr, cfg=cfg, crop_size=0, is_train=False, fusion_p=1.0, seed=args.seed)
    test_ds = IdaBDStage2DamageFusionECUS(test_tr, cfg=cfg, crop_size=0, is_train=False, fusion_p=1.0, seed=args.seed)

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=max(0, args.workers // 2), pin_memory=True)
    test_ld  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=max(0, args.workers // 2), pin_memory=True)

    # Stage-2 model
    model = build_damage_model_from_weight(init_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL, reduction="mean")

    # scaler + autocast
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp) if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") \
                 else torch.cuda.amp.GradScaler(enabled=args.amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    if args.amp and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        def amp_ctx():
            return torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"))
    else:
        def amp_ctx():
            return torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp))

    os.makedirs(args.out_dir, exist_ok=True)
    base = path.splitext(path.basename(init_weight))[0]
    best_ckpt_path = path.join(args.out_dir, f"{base}_idabd_stage2_ft_best.pth")
    last_ckpt_path = path.join(args.out_dir, f"{base}_idabd_stage2_ft_last.pth")

    best_val = 1e9
    best_epoch = -1
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
        val_loss, val_skipped = eval_val_loss(model, val_ld, device, ce_loss, amp=args.amp)

        print(f"[EPOCH {epoch:03d}/{args.epochs}] train_loss={train_loss:.5f} (skipped={skipped}) | "
              f"val_loss={val_loss:.5f} (skipped={val_skipped})")

        torch.save(
            {"epoch": epoch, "init_weight": init_weight, "state_dict": model.state_dict(),
             "args": vars(args), "train_loss": train_loss, "val_loss": val_loss},
            last_ckpt_path
        )

        improved = (val_loss < (best_val - args.early_stop_min_delta))
        if improved:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {"epoch": epoch, "init_weight": init_weight, "state_dict": model.state_dict(),
                 "args": vars(args), "train_loss": train_loss, "val_loss": val_loss},
                best_ckpt_path
            )
            print(f"[SAVE ] Best (val) -> {best_ckpt_path} (best_val={best_val:.6f})")
        else:
            if epoch >= args.early_stop_warmup:
                no_improve += 1

        if epoch >= args.early_stop_warmup and args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"[EARLY STOP] No VAL improvement for {args.early_stop_patience} epochs. "
                  f"Stopping at epoch {epoch}. Best was epoch {best_epoch} (val={best_val:.6f}).")
            break

    # Load best and pipeline eval (uncal only)
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["state_dict"], strict=True)
    model.eval()
    print(f"\n[LOAD ] BEST checkpoint for test: {best_ckpt_path}")

    acc_u, macro_u, f1_u, locP_u, locR_u, locF_u = eval_loader_pipeline_uncal(
        model, test_ld, device, loc_model=loc_model, loc_a=loc_a, loc_b=loc_b,
        loc_thresh=args.loc_thresh, amp=args.amp
    )

    print("\n==================== PIPELINE TEST (UNCALIBRATED ONLY) ====================")
    print(f"pipeline_acc(0..4)={acc_u:.6f}")
    print(f"F1 Localization={locF_u:.6f} (P={locP_u:.6f}, R={locR_u:.6f})")
    print(f"macroF1(Damage 1..4)={macro_u:.6f}")
    for i in range(4):
        print(f"F1 {CLASS_NAMES_4[i]:>9s}: {f1_u[i]:.6f}")
    print("==========================================================================\n")

    row = {
        "seed": args.seed,
        "init_weight": init_weight,
        "loc_weight": args.loc_weight,
        "loc_platt": args.loc_platt,
        "loc_thresh": args.loc_thresh,
        "fusion_p": args.fusion_p,
        **cfg.to_dict(),
        "pipeline_acc_uncal": acc_u,
        "pipeline_loc_precision_uncal": locP_u,
        "pipeline_loc_recall_uncal": locR_u,
        "pipeline_loc_f1_uncal": locF_u,
        "pipeline_macroF1_damage_uncal": macro_u,
        "pipeline_f1_no_damage_uncal": f1_u[0],
        "pipeline_f1_minor_uncal": f1_u[1],
        "pipeline_f1_major_uncal": f1_u[2],
        "pipeline_f1_destroyed_uncal": f1_u[3],
    }
    append_csv_row(args.csv_path, CSV_FIELDS, row)
    print(f"[CSV  ] Appended -> {args.csv_path}")


def main():
    ap = argparse.ArgumentParser()

    # dataset / paths
    ap.add_argument("--img_dir", default="")
    ap.add_argument("--mask_dir", default="")
    ap.add_argument("--weights_dir", default="")
    ap.add_argument("--gt_split", choices=["pre", "post"], default="post")

    # split
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)

    # train
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")

    # early stopping
    ap.add_argument("--early_stop_patience", type=int, default=15)
    ap.add_argument("--early_stop_warmup", type=int, default=5)
    ap.add_argument("--early_stop_min_delta", type=float, default=0.0)

    # crops / destroyed-aware
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--min_build_px", type=int, default=50)
    ap.add_argument("--focus_destroyed_p", type=float, default=0.6)
    ap.add_argument("--min_destroyed_px", type=int, default=50)
    ap.add_argument("--crop_attempts", type=int, default=20)

    # fusion probability
    ap.add_argument("--fusion_p", type=float, default=0.7)

    # Fusion ECUS params
    ap.add_argument("--alpha_unsharp", type=float, default=0.7)
    ap.add_argument("--unsharp_sigma", type=float, default=1.2)
    ap.add_argument("--unsharp_amount", type=float, default=1.0)
    ap.add_argument("--alpha_contrast", type=float, default=0.6)
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_grid", type=int, default=8)
    ap.add_argument("--alpha_edge", type=float, default=0.6)
    ap.add_argument("--canny_t1", type=int, default=50)
    ap.add_argument("--canny_t2", type=int, default=150)
    ap.add_argument("--edge_dilate", type=int, default=2)

    # weights selection
    ap.add_argument("--init_weight", default="")
    ap.add_argument("--include_idabd_finetune", action="store_true")

    # output
    ap.add_argument("--out_dir", default="idabd_stage2_damage_ft_checkpoints_FUSION_ONLY_ECUS_NO_DA")
    ap.add_argument("--csv_path", default="idabd_stage2_pipeline_results_FUSION_ONLY_ECUS_NO_DA.csv")
    ap.add_argument("--overwrite_csv", action="store_true")

    # stage-1 gating
    ap.add_argument("--loc_weight", type=str, default="")
    ap.add_argument("--loc_platt", type=str, default="")
    ap.add_argument("--loc_thresh", type=float, default=0.5)

    args = ap.parse_args()
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    # AUTO paths
    IMG_FALLBACK = "../idabd/images"
    MASK_FALLBACK = "../idabd/masks"
    WEIGHTS_FALLBACK = "weights"

    img_auto, mask_auto = auto_defaults_for_dataset(__file__, IMG_FALLBACK, MASK_FALLBACK)
    if not args.img_dir or not path.isdir(args.img_dir):
        args.img_dir = img_auto
    if not args.mask_dir or not path.isdir(args.mask_dir):
        args.mask_dir = mask_auto
    if not args.weights_dir or not path.isdir(args.weights_dir):
        args.weights_dir = auto_defaults_for_weights_dir(__file__, WEIGHTS_FALLBACK)

    if not args.loc_weight:
        args.loc_weight = auto_find_stage1_loc_weight(__file__)
    if not args.loc_platt and args.loc_weight:
        args.loc_platt = auto_find_stage1_platt(args.loc_weight)

    print("\n[AUTO CONFIG]")
    print(f"  img_dir     : {args.img_dir}")
    print(f"  mask_dir    : {args.mask_dir}")
    print(f"  weights_dir : {args.weights_dir}")
    print(f"  loc_weight  : {args.loc_weight if args.loc_weight else '(NOT FOUND)'}")
    print(f"  loc_platt   : {args.loc_platt if args.loc_platt else '(none)'}")

    if not path.isdir(args.img_dir):
        raise FileNotFoundError("Could not find img_dir. Pass --img_dir explicitly.")
    if not path.isdir(args.mask_dir):
        raise FileNotFoundError("Could not find mask_dir. Pass --mask_dir explicitly.")
    if not args.loc_weight or not path.exists(args.loc_weight):
        raise FileNotFoundError("Could not find Stage-1 loc checkpoint. Pass --loc_weight explicitly.")

    # seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ensure_csv(args.csv_path, CSV_FIELDS, overwrite=args.overwrite_csv)
    print(f"[CSV  ] Ready: {args.csv_path}")

    # run one or all
    if args.init_weight:
        if not path.exists(args.init_weight):
            raise FileNotFoundError(args.init_weight)
        train_and_eval_one(args, args.init_weight)
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
        train_and_eval_one(args, w)


if __name__ == "__main__":
    main()