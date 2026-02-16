#!/usr/bin/env python3
import os
from os import path
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

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
from idabd_stage2.train.split import split_triplets, ensure_destroyed_in_splits
from idabd_stage2.utils.csv_utils import ensure_csv, append_csv_row

from idabd_stage2.aug.unsharp_only import post_unsharp_only

CSV_FIELDS = [
    "seed","init_weight","loc_weight","loc_platt","loc_thresh",
    "unsharp_alpha","unsharp_amount","unsharp_sigma","unsharp_p",
    "pipeline_acc_uncal","pipeline_loc_precision_uncal","pipeline_loc_recall_uncal","pipeline_loc_f1_uncal",
    "pipeline_macroF1_damage_uncal","pipeline_f1_no_damage_uncal","pipeline_f1_minor_uncal",
    "pipeline_f1_major_uncal","pipeline_f1_destroyed_uncal",
    "pipeline_acc_cal","pipeline_loc_precision_cal","pipeline_loc_recall_cal","pipeline_loc_f1_cal",
    "pipeline_macroF1_damage_cal","pipeline_f1_no_damage_cal","pipeline_f1_minor_cal",
    "pipeline_f1_major_cal","pipeline_f1_destroyed_cal",
]

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
    ap.add_argument("--amp", action="store_true")  # keep as a flag like your original unsharp-only script

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

    # Unsharp-only config
    ap.add_argument("--unsharp_p", type=float, default=1.0)
    ap.add_argument("--unsharp_alpha", type=float, default=0.6)
    ap.add_argument("--unsharp_amount", type=float, default=1.0)
    ap.add_argument("--unsharp_sigma", type=float, default=1.0)

    ap.add_argument("--min_val_destroyed_tiles", type=int, default=1)
    ap.add_argument("--min_train_destroyed_tiles", type=int, default=1)

    ap.add_argument("--init_weight", default="")  # if empty -> run all
    ap.add_argument("--include_idabd_finetune", action="store_true")

    ap.add_argument("--out_dir", default="idabd_stage2_damage_ft_checkpoints_RGB_UNSHARP_ONLY")
    ap.add_argument("--loc_weight", default="")
    ap.add_argument("--loc_platt", default="")
    ap.add_argument("--loc_thresh", type=float, default=0.5)

    ap.add_argument("--csv_path", default="idabd_stage2_pipeline_results_RGB_UNSHARP_ONLY.csv")
    ap.add_argument("--overwrite_csv", action="store_true")

    args = ap.parse_args()

    ensure_csv(args.csv_path, CSV_FIELDS, overwrite=args.overwrite_csv)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # stage-1 loc auto-find
    if not args.loc_weight:
        args.loc_weight = auto_find_stage1_loc_weight(Path(__file__))
    if not args.loc_platt:
        args.loc_platt = auto_find_stage1_platt(args.stage1_dir)

    if not args.loc_weight or not path.exists(args.loc_weight):
        raise FileNotFoundError("Stage-1 loc checkpoint not found. Pass --loc_weight explicitly.")

    loc_model = build_loc_model_from_weight(args.loc_weight).to(device).eval()
    for p in loc_model.parameters(): p.requires_grad = False

    loc_a, loc_b = 1.0, 0.0
    if args.loc_platt and path.exists(args.loc_platt):
        d = np.load(args.loc_platt)
        loc_a, loc_b = float(d["a"]), float(d["b"])

    init_weights = [args.init_weight] if args.init_weight else auto_find_stage2_init_weights(
        args.weights_dir, include_idabd_finetune=args.include_idabd_finetune
    )
    if isinstance(init_weights, tuple):
        init_weights = init_weights[0]   # take the actual list of paths

    flat = []
    for w in init_weights:
        if isinstance(w, (list, tuple)):
            # if it's (path, score) or [path, ...], take first; if it's a nested list, flatten
            if len(w) > 0 and isinstance(w[0], (str, os.PathLike, Path)):
                flat.append(str(w[0]))
            else:
                for ww in w:
                    flat.append(str(ww))
        else:
            flat.append(str(w))

    init_weights = flat
    # ---------------------------------------------------------------

    if not init_weights:
        raise FileNotFoundError("No stage-2 init weights found. Pass --init_weight or check --weights_dir.")

    triplets = build_stage2_triplets(args.img_dir, args.mask_dir, gt_split=args.gt_split)
    if not triplets:
        raise FileNotFoundError("No (pre, post, mask) triplets found.")

    train_tr, val_tr, test_tr = split_triplets(triplets, args.seed, args.val_ratio, args.test_ratio)
    ensure_destroyed_in_splits(
        train_tr, val_tr,
        min_train=int(args.min_train_destroyed_tiles),
        min_val=int(args.min_val_destroyed_tiles),
    )

    def unsharp_fn(post_bgr):
        return post_unsharp_only(
            post_bgr,
            alpha=args.unsharp_alpha,
            amount=args.unsharp_amount,
            sigma=args.unsharp_sigma,
        )

    # UNsharp is part of the *input definition* for this experiment:
    # - Train: apply with prob unsharp_p
    # - Eval: apply iff unsharp_p >= 1.0, else raw (same as your original logic)
    eval_tf = unsharp_fn if args.unsharp_p >= 1.0 else None

    train_ds = IdaBDStage2Dataset6CH(
        train_tr, crop_size=args.crop, is_train=True, seed=args.seed,
        min_build_px=args.min_build_px,
        focus_destroyed_p=args.focus_destroyed_p, min_destroyed_px=args.min_destroyed_px,
        crop_attempts=args.crop_attempts,
        post_p_train=args.unsharp_p,
        post_transform_train=unsharp_fn,
        post_transform_eval=eval_tf,
    )
    val_ds  = IdaBDStage2Dataset6CH(val_tr,  crop_size=0, is_train=False, seed=args.seed,
                                   post_transform_eval=eval_tf)
    test_ds = IdaBDStage2Dataset6CH(test_tr, crop_size=0, is_train=False, seed=args.seed,
                                   post_transform_eval=eval_tf)

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                          pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=max(0, args.workers//2),
                          pin_memory=True)
    test_ld  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=max(0, args.workers//2),
                          pin_memory=True)

    for init_w in init_weights:
        if not path.exists(init_w):
            raise FileNotFoundError(init_w)

        if args.use_class_weights:
            w4, _ = compute_class_weights_from_triplets(train_tr, max_images=args.weight_count_images, seed=args.seed)
            w4[3] *= float(args.destroyed_weight_mul)
            weight5 = np.ones(5, dtype=np.float32); weight5[1:5] = w4
            ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL, reduction="mean",
                                          weight=torch.from_numpy(weight5).to(device))
        else:
            ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL, reduction="mean")

        model = build_damage_model_from_weight(init_w).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        os.makedirs(args.out_dir, exist_ok=True)
        base = path.splitext(path.basename(init_w))[0]
        best_ckpt = path.join(args.out_dir, f"{base}_idabd_stage2_ft_best_6ch.pth")
        last_ckpt = path.join(args.out_dir, f"{base}_idabd_stage2_ft_last_6ch.pth")
        calib_npz = path.join(args.out_dir, f"{base}_idabd_stage2_calib_vector_scaling_6ch.npz")

        train_with_checkpoints(
            model, train_ld, val_ld, device, ce_loss, opt,
            epochs=args.epochs, amp=args.amp,
            last_ckpt_path=last_ckpt, best_ckpt_path=best_ckpt,
            ckpt_extra={"init_weight": init_w, "args": vars(args), "stage2_in_channels": 6},
            early_stop_patience=args.early_stop_patience,
            early_stop_warmup=args.early_stop_warmup,
            early_stop_min_delta=args.early_stop_min_delta,
        )

        best = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(best["state_dict"], strict=False)
        model.eval()

        # calibration is fit on VAL logits for *this experiment’s* input definition
        X, Y, _ = collect_val_logits_balanced(model, val_ld, device,
                                              per_class_pixels=args.calib_pixels_per_class, seed=args.seed)
        W_np, b_np = fit_vector_scaling(X, Y, device, lr=args.calib_lr, epochs=args.calib_epochs,
                                        batch_size=args.calib_batch, wd=args.calib_wd)
        np.savez(calib_npz, W=W_np, b=b_np)
        calib = (torch.from_numpy(W_np).to(device), torch.from_numpy(b_np).to(device))

        acc_u, macro_u, f1_u, locP_u, locR_u, locF_u = eval_loader_pipeline(
            model, test_ld, device, loc_model, loc_a, loc_b, args.loc_thresh, calib=None, amp=args.amp
        )
        acc_c, macro_c, f1_c, locP_c, locR_c, locF_c = eval_loader_pipeline(
            model, test_ld, device, loc_model, loc_a, loc_b, args.loc_thresh, calib=calib, amp=args.amp
        )

        row = {
            "seed": args.seed, "init_weight": init_w, "loc_weight": args.loc_weight,
            "loc_platt": args.loc_platt, "loc_thresh": args.loc_thresh,

            "unsharp_alpha": args.unsharp_alpha,
            "unsharp_amount": args.unsharp_amount,
            "unsharp_sigma": args.unsharp_sigma,
            "unsharp_p": args.unsharp_p,

            "pipeline_acc_uncal": acc_u,
            "pipeline_loc_precision_uncal": locP_u, "pipeline_loc_recall_uncal": locR_u, "pipeline_loc_f1_uncal": locF_u,
            "pipeline_macroF1_damage_uncal": macro_u,
            "pipeline_f1_no_damage_uncal": f1_u[0], "pipeline_f1_minor_uncal": f1_u[1],
            "pipeline_f1_major_uncal": f1_u[2], "pipeline_f1_destroyed_uncal": f1_u[3],

            "pipeline_acc_cal": acc_c,
            "pipeline_loc_precision_cal": locP_c, "pipeline_loc_recall_cal": locR_c, "pipeline_loc_f1_cal": locF_c,
            "pipeline_macroF1_damage_cal": macro_c,
            "pipeline_f1_no_damage_cal": f1_c[0], "pipeline_f1_minor_cal": f1_c[1],
            "pipeline_f1_major_cal": f1_c[2], "pipeline_f1_destroyed_cal": f1_c[3],
        }
        append_csv_row(args.csv_path, CSV_FIELDS, row)

        print("\nPer-damage F1 (calibrated):")
        for i, name in enumerate(CLASS_NAMES_4):
            print(f"  F1 {name:>9s}: {f1_c[i]:.6f}")

if __name__ == "__main__":
    main()