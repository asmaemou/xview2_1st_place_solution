#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
from os import path
from pathlib import Path

from proposed_approach.utils.paths import find_repo_root, ensure_repo_on_syspath, abs_from_repo
from proposed_approach.utils.seed import seed_everything
from proposed_approach.utils.csv_log import ensure_csv
from proposed_approach.train.pipeline_runner import gather_init_weights, run_one_init_weight

def auto_find_loc_weight(repo_root: Path) -> str:
    cand_dir = repo_root / "idabd_stage1_loc_ft_checkpoints"
    if not cand_dir.is_dir():
        return ""
    hits = sorted(glob.glob(str(cand_dir / "*_idabd_ft_best.pth")))
    return hits[0] if hits else ""

def auto_find_loc_platt(repo_root: Path) -> str:
    cand_dir = repo_root / "idabd_stage1_loc_ft_checkpoints"
    if not cand_dir.is_dir():
        return ""
    hits = sorted(glob.glob(str(cand_dir / "*_idabd_platt.npz")))
    return hits[0] if hits else ""

def build_args():
    repo_root = find_repo_root(Path(__file__).resolve())
    ensure_repo_on_syspath(repo_root)

    default_loc_weight = auto_find_loc_weight(repo_root)
    default_loc_platt  = auto_find_loc_platt(repo_root)

    ap = argparse.ArgumentParser()

    # paths (resolved relative to repo root automatically)
    ap.add_argument("--img_dir", default=str((repo_root / "idabd" / "images").resolve()))
    ap.add_argument("--mask_dir", default=str((repo_root / "idabd" / "masks").resolve()))
    ap.add_argument("--weights_dir", default=str((repo_root / "weights").resolve()))

    ap.add_argument("--gt_split", choices=["pre", "post"], default="post")

    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")

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
    ap.add_argument("--train_all", action="store_true")

    ap.add_argument("--include_idabd_finetune", action="store_true")

    ap.add_argument("--out_dir", default=str((repo_root / "idabd_stage2_damage_ft_checkpoints_retrained_destroyed").resolve()))

    ap.add_argument("--min_val_destroyed_tiles", type=int, default=1)
    ap.add_argument("--min_train_destroyed_tiles", type=int, default=1)

    # Stage-1 loc (auto-filled so you can run with no args)
    ap.add_argument("--loc_weight", type=str, default=default_loc_weight)
    ap.add_argument("--loc_platt", type=str, default=default_loc_platt)
    ap.add_argument("--loc_thresh", type=float, default=0.5)

    ap.add_argument("--csv_path", default=str((repo_root / "idabd_stage2_pipeline_results_DESTROYED_DEFENSIBLE.csv").resolve()))
    ap.add_argument("--overwrite_csv", action="store_true")

    args = ap.parse_args()
    return args

def main():
    args = build_args()

    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    if not args.loc_weight or not path.exists(args.loc_weight):
        raise ValueError(
            "Stage-1 loc weight is required.\n"
            "Either:\n"
            "  - Put a file like idabd_stage1_loc_ft_checkpoints/*_idabd_ft_best.pth\n"
            "or:\n"
            "  - Pass --loc_weight <path-to-stage1-best.pth>\n"
        )

    seed_everything(args.seed)

    ensure_csv(args.csv_path, overwrite=args.overwrite_csv)
    print(f"[CSV  ] Ready: {args.csv_path}")

    if args.train_all:
        weight_files = gather_init_weights(args.weights_dir, include_idabd_finetune=args.include_idabd_finetune)
        if not weight_files:
            raise FileNotFoundError(f"No cls weights found in {args.weights_dir}")

        print("[WEIGHTS] Stage-2 init checkpoints to run:")
        for w in weight_files:
            print("  -", w)

        for w in weight_files:
            run_one_init_weight(args, w)
    else:
        if not args.init_weight:
            raise ValueError("Provide --init_weight <path> OR use --train_all")
        if not path.exists(args.init_weight):
            raise FileNotFoundError(args.init_weight)
        run_one_init_weight(args, args.init_weight)

if __name__ == "__main__":
    main()