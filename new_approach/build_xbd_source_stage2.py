#!/usr/bin/env python3
import os
import argparse
import shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
def is_img(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path, mode: str, stats: dict):
    try:
        if dst.exists():
            # already there
            stats["skipped_exists"] += 1
            return
        if mode == "hardlink":
            os.link(str(src), str(dst))
            stats["hardlinked"] += 1
        else:
            shutil.copy2(str(src), str(dst))
            stats["copied"] += 1
    except Exception:
        stats["errors"] += 1

def tile_id_from_name(name: str) -> str:
    # remove suffixes used in xView2
    for suf in ["_pre_disaster", "_post_disaster"]:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name

def sanity_check(out_images: Path, out_masks: Path):
    pre = sorted([p for p in out_images.iterdir() if is_img(p) and p.stem.endswith("_pre_disaster")])
    post = sorted([p for p in out_images.iterdir() if is_img(p) and p.stem.endswith("_post_disaster")])
    masks = sorted([p for p in out_masks.iterdir() if is_img(p)])

    pre_set = {tile_id_from_name(p.stem) for p in pre}
    post_set = {tile_id_from_name(p.stem) for p in post}

    # expected mask name: {tile_id}_post_disaster.png
    missing_mask = 0
    good = 0
    for tid in pre_set:
        if tid not in post_set:
            continue
        expected = out_masks / f"{tid}_post_disaster.png"
        if expected.exists():
            good += 1
        else:
            missing_mask += 1

    print("\n=== SANITY CHECK (Stage-2 triplets) ===")
    print(f"pre_count: {len(pre)}  post_count: {len(post)}  mask_files: {len(masks)}")
    print(f"tile_ids_with_pre: {len(pre_set)}  tile_ids_with_post: {len(post_set)}")
    print(f"good_triplets (pre+post+damage_mask): {good}")
    print(f"missing_damage_mask: {missing_mask}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xbd_root", type=str, required=True, help="Path to xbd root that contains train/ and tier3/")
    ap.add_argument("--out_root", type=str, required=True, help="Output merged SOURCE folder (will create images/ masks/)")
    ap.add_argument("--mode", type=str, default="copy", choices=["copy", "hardlink"])
    ap.add_argument("--splits", nargs="+", default=["train", "tier3"])
    ap.add_argument("--clean_out", action="store_true", help="Delete output folders before building")
    ap.add_argument("--include_loc_masks", action="store_true", help="Also export *_pre_disaster.png masks into loc_masks/")
    args = ap.parse_args()

    xbd_root = Path(args.xbd_root)
    out_root = Path(args.out_root)
    out_images = out_root / "images"
    out_masks = out_root / "masks"          # damage masks only (post)
    out_loc = out_root / "loc_masks"        # optional (pre)

    if args.clean_out and out_root.exists():
        shutil.rmtree(out_root)

    safe_mkdir(out_images)
    safe_mkdir(out_masks)
    if args.include_loc_masks:
        safe_mkdir(out_loc)

    stats_img = {"copied": 0, "hardlinked": 0, "skipped_exists": 0, "errors": 0}
    stats_msk = {"copied": 0, "hardlinked": 0, "skipped_exists": 0, "errors": 0}
    stats_loc = {"copied": 0, "hardlinked": 0, "skipped_exists": 0, "errors": 0}

    for sp in args.splits:
        sp_dir = xbd_root / sp
        img_dir = sp_dir / "images"
        msk_dir = sp_dir / "masks"

        if not img_dir.exists():
            raise FileNotFoundError(f"Missing: {img_dir}")
        if not msk_dir.exists():
            raise FileNotFoundError(f"Missing: {msk_dir}")

        # 1) images: copy all pre+post
        for p in img_dir.iterdir():
            if not p.is_file() or not is_img(p):
                continue
            # keep original filenames
            dst = out_images / p.name
            link_or_copy(p, dst, args.mode, stats_img)

        # 2) damage masks: ONLY *_post_disaster.png
        for p in msk_dir.iterdir():
            if not p.is_file() or not is_img(p):
                continue
            if p.stem.endswith("_post_disaster"):
                dst = out_masks / p.name
                link_or_copy(p, dst, args.mode, stats_msk)
            elif args.include_loc_masks and p.stem.endswith("_pre_disaster"):
                dst = out_loc / p.name
                link_or_copy(p, dst, args.mode, stats_loc)

    print("\n=== BUILD DONE ===")
    print(f"OUT: {out_root}")
    print(f"images: {stats_img}")
    print(f"damage masks (_post_disaster): {stats_msk}")
    if args.include_loc_masks:
        print(f"loc masks (_pre_disaster): {stats_loc}")

    sanity_check(out_images, out_masks)

if __name__ == "__main__":
    main()
