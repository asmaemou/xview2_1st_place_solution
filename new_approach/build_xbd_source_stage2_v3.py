#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path
from collections import defaultdict

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def tile_id_from_name(name: str) -> str:
    # removes known suffixes to obtain stable tile_id
    stem = Path(name).stem
    for suf in ["_pre_disaster", "_post_disaster", "_pre", "_post", "_mask", "_label"]:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    return stem

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_or_link(src: Path, dst: Path, mode: str):
    try:
        if mode == "hardlink":
            if dst.exists():
                return "skipped_exists"
            os.link(str(src), str(dst))
            return "hardlinked"
        else:
            if dst.exists():
                return "skipped_exists"
            shutil.copy2(str(src), str(dst))
            return "copied"
    except Exception:
        # fallback to copy
        try:
            if not dst.exists():
                shutil.copy2(str(src), str(dst))
                return "copied"
            return "skipped_exists"
        except Exception:
            return "errors"

def list_files(folder: Path):
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

def choose_mask_folder(split_root: Path, prefer: str = "auto") -> Path | None:
    """
    prefer:
      - auto: prefer targets if non-empty, else masks if non-empty
      - targets: force targets
      - masks: force masks
    """
    masks_dir = split_root / "masks"
    targets_dir = split_root / "targets"

    masks_files = list_files(masks_dir)
    targets_files = list_files(targets_dir)

    if prefer == "targets":
        return targets_dir if targets_files else None
    if prefer == "masks":
        return masks_dir if masks_files else None

    # auto
    if targets_files:
        return targets_dir
    if masks_files:
        return masks_dir
    return None

def build_maps(mask_dir: Path):
    """
    Build mapping tile_id -> actual mask file.
    Allows masks named either:
      tile_id_post_disaster.png
      tile_id.png
      tile_id_anything.png (we strip suffixes)
    """
    m = {}
    for f in list_files(mask_dir):
        tid = tile_id_from_name(f.name)
        # keep first occurrence; you can change policy if needed
        if tid not in m:
            m[tid] = f
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xbd_root", type=str, required=True, help="Path to xbd root that contains train/ tier3/")
    ap.add_argument("--out_root", type=str, required=True, help="Output folder (e.g., ./xbd_source)")
    ap.add_argument("--mode", type=str, default="copy", choices=["copy", "hardlink"])
    ap.add_argument("--clean_out", action="store_true", help="Delete out_root before building")
    ap.add_argument("--mask_source", type=str, default="auto", choices=["auto", "masks", "targets"],
                    help="Where to read damage masks from inside each split (auto prefers targets then masks)")
    ap.add_argument("--splits", type=str, default="train,tier3", help="Comma list of splits to merge")
    args = ap.parse_args()

    xbd_root = Path(args.xbd_root)
    out_root = Path(args.out_root)

    if args.clean_out and out_root.exists():
        shutil.rmtree(out_root)

    out_images = out_root / "images"
    out_masks = out_root / "masks"
    safe_mkdir(out_images)
    safe_mkdir(out_masks)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    ops = defaultdict(int)

    total_post = 0
    good_triplets = 0
    missing_mask = 0

    for split in splits:
        split_root = xbd_root / split
        img_dir = split_root / "images"
        if not img_dir.exists():
            print(f"[WARN] Missing images folder: {img_dir}")
            continue

        # choose mask folder
        mask_dir = choose_mask_folder(split_root, prefer=args.mask_source)
        if mask_dir is None:
            print(f"[WARN] No mask folder found for split={split}. Tried targets/ and masks/.")
            mask_map = {}
        else:
            mask_map = build_maps(mask_dir)
            print(f"[INFO] split={split} mask_dir={mask_dir} mask_files={len(mask_map)}")

        # gather images
        images = list_files(img_dir)
        if not images:
            print(f"[WARN] No images found in {img_dir}")
            continue

        # split pre vs post
        pre = [p for p in images if "_pre_disaster" in p.stem]
        post = [p for p in images if "_post_disaster" in p.stem]

        pre_map = {tile_id_from_name(p.name): p for p in pre}
        post_map = {tile_id_from_name(p.name): p for p in post}

        keys = sorted(set(pre_map.keys()) & set(post_map.keys()))
        print(f"[INFO] split={split} pairs(pre+post)={len(keys)}")

        for tid in keys:
            pre_src = pre_map[tid]
            post_src = post_map[tid]

            # copy images
            ops[copy_or_link(pre_src, out_images / pre_src.name, args.mode)] += 1
            ops[copy_or_link(post_src, out_images / post_src.name, args.mode)] += 1

            total_post += 1

            # find mask
            msrc = mask_map.get(tid, None)
            if msrc is None:
                missing_mask += 1
                continue

            # force standard name so training script can find it with --gt_split post
            dst_name = f"{tid}_post_disaster{msrc.suffix.lower()}"
            ops[copy_or_link(msrc, out_masks / dst_name, args.mode)] += 1
            good_triplets += 1

    print("\n=== BUILD DONE ===")
    print(f"OUT: {out_root}")
    print(f"ops: {dict(ops)}")

    print("\n=== SANITY CHECK (Stage-2 triplets) ===")
    pre_count = len([p for p in out_images.iterdir() if p.is_file() and "_pre_disaster" in p.stem])
    post_count = len([p for p in out_images.iterdir() if p.is_file() and "_post_disaster" in p.stem])
    mask_count = len([p for p in out_masks.iterdir() if p.is_file()])
    print(f"pre_count: {pre_count}  post_count: {post_count}  mask_files: {mask_count}")
    print(f"expected_post_pairs: {total_post}")
    print(f"good_triplets(pre+post+mask): {good_triplets}")
    print(f"missing_damage_mask: {missing_mask}")

if __name__ == "__main__":
    main()
