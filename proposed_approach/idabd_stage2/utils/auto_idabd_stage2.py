# proposed_approach/utils/auto_idabd_stage2.py
from __future__ import annotations
from pathlib import Path
from os import path
import glob


def _script_dir(file: str) -> Path:
    return Path(file).resolve().parent


def _candidate_dirs(file: str, *rel_paths):
    roots = [_script_dir(file), _script_dir(file).parent, _script_dir(file).parent.parent, Path.cwd(), Path.cwd().parent]
    out = []
    for r in roots:
        for rel in rel_paths:
            d = (r / rel).resolve()
            if d.is_dir():
                out.append(str(d))
    seen, dedup = set(), []
    for d in out:
        if d not in seen:
            dedup.append(d)
            seen.add(d)
    return dedup


def auto_defaults_for_dataset(file: str, fallback_img: str, fallback_mask: str):
    img_dirs = _candidate_dirs(file, "idabd/images", "../idabd/images", "data/idabd/images", "dataset/idabd/images")
    mask_dirs = _candidate_dirs(file, "idabd/masks", "../idabd/masks", "data/idabd/masks", "dataset/idabd/masks")
    return (img_dirs[0] if img_dirs else fallback_img), (mask_dirs[0] if mask_dirs else fallback_mask)


def auto_defaults_for_weights_dir(file: str, fallback: str):
    wdirs = _candidate_dirs(file, "weights", "../weights", "checkpoints/weights", "models/weights")
    return (wdirs[0] if wdirs else fallback)


def auto_find_stage1_loc_weight(file: str):
    loc_dirs = _candidate_dirs(
        file,
        "idabd_stage1_loc_ft_checkpoints",
        "../idabd_stage1_loc_ft_checkpoints",
        "checkpoints/idabd_stage1_loc_ft_checkpoints",
        "stage1_loc_checkpoints",
        "../stage1_loc_checkpoints",
    )
    patterns = ["*_idabd_ft_best.pth", "*ft_best*.pth", "*best*.pth", "*.pth"]
    for d in loc_dirs:
        for ptn in patterns:
            hits = sorted(glob.glob(path.join(d, ptn)))
            if hits:
                return hits[0]
    return ""


def auto_find_stage1_platt(loc_weight_path: str):
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
            return cand
    hits = sorted(glob.glob(path.join(d, "*platt*.npz")))
    return hits[0] if hits else ""


def auto_find_stage2_init_weights(weights_dir: str):
    patterns = ["dpn92_cls*.pth", "res34_cls*.pth", "res34_cls2*.pth", "res50_cls*.pth", "se154_cls*.pth"]
    files = []
    for ptn in patterns:
        files += glob.glob(path.join(weights_dir, ptn))
    # dedup preserve order
    out, seen = [], set()
    for f in sorted(files):
        ff = str(Path(f).resolve())
        if ff not in seen:
            out.append(ff)
            seen.add(ff)
    return out, patterns