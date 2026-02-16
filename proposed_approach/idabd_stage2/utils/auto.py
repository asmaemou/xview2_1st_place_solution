from __future__ import annotations
from pathlib import Path
import glob
from os import path

def ensure_import_paths(script_file: Path):
    """
    - adds proposed_approach/ to sys.path for imports like: from train...
    - adds repo root that contains /zoo to sys.path so zoo.models works
    """
    import sys

    script_dir = script_file.resolve().parent
    proj_root = script_dir.parent  # proposed_approach
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))

    # find repo root containing "zoo"
    cur = script_dir
    for _ in range(6):
        if (cur / "zoo").is_dir():
            if str(cur) not in sys.path:
                sys.path.insert(0, str(cur))
            break
        cur = cur.parent

def candidate_dirs(script_file: Path, rels: list[str]):
    script_dir = script_file.resolve().parent
    proj_root = script_dir.parent
    repo_root = proj_root.parent

    roots = [script_dir, proj_root, repo_root, Path.cwd(), Path.cwd().parent]
    out = []
    for r in roots:
        for rel in rels:
            d = (r / rel).resolve()
            if d.is_dir():
                out.append(str(d))
    # dedup preserve order
    seen, dedup = set(), []
    for d in out:
        if d not in seen:
            dedup.append(d); seen.add(d)
    return dedup

def auto_defaults_for_dataset(script_file: Path):
    img_dirs = candidate_dirs(script_file, ["idabd/images", "../idabd/images", "data/idabd/images", "dataset/idabd/images"])
    msk_dirs = candidate_dirs(script_file, ["idabd/masks",  "../idabd/masks",  "data/idabd/masks",  "dataset/idabd/masks"])
    return (img_dirs[0] if img_dirs else ""), (msk_dirs[0] if msk_dirs else "")

def auto_defaults_for_weights_dir(script_file: Path):
    wdirs = candidate_dirs(script_file, ["weights", "../weights", "checkpoints/weights", "models/weights"])
    return wdirs[0] if wdirs else ""

def auto_find_stage1_loc_weight(script_file: Path):
    loc_dirs = candidate_dirs(script_file, [
        "idabd_stage1_loc_ft_checkpoints",
        "../idabd_stage1_loc_ft_checkpoints",
        "checkpoints/idabd_stage1_loc_ft_checkpoints",
        "stage1_loc_checkpoints",
        "../stage1_loc_checkpoints",
    ])
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
    hits = sorted(glob.glob(path.join(d, "*platt*.npz")))
    return hits[0] if hits else ""

def auto_find_stage2_init_weights(weights_dir: str, include_idabd_finetune: bool):
    patterns = ["dpn92_cls*.pth", "res34_cls*.pth", "res34_cls2*.pth", "res50_cls*.pth", "se154_cls*.pth"]
    files = []
    for ptn in patterns:
        files += glob.glob(path.join(weights_dir, ptn))
    files = sorted({str(Path(f).resolve()) for f in files})
    if not include_idabd_finetune:
        files = [f for f in files if "idabd_finetune" not in Path(f).name.lower()]
    return files, patterns
