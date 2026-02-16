#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE-2 DAMAGE (IDABD) — FUSION-ONLY (Edge+Contrast+Unsharp) + NO DA + EarlyStop
x6 = [pre_fusion_3ch, post_fusion_3ch]  (6ch for *_Unet_Double)
Pipeline eval uses Stage-1 localization gating (Platt optional).

Run (from inside proposed_approach/):
  python train_idabd_stage2_pipeline_FUSION_ONLY_ECUS_noDA_earlyStop.py --workers 0

Also works from repo root:
  python proposed_approach/train_idabd_stage2_pipeline_FUSION_ONLY_ECUS_noDA_earlyStop.py --workers 0
"""

from __future__ import annotations

import os
from os import path
import sys
import argparse
import random
from pathlib import Path
import numpy as np
import types
import importlib.util
import inspect

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import multiprocessing as mp


# =============================================================================
# sys.path bootstrap + compatibility aliases
# =============================================================================
THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent  # .../proposed_approach (script is inside proposed_approach/)

def _add_to_syspath(p: Path):
    p = p.resolve()
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# Always add script dir (proposed_approach) so `import idabd_stage2...` works
_add_to_syspath(SCRIPT_DIR)

# Find repo root (folder that CONTAINS "proposed_approach/")
REPO_ROOT = None
for cand in [SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent, SCRIPT_DIR.parent.parent.parent]:
    if (cand / "proposed_approach").is_dir():
        REPO_ROOT = cand.resolve()
        break

# Add repo root and proposed_approach to sys.path to support both run modes
if REPO_ROOT is not None:
    _add_to_syspath(REPO_ROOT)                       # enables `import proposed_approach...`
    _add_to_syspath(REPO_ROOT / "proposed_approach") # enables `import idabd_stage2...`
else:
    _add_to_syspath(SCRIPT_DIR.parent)

# Directory where your stage2 package lives
IDABD_STAGE2_DIR = (REPO_ROOT / "proposed_approach" / "idabd_stage2") if REPO_ROOT else (SCRIPT_DIR / "idabd_stage2")
if not IDABD_STAGE2_DIR.is_dir():
    IDABD_STAGE2_DIR = (SCRIPT_DIR / "idabd_stage2")

def _ensure_pkg(modname: str, pkg_dir: Path | None = None):
    """
    Create a package module in sys.modules with __path__ pointing to pkg_dir.
    This lets legacy imports like `proposed_approach.aug.*` resolve even though
    the real code is under `idabd_stage2/*`.
    """
    if modname in sys.modules:
        return sys.modules[modname]
    m = types.ModuleType(modname)
    if pkg_dir is not None:
        m.__path__ = [str(pkg_dir.resolve())]
    else:
        m.__path__ = []
    sys.modules[modname] = m
    return m

# ---- Compatibility layer ----
_proposed_dir = (REPO_ROOT / "proposed_approach") if REPO_ROOT else SCRIPT_DIR
_ensure_pkg("proposed_approach", _proposed_dir)

# map "proposed_approach.<subpkg>" -> "proposed_approach/idabd_stage2/<subpkg>"
for sub in ["aug", "models", "train", "utils", "data"]:
    tgt = IDABD_STAGE2_DIR / sub
    if tgt.is_dir():
        _ensure_pkg(f"proposed_approach.{sub}", tgt)

# (Optional) also expose proposed_approach.idabd_stage2 as a package for convenience
if IDABD_STAGE2_DIR.is_dir():
    _ensure_pkg("proposed_approach.idabd_stage2", IDABD_STAGE2_DIR)


# =============================================================================
# Dynamic loader: find + import the pipeline eval function (names differ across repos)
# =============================================================================
def _load_module_from_file(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {file_path}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)
    return m

def _find_pipeline_eval_py():
    """
    Search for pipeline eval python file in idabd_stage2/train (preferred),
    else anywhere under idabd_stage2.
    """
    candidates = []
    train_dir = IDABD_STAGE2_DIR / "train"
    if train_dir.is_dir():
        candidates += sorted(train_dir.glob("pipeline_eval*.py"))
        candidates += sorted(train_dir.glob("*pipeline*eval*.py"))

    # fallback broader search (still local)
    if not candidates and IDABD_STAGE2_DIR.is_dir():
        candidates += sorted(IDABD_STAGE2_DIR.rglob("pipeline_eval*.py"))

    # remove duplicates while preserving order
    seen = set()
    out = []
    for p in candidates:
        rp = str(p.resolve())
        if rp not in seen and p.is_file():
            seen.add(rp)
            out.append(p)
    return out

def _pick_eval_function_from_module(m):
    """
    Prefer uncalibrated pipeline eval functions if available.
    """
    preferred_names = [
        "eval_loader_pipeline_uncal",
        "eval_pipeline_uncal",
        "eval_loader_pipeline_uncalibrated",
        "eval_pipeline_uncalibrated",
        "eval_loader_pipeline",
        "eval_pipeline",
    ]
    for name in preferred_names:
        if hasattr(m, name) and callable(getattr(m, name)):
            return getattr(m, name), name

    # last resort: pick ANY callable that contains both 'eval' and 'pipeline'
    for name in dir(m):
        if "eval" in name.lower() and "pipeline" in name.lower():
            fn = getattr(m, name)
            if callable(fn):
                return fn, name

    return None, None

def _resolve_pipeline_eval():
    """
    Try normal imports first (in case you DO have a stable module name),
    else auto-find and load the file.
    """
    # 1) standard guesses
    for mod_path in [
        "idabd_stage2.train.pipeline_eval_uncal",
        "idabd_stage2.train.pipeline_eval",
        "idabd_stage2.train.pipeline_eval_gated",
        "idabd_stage2.train.pipeline_eval_damage_f1_gated",
        "proposed_approach.train.pipeline_eval_uncal",
        "proposed_approach.train.pipeline_eval",
    ]:
        try:
            m = __import__(mod_path, fromlist=["*"])
            fn, fn_name = _pick_eval_function_from_module(m)
            if fn is not None:
                print(f"[PIPELINE EVAL] Using {mod_path}.{fn_name}()")
                return fn
        except Exception:
            pass

    # 2) filesystem search
    files = _find_pipeline_eval_py()
    if not files:
        raise ModuleNotFoundError(
            "Could not find any pipeline eval module.\n"
            f"Expected something like idabd_stage2/train/pipeline_eval*.py\n"
            f"Searched under: {IDABD_STAGE2_DIR}"
        )

    load_errors = []
    for i, f in enumerate(files):
        try:
            m = _load_module_from_file(f"_pipeline_eval_autoload_{i}", f)
            fn, fn_name = _pick_eval_function_from_module(m)
            if fn is not None:
                print(f"[PIPELINE EVAL] Auto-loaded {f} -> {fn_name}()")
                return fn
            else:
                load_errors.append(f"{f}: loaded, but no eval pipeline function found")
        except Exception as e:
            load_errors.append(f"{f}: {repr(e)}")

    msg = "Found pipeline_eval*.py files but none exposed a usable eval function.\n"
    msg += "\n".join(load_errors[:15])
    raise ModuleNotFoundError(msg)

def _call_pipeline_eval(eval_fn, model, test_loader, device, loc_model, loc_a, loc_b, loc_thresh, amp):
    """
    Call eval_fn in a signature-robust way (different repos use different arg names).
    Returns a normalized tuple:
        (acc_u, macro_u, f1_u[4], locP_u, locR_u, locF_u)
    """
    sig = None
    try:
        sig = inspect.signature(eval_fn)
    except Exception:
        sig = None

    mapping = {
        # models
        "model": model, "damage_model": model, "net": model, "stage2_model": model,
        # loader
        "loader": test_loader, "test_loader": test_loader, "dl": test_loader, "data_loader": test_loader,
        # device
        "device": device,
        # loc + platt
        "loc_model": loc_model, "loc_net": loc_model, "localizer": loc_model, "stage1_model": loc_model,
        "loc_a": loc_a, "a": loc_a,
        "loc_b": loc_b, "b": loc_b,
        "loc_thresh": loc_thresh, "thresh": loc_thresh,
        # amp
        "amp": amp, "use_amp": amp,
    }

    # Build kwargs based on parameter names if possible
    kwargs = {}
    args = []
    if sig is not None:
        for p in sig.parameters.values():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if p.name in mapping:
                kwargs[p.name] = mapping[p.name]
            else:
                # If required param with no default, we cannot satisfy by name
                if p.default is p.empty:
                    # We'll fall back to positional attempt
                    kwargs = None
                    break

    # Try: named kwargs
    if kwargs is not None:
        try:
            out = eval_fn(**kwargs)
            return _normalize_pipeline_eval_output(out)
        except TypeError:
            pass

    # Try: standard positional layout (most common)
    try:
        out = eval_fn(
            model, test_loader, device,
            loc_model=loc_model, loc_a=loc_a, loc_b=loc_b,
            loc_thresh=loc_thresh, amp=amp
        )
        return _normalize_pipeline_eval_output(out)
    except TypeError:
        pass

    # Try: fully positional
    try:
        out = eval_fn(model, test_loader, device, loc_model, loc_a, loc_b, loc_thresh, amp)
        return _normalize_pipeline_eval_output(out)
    except Exception as e:
        raise RuntimeError(f"Could not call pipeline eval function {eval_fn.__name__}: {repr(e)}")

def _normalize_pipeline_eval_output(out):
    """
    Accept either:
      - tuple/list: (acc, macro, f1_vec, locP, locR, locF)
      - dict with common keys
    """
    if isinstance(out, (tuple, list)) and len(out) >= 6:
        acc_u, macro_u, f1_u, locP_u, locR_u, locF_u = out[:6]
        f1_u = np.array(f1_u, dtype=float).reshape(-1)
        if f1_u.size >= 4:
            f1_u = f1_u[:4]
        else:
            # pad
            f1_u = np.pad(f1_u, (0, 4 - f1_u.size), constant_values=0.0)
        return float(acc_u), float(macro_u), f1_u, float(locP_u), float(locR_u), float(locF_u)

    if isinstance(out, dict):
        # common variants
        acc_u = out.get("pipeline_acc", out.get("acc", out.get("accuracy", 0.0)))
        macro_u = out.get("macro_f1_damage", out.get("macroF1_damage", out.get("macro_f1", 0.0)))
        f1_u = out.get("f1_damage", out.get("f1_by_class", out.get("f1", [0, 0, 0, 0])))
        locP_u = out.get("loc_precision", out.get("precision_loc", 0.0))
        locR_u = out.get("loc_recall", out.get("recall_loc", 0.0))
        locF_u = out.get("loc_f1", out.get("f1_loc", 0.0))
        f1_u = np.array(f1_u, dtype=float).reshape(-1)
        if f1_u.size >= 4:
            f1_u = f1_u[:4]
        else:
            f1_u = np.pad(f1_u, (0, 4 - f1_u.size), constant_values=0.0)
        return float(acc_u), float(macro_u), f1_u, float(locP_u), float(locR_u), float(locF_u)

    raise RuntimeError(f"Unexpected pipeline eval output type/shape: {type(out)}")


# =============================================================================
# Imports (your real structure) — DO NOT import pipeline eval here (we resolve dynamically)
# =============================================================================
from idabd_stage2.aug.fusion_ecus import FusionECUSConfig
from idabd_stage2.models.xview2_builders import (
    build_damage_model_from_weight,
    build_loc_model_from_weight
)
from idabd_stage2.train.dataset_fusion_ecus import (
    build_stage2_triplets,
    IdaBDStage2DamageFusionECUS
)
from idabd_stage2.utils.auto_idabd_stage2 import (
    auto_defaults_for_dataset,
    auto_defaults_for_weights_dir,
    auto_find_stage1_loc_weight,
    auto_find_stage1_platt,
    auto_find_stage2_init_weights,
)
from idabd_stage2.utils.csv_simple import ensure_csv, append_csv_row


# =============================================================================
# Constants
# =============================================================================
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


# =============================================================================
# Robust helpers: unwrap/flatten auto_* returns
# =============================================================================
def _first(x):
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return x[0]
    return x

def _auto_dataset_defaults(script_file, img_fb, mask_fb):
    res = auto_defaults_for_dataset(script_file, img_fb, mask_fb)
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        return str(res[0]), str(res[1])
    return img_fb, mask_fb

def _auto_weights_dir(script_file, weights_fb):
    res = auto_defaults_for_weights_dir(script_file, weights_fb)
    res = _first(res)
    return str(res)

def _auto_loc_weight(script_file):
    res = auto_find_stage1_loc_weight(script_file)
    res = _first(res)
    return str(res) if res else ""

def _auto_platt(script_or_loc):
    res = auto_find_stage1_platt(script_or_loc)
    res = _first(res)
    return str(res) if res else ""

def _normalize_weights(ret):
    patterns = None
    if isinstance(ret, (tuple, list)) and len(ret) == 2 and isinstance(ret[0], (list, tuple)):
        weights_obj, patterns = ret[0], ret[1]
    else:
        weights_obj = ret

    flat = []
    def rec(o):
        if o is None:
            return
        if isinstance(o, (list, tuple)):
            for it in o:
                rec(it)
        else:
            flat.append(str(o))
    rec(weights_obj)

    out = []
    seen = set()
    for w in flat:
        if w and w not in seen:
            seen.add(w)
            out.append(w)

    return out, patterns


# =============================================================================
# Loss helpers
# =============================================================================
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


# =============================================================================
# Train + eval
# =============================================================================
def train_and_eval_one(args, init_weight: str, eval_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] {device}")
    print(f"[INIT ] {init_weight}")

    if not args.loc_weight or not path.exists(args.loc_weight):
        raise FileNotFoundError(f"Stage-1 loc checkpoint not found: {args.loc_weight}")

    # Stage-1 loc (frozen)
    loc_model = build_loc_model_from_weight(args.loc_weight).to(device).eval()
    for p in loc_model.parameters():
        p.requires_grad = False

    # Optional Platt params for localization
    loc_a, loc_b = 1.0, 0.0
    if args.loc_platt and path.exists(args.loc_platt):
        d = np.load(args.loc_platt)
        loc_a, loc_b = float(d["a"]), float(d["b"])
        print(f"[LOC  ] platt: {args.loc_platt} (a={loc_a:.6f}, b={loc_b:.6f})")
    else:
        if args.loc_platt:
            print(f"[LOC  ] platt: {args.loc_platt} (MISSING -> ignored)")
        else:
            print("[LOC  ] platt: (none)")
    print(f"[LOC  ] thresh: {args.loc_thresh}")

    # Triplets + split
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
    val_tr   = triplets[n_test:n_test + n_val]
    train_tr = triplets[n_test + n_val:]
    print(f"[DATA ] total={n} | train={len(train_tr)} | val={len(val_tr)} | test={len(test_tr)}")

    # Fusion config (ECUS)
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
    val_ds  = IdaBDStage2DamageFusionECUS(val_tr,  cfg=cfg, crop_size=0, is_train=False, fusion_p=1.0, seed=args.seed)
    test_ds = IdaBDStage2DamageFusionECUS(test_tr, cfg=cfg, crop_size=0, is_train=False, fusion_p=1.0, seed=args.seed)

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val_ds, batch_size=1, shuffle=False,
                          num_workers=max(0, args.workers // 2), pin_memory=True)
    test_ld  = DataLoader(test_ds, batch_size=1, shuffle=False,
                          num_workers=max(0, args.workers // 2), pin_memory=True)

    # Stage-2 model
    model = build_damage_model_from_weight(init_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL, reduction="mean")

    # scaler + autocast
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

    acc_u, macro_u, f1_u, locP_u, locR_u, locF_u = _call_pipeline_eval(
        eval_fn,
        model=model,
        test_loader=test_ld,
        device=device,
        loc_model=loc_model,
        loc_a=loc_a,
        loc_b=loc_b,
        loc_thresh=args.loc_thresh,
        amp=args.amp
    )

    print("\n==================== PIPELINE TEST (UNCALIBRATED ONLY) ====================")
    print(f"pipeline_acc(0..4)={acc_u:.6f}")
    print(f"F1 Localization={locF_u:.6f} (P={locP_u:.6f}, R={locR_u:.6f})")
    print(f"macroF1(Damage 1..4)={macro_u:.6f}")
    for i in range(4):
        print(f"F1 {CLASS_NAMES_4[i]:>9s}: {float(f1_u[i]):.6f}")
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
        "pipeline_f1_no_damage_uncal": float(f1_u[0]),
        "pipeline_f1_minor_uncal": float(f1_u[1]),
        "pipeline_f1_major_uncal": float(f1_u[2]),
        "pipeline_f1_destroyed_uncal": float(f1_u[3]),
    }
    append_csv_row(args.csv_path, CSV_FIELDS, row)
    print(f"[CSV  ] Appended -> {args.csv_path}")


def main():
    # resolve pipeline eval function NOW (after sys.path/aliases set)
    eval_fn = _resolve_pipeline_eval()

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

    img_auto, mask_auto = _auto_dataset_defaults(str(THIS_FILE), IMG_FALLBACK, MASK_FALLBACK)
    if not args.img_dir or not path.isdir(args.img_dir):
        args.img_dir = img_auto
    if not args.mask_dir or not path.isdir(args.mask_dir):
        args.mask_dir = mask_auto
    if not args.weights_dir or not path.isdir(args.weights_dir):
        args.weights_dir = _auto_weights_dir(str(THIS_FILE), WEIGHTS_FALLBACK)

    if not args.loc_weight:
        args.loc_weight = _auto_loc_weight(str(THIS_FILE))

    if not args.loc_platt and args.loc_weight:
        pl = ""
        try:
            pl = _auto_platt(args.loc_weight)
        except Exception:
            pl = ""
        if not pl:
            try:
                pl = _auto_platt(str(THIS_FILE))
            except Exception:
                pl = ""
        args.loc_platt = pl

    print("\n[AUTO CONFIG]")
    print(f"  img_dir     : {args.img_dir}")
    print(f"  mask_dir    : {args.mask_dir}")
    print(f"  weights_dir : {args.weights_dir}")
    print(f"  loc_weight  : {args.loc_weight if args.loc_weight else '(NOT FOUND)'}")
    print(f"  loc_platt   : {args.loc_platt if args.loc_platt else '(none)'}")
    print(f"  amp         : {args.amp}")

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
        train_and_eval_one(args, args.init_weight, eval_fn=eval_fn)
        return

    # AUTO stage-2 init weights
    ret = auto_find_stage2_init_weights(args.weights_dir)
    weight_files, patterns = _normalize_weights(ret)

    if not args.include_idabd_finetune:
        weight_files = [w for w in weight_files if "idabd_finetune" not in path.basename(w).lower()]

    if not weight_files:
        raise FileNotFoundError(f"No Stage-2 init weights found. Expected patterns: {patterns}")

    print("[WEIGHTS] Stage-2 init checkpoints to run:")
    for w in weight_files:
        print("  -", w)

    for w in weight_files:
        if not path.exists(w):
            raise FileNotFoundError(w)
        train_and_eval_one(args, w, eval_fn=eval_fn)


if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    main()
