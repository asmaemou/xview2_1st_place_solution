# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# ================================================================================
# EVAL: ENSEMBLE STAGE-1 LOC + ENSEMBLE STAGE-2 DAMAGE (UNCAL ONLY)
# FUSION-ONLY INPUT (NO RAW RGB):
#   x6 = [pre_fusion_3ch, post_fusion_3ch]
# NO DA:
#   - no calibration/vector scaling for stage-2
#   - stage-1 may still use per-model Platt (if present) exactly like your pipeline

# This script is updated to MATCH the train script output folder naming:
#   train_idabd_stage2_pipeline_FUSION_ONLY_noDA.py  ->
#     out_dir = idabd_stage2_damage_ft_checkpoints_FUSION_ONLY_noDA

# RUN (auto-find dirs):
#   python eval_idabd_pipeline_FUSION_ONLY_noDA.py

# RUN (explicit stage2_dir):
#   python eval_idabd_pipeline_FUSION_ONLY_noDA.py --stage2_dir idabd_stage2_damage_ft_checkpoints_FUSION_ONLY_noDA
# ================================================================================
# """

# import os
# from os import path
# import sys
# from pathlib import Path
# import glob
# import argparse
# import random
# import csv

# import numpy as np
# import cv2
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader


# # ---------------------------------------------------------------------
# # Make sure "zoo" is importable
# # ---------------------------------------------------------------------
# THIS_DIR = Path(__file__).resolve().parent
# for cand in [THIS_DIR, THIS_DIR.parent, THIS_DIR.parent.parent]:
#     if (cand / "zoo").is_dir():
#         sys.path.insert(0, str(cand))
#         break


# # -----------------------------------------------------------------------------
# # Defaults (match train script outputs)
# # -----------------------------------------------------------------------------
# IMG_DIR = "../idabd/images"
# MASK_DIR = "../idabd/masks"

# # MUST match your train script out_dir:
# DEFAULT_STAGE2_DIR = "idabd_stage2_damage_ft_checkpoints_FUSION_ONLY_noDA"

# DEFAULT_LOC_DIR_CANDIDATES = [
#     "idabd_stage1_loc_ft_checkpoints",
#     "idabd_stage1_loc_ft_checkpoints_secondplace",
#     "stage1_loc_checkpoints",
#     "loc_checkpoints",
# ]

# IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
# IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# IGNORE_LABEL = 255
# CLASS_NAMES_4 = ["No-Damage", "Minor", "Major", "Destroyed"]
# BUILD_TENSOR_CPU = torch.tensor([1, 2, 3, 4], dtype=torch.long)

# CSV_FIELDS = [
#     "seed",
#     "img_dir",
#     "mask_dir",

#     "loc_dir",
#     "loc_models_used",
#     "loc_glob",
#     "loc_thresh",

#     "stage2_dir",
#     "stage2_models_used",
#     "stage2_glob",
#     "stage2_max_models",

#     "alpha_unsharp",
#     "unsharp_sigma",
#     "unsharp_amount",
#     "alpha_contrast",
#     "clahe_clip",
#     "clahe_grid",
#     "alpha_edge",
#     "canny_t1",
#     "canny_t2",
#     "edge_dilate",

#     "pipeline_acc_uncal",
#     "pipeline_loc_precision_uncal",
#     "pipeline_loc_recall_uncal",
#     "pipeline_loc_f1_uncal",
#     "pipeline_macroF1_damage_uncal",
#     "pipeline_f1_no_damage_uncal",
#     "pipeline_f1_minor_uncal",
#     "pipeline_f1_major_uncal",
#     "pipeline_f1_destroyed_uncal",
# ]


# def ensure_csv(csv_path: str, overwrite: bool = False):
#     if not csv_path:
#         return
#     out_dir = path.dirname(csv_path)
#     if out_dir:
#         os.makedirs(out_dir, exist_ok=True)

#     if overwrite and path.exists(csv_path):
#         os.remove(csv_path)

#     if not path.exists(csv_path):
#         with open(csv_path, "w", newline="", encoding="utf-8") as f:
#             w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
#             w.writeheader()
#             f.flush()


# def append_csv_row(csv_path: str, row: dict):
#     if not csv_path:
#         return
#     with open(csv_path, "a", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
#         w.writerow(row)
#         f.flush()


# # =============================================================================
# # Triplets
# # =============================================================================
# def tile_id_from_name(fname: str) -> str:
#     base = path.splitext(path.basename(fname))[0]
#     base = base.replace("_pre_disaster", "").replace("_post_disaster", "")
#     return base


# def list_split_files(root: str, split: str):
#     exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
#     files = []
#     for e in exts:
#         files += glob.glob(path.join(root, e))
#     files = sorted(files)
#     if split == "pre":
#         files = [f for f in files if "_pre_disaster" in path.basename(f)]
#     elif split == "post":
#         files = [f for f in files if "_post_disaster" in path.basename(f)]
#     return files


# def find_mask(mask_dir: str, tile_id: str, gt_split: str):
#     for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
#         cand = path.join(mask_dir, f"{tile_id}_{gt_split}_disaster{ext}")
#         if path.exists(cand):
#             return cand
#     return None


# def build_stage2_triplets(img_dir: str, mask_dir: str, gt_split: str = "post"):
#     pre_imgs  = list_split_files(img_dir, "pre")
#     post_imgs = list_split_files(img_dir, "post")

#     pre_map  = {tile_id_from_name(p): p for p in pre_imgs}
#     post_map = {tile_id_from_name(p): p for p in post_imgs}

#     tile_ids = sorted(set(pre_map.keys()) & set(post_map.keys()))
#     triplets = []
#     for tid in tile_ids:
#         m = find_mask(mask_dir, tid, gt_split)
#         if m is None:
#             continue
#         triplets.append((pre_map[tid], post_map[tid], m))
#     return triplets


# # =============================================================================
# # Fusion-only (MUST match train script fusion)
# # =============================================================================
# def blend_bgr(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
#     alpha = float(np.clip(alpha, 0.0, 1.0))
#     if alpha <= 0:
#         return a
#     if alpha >= 1:
#         return b
#     return cv2.addWeighted(a, 1.0 - alpha, b, alpha, 0.0)


# def unsharp_bgr(img_bgr: np.ndarray, sigma: float, amount: float) -> np.ndarray:
#     sigma = float(max(0.01, sigma))
#     amount = float(max(0.0, amount))
#     blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
#     sharp = cv2.addWeighted(img_bgr, 1.0 + amount, blur, -amount, 0.0)
#     return np.clip(sharp, 0, 255).astype(np.uint8)


# def clahe_contrast_bgr(img_bgr: np.ndarray, clip_limit: float, grid: int) -> np.ndarray:
#     clip_limit = float(max(0.1, clip_limit))
#     grid = int(max(2, grid))
#     lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
#     L, A, B = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
#     L2 = clahe.apply(L)
#     return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)


# def edge_overlay_bgr(img_bgr: np.ndarray, t1: int, t2: int, dilate: int, alpha_edge: float) -> np.ndarray:
#     t1 = int(max(1, t1))
#     t2 = int(max(t1 + 1, t2))
#     dilate = int(max(0, dilate))
#     alpha_edge = float(max(0.0, alpha_edge))

#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     e = cv2.Canny(gray, t1, t2)
#     if dilate > 0:
#         k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
#         e = cv2.dilate(e, k, iterations=1)
#     e_bgr = cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)
#     out = cv2.addWeighted(img_bgr, 1.0, e_bgr, alpha_edge, 0.0)
#     return np.clip(out, 0, 255).astype(np.uint8)


# def fusion_only_bgr(
#     img_bgr: np.ndarray,
#     alpha_unsharp: float,
#     unsharp_sigma: float,
#     unsharp_amount: float,
#     alpha_contrast: float,
#     clahe_clip: float,
#     clahe_grid: int,
#     alpha_edge: float,
#     canny_t1: int,
#     canny_t2: int,
#     edge_dilate: int,
# ) -> np.ndarray:
#     """
#     Returns a 3-channel "fusion-only" image (not raw RGB, but 3-channel).
#     """
#     u = unsharp_bgr(img_bgr, sigma=unsharp_sigma, amount=unsharp_amount)
#     m = blend_bgr(img_bgr, u, alpha_unsharp)

#     c = clahe_contrast_bgr(m, clip_limit=clahe_clip, grid=clahe_grid)
#     mc = blend_bgr(m, c, alpha_contrast)

#     e = edge_overlay_bgr(mc, t1=canny_t1, t2=canny_t2, dilate=edge_dilate, alpha_edge=alpha_edge)
#     return e


# def preprocess_rgb(img_bgr: np.ndarray) -> torch.Tensor:
#     img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
#     img = (img - IMAGENET_MEAN) / IMAGENET_STD
#     img = np.transpose(img, (2, 0, 1))
#     return torch.from_numpy(img)


# def preprocess_6ch_fusion_only(pre_bgr: np.ndarray, post_bgr: np.ndarray, fusion_cfg: dict) -> torch.Tensor:
#     pre_f = fusion_only_bgr(pre_bgr, **fusion_cfg)
#     post_f = fusion_only_bgr(post_bgr, **fusion_cfg)
#     pre_t = preprocess_rgb(pre_f)
#     post_t = preprocess_rgb(post_f)
#     return torch.cat([pre_t, post_t], dim=0)


# def pad_to_factor(img, factor=32):
#     h, w = img.shape[:2]
#     new_h = int(np.ceil(h / factor) * factor)
#     new_w = int(np.ceil(w / factor) * factor)
#     pad_h = new_h - h
#     pad_w = new_w - w
#     if pad_h == 0 and pad_w == 0:
#         return img, (0, 0)
#     padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT_101)
#     return padded, (pad_h, pad_w)


# def load_mask_raw(mask_path: str) -> np.ndarray:
#     m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
#     if m is None:
#         raise FileNotFoundError(mask_path)
#     if m.ndim == 3:
#         m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
#     return m.astype(np.int64)


# class IdaBDEvalFusionOnly(Dataset):
#     def __init__(self, triplets, fusion_cfg):
#         self.triplets = triplets
#         self.fusion_cfg = fusion_cfg

#     def __len__(self):
#         return len(self.triplets)

#     def __getitem__(self, idx):
#         pre_path, post_path, mask_path = self.triplets[idx]
#         pre = cv2.imread(pre_path, cv2.IMREAD_COLOR)
#         post = cv2.imread(post_path, cv2.IMREAD_COLOR)
#         if pre is None:
#             raise FileNotFoundError(pre_path)
#         if post is None:
#             raise FileNotFoundError(post_path)

#         m = load_mask_raw(mask_path)

#         pre, pad_hw = pad_to_factor(pre, 32)
#         post, _ = pad_to_factor(post, 32)
#         if pad_hw != (0, 0):
#             m = np.pad(m, ((0, pad_hw[0]), (0, pad_hw[1])), mode="edge")

#         x = preprocess_6ch_fusion_only(pre, post, self.fusion_cfg)
#         y = torch.from_numpy(m).long()
#         return x.float(), y.long(), path.basename(post_path)


# # =============================================================================
# # Models
# # =============================================================================
# def strip_prefixes(sd: dict):
#     prefixes = ["module.", "model.", "net."]
#     out = {}
#     for k, v in sd.items():
#         kk = k
#         for p in prefixes:
#             if kk.startswith(p):
#                 kk = kk[len(p):]
#         out[kk] = v
#     return out


# def load_state_dict_any(weight_path: str):
#     try:
#         ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
#     except TypeError:
#         ckpt = torch.load(weight_path, map_location="cpu")

#     if isinstance(ckpt, dict) and "state_dict" in ckpt:
#         sd = ckpt["state_dict"]
#     elif isinstance(ckpt, dict) and "model" in ckpt:
#         sd = ckpt["model"]
#     else:
#         sd = ckpt

#     if not isinstance(sd, dict):
#         raise ValueError(f"Checkpoint not a state_dict mapping: {weight_path}")
#     return strip_prefixes(sd)


# def detect_backbone_from_filename(fname_lower: str) -> str:
#     for key in ["dpn92", "res34", "res50", "se154"]:
#         if key in fname_lower:
#             return key
#     return ""


# def build_damage_model_from_weight(weight_path: str):
#     import zoo.models as zm
#     fname = path.basename(weight_path).lower()
#     bb = detect_backbone_from_filename(fname)
#     if bb == "dpn92":
#         model = zm.Dpn92_Unet_Double(pretrained=None)
#     elif bb == "res34":
#         model = zm.Res34_Unet_Double(pretrained=False)
#     elif bb == "res50":
#         model = zm.SeResNext50_Unet_Double(pretrained=None)
#     elif bb == "se154":
#         model = zm.SeNet154_Unet_Double(pretrained=None)
#     else:
#         raise ValueError(f"Unrecognized stage-2 backbone: {fname}")
#     sd = load_state_dict_any(weight_path)
#     model.load_state_dict(sd, strict=True)
#     return model


# def build_loc_model_from_weight(weight_path: str):
#     import zoo.models as zm
#     fname = path.basename(weight_path).lower()
#     bb = detect_backbone_from_filename(fname)
#     if bb == "dpn92":
#         model = zm.Dpn92_Unet_Loc(pretrained=None)
#     elif bb == "res34":
#         model = zm.Res34_Unet_Loc(pretrained=False)
#     elif bb == "res50":
#         model = zm.SeResNext50_Unet_Loc(pretrained=None)
#     elif bb == "se154":
#         model = zm.SeNet154_Unet_Loc(pretrained=None)
#     else:
#         raise ValueError(f"Unrecognized stage-1 backbone: {fname}")
#     sd = load_state_dict_any(weight_path)
#     model.load_state_dict(sd, strict=True)
#     return model


# # =============================================================================
# # Metrics + ensembles (uncal)
# # =============================================================================
# def _confusion_add(conf, gt_flat, pr_flat, ncls):
#     idx = (gt_flat * ncls + pr_flat).astype(np.int64)
#     binc = np.bincount(idx, minlength=ncls * ncls)
#     conf += binc.reshape(ncls, ncls)


# def f1s_from_conf(conf):
#     n = conf.shape[0]
#     f1s = []
#     for c in range(n):
#         tp = conf[c, c]
#         fp = conf[:, c].sum() - tp
#         fn = conf[c, :].sum() - tp
#         f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)
#         f1s.append(float(f1))
#     return f1s


# def acc_from_conf(conf):
#     return float(np.trace(conf) / (conf.sum() + 1e-9))


# def prf_from_counts(tp, fp, fn):
#     p = tp / (tp + fp + 1e-9)
#     r = tp / (tp + fn + 1e-9)
#     f = (2 * tp) / (2 * tp + fp + fn + 1e-9)
#     return float(p), float(r), float(f)


# def make_amp_ctx(device: torch.device, use_amp: bool):
#     # Prefer torch.amp.autocast when available (PyTorch 2.x)
#     if use_amp and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
#         def _ctx():
#             return torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"))
#         return _ctx
#     else:
#         def _ctx():
#             return torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda"))
#         return _ctx


# @torch.no_grad()
# def predict_build_mask_ensemble(loc_pack, x6, thresh, amp_ctx):
#     # Stage-1 uses PRE image only (first 3 channels), exactly like the pipeline
#     pre = x6[:, 0:3, :, :]
#     prob_sum = None

#     for (m, a, b) in loc_pack:
#         with amp_ctx():
#             logit = m(pre)
#             if isinstance(logit, (tuple, list)):
#                 logit = logit[0]
#             if logit.ndim == 3:
#                 logit = logit.unsqueeze(1)
#             if logit.ndim == 4 and logit.shape[1] > 1:
#                 logit = logit[:, 0:1, :, :]
#             logit = a * logit + b
#             prob = torch.sigmoid(logit)[:, 0, :, :]

#         prob_sum = prob if prob_sum is None else (prob_sum + prob)

#     prob_mean = prob_sum / float(len(loc_pack))
#     return prob_mean >= thresh


# @torch.no_grad()
# def predict_damage_ensemble_uncal(stage2_models, x6, amp_ctx):
#     # Stage-2 uses softmax over classes 1..4, then ensemble average
#     prob_sum = None
#     for m in stage2_models:
#         with amp_ctx():
#             logits = m(x6)
#             if isinstance(logits, (tuple, list)):
#                 logits = logits[0]
#             z = logits[:, 1:5, :, :]
#             probs = F.softmax(z, dim=1)
#         prob_sum = probs if prob_sum is None else (prob_sum + probs)

#     prob_mean = prob_sum / float(len(stage2_models))
#     return torch.argmax(prob_mean, dim=1) + 1


# @torch.no_grad()
# def eval_pipeline_ensemble_uncal(loader, device, loc_pack, stage2_models, loc_thresh, use_amp: bool):
#     amp_ctx = make_amp_ctx(device, use_amp)

#     conf5 = np.zeros((5, 5), dtype=np.int64)
#     loc_tp = 0.0
#     loc_fp = 0.0
#     loc_fn = 0.0
#     build_tensor = BUILD_TENSOR_CPU.to(device)

#     for x6, raw, _ in loader:
#         x6 = x6.to(device, non_blocking=True)
#         raw = raw.to(device, non_blocking=True)

#         build_mask = predict_build_mask_ensemble(loc_pack, x6, loc_thresh, amp_ctx=amp_ctx)
#         pred_damage = predict_damage_ensemble_uncal(stage2_models, x6, amp_ctx=amp_ctx)

#         pred_final = torch.zeros_like(pred_damage)
#         pred_final[build_mask] = pred_damage[build_mask]

#         valid = (raw != IGNORE_LABEL)
#         gt_build = valid & torch.isin(raw, build_tensor)
#         pr_build = valid & build_mask

#         loc_tp += float((pr_build & gt_build).sum().item())
#         loc_fp += float((pr_build & (~gt_build)).sum().item())
#         loc_fn += float(((~pr_build) & gt_build).sum().item())

#         gt_np = raw.detach().cpu().numpy().astype(np.int64)
#         pr_np = pred_final.detach().cpu().numpy().astype(np.int64)

#         for i in range(gt_np.shape[0]):
#             g = gt_np[i]
#             p = pr_np[i]
#             v = (g != IGNORE_LABEL)
#             if v.sum() == 0:
#                 continue
#             _confusion_add(conf5, g[v], np.clip(p[v], 0, 4), 5)

#     acc5 = acc_from_conf(conf5)
#     f1s5 = f1s_from_conf(conf5)
#     f1s_damage = [f1s5[c] for c in [1, 2, 3, 4]]
#     macro_damage = float(np.mean(f1s_damage))
#     loc_p, loc_r, loc_f1 = prf_from_counts(loc_tp, loc_fp, loc_fn)
#     return acc5, loc_p, loc_r, loc_f1, macro_damage, f1s_damage


# # =============================================================================
# # Load packs
# # =============================================================================
# def strip_known_suffix(name_noext: str, suffixes):
#     for suf in suffixes:
#         if name_noext.endswith(suf):
#             return name_noext[:-len(suf)]
#     return name_noext


# def load_loc_pack(loc_dir: str, device, loc_glob="*_idabd_ft_best.pth"):
#     """
#     Loads ALL stage-1 loc models in loc_dir matching loc_glob.
#     For each model, if a matching *_idabd_platt.npz exists, it is applied (a,b).
#     """
#     weights = sorted(glob.glob(path.join(loc_dir, loc_glob)))
#     if not weights:
#         raise FileNotFoundError(f"No Stage-1 loc weights in {loc_dir} with {loc_glob}")

#     pack = []
#     for w in weights:
#         base = path.basename(w)
#         noext = path.splitext(base)[0]
#         stem = strip_known_suffix(noext, ["_idabd_ft_best"])
#         platt = path.join(loc_dir, f"{stem}_idabd_platt.npz")

#         model = build_loc_model_from_weight(w).to(device).eval()
#         for p in model.parameters():
#             p.requires_grad = False

#         a, b = 1.0, 0.0
#         if path.exists(platt):
#             d = np.load(platt)
#             a = float(d["a"])
#             b = float(d["b"])

#         pack.append((model, a, b))

#     return pack, weights


# def load_stage2_models(stage2_dir: str, device, stage2_glob="*_idabd_stage2_ft_best.pth", stage2_max_models: int = 12):
#     ckpts = sorted(glob.glob(path.join(stage2_dir, stage2_glob)))
#     if not ckpts:
#         raise FileNotFoundError(f"No Stage-2 checkpoints in {stage2_dir} with {stage2_glob}")

#     if stage2_max_models > 0 and len(ckpts) > stage2_max_models:
#         ckpts = ckpts[:stage2_max_models]

#     models = []
#     for ck in ckpts:
#         m = build_damage_model_from_weight(ck).to(device).eval()
#         for p in m.parameters():
#             p.requires_grad = False
#         models.append(m)

#     return models, ckpts


# # =============================================================================
# # Auto dirs
# # =============================================================================
# def _script_dir() -> Path:
#     return Path(__file__).resolve().parent


# def _candidate_dirs(*rel_paths):
#     roots = [_script_dir(), _script_dir().parent, _script_dir().parent.parent, Path.cwd(), Path.cwd().parent]
#     out = []
#     for r in roots:
#         for rel in rel_paths:
#             d = (r / rel).resolve()
#             if d.is_dir():
#                 out.append(str(d))
#     seen, dedup = set(), []
#     for d in out:
#         if d not in seen:
#             dedup.append(d)
#             seen.add(d)
#     return dedup


# def auto_find_dataset_dirs():
#     img_dirs = _candidate_dirs("idabd/images", "../idabd/images", "data/idabd/images", "dataset/idabd/images")
#     mask_dirs = _candidate_dirs("idabd/masks", "../idabd/masks", "data/idabd/masks", "dataset/idabd/masks")
#     return (img_dirs[0] if img_dirs else IMG_DIR), (mask_dirs[0] if mask_dirs else MASK_DIR)


# def auto_find_stage2_dir():
#     # Prefer the train output folder name, but allow older names too.
#     candidates = [
#         "idabd_stage2_damage_ft_checkpoints_FUSION_ONLY_noDA",
#         "idabd_stage2_damage_ft_checkpoints_FUSION_ONLY_edge_contrast_unsharp_NO_DA",
#         "idabd_stage2_damage_ft_checkpoints_FUSION_ONLY_edge_contrast_unsharp_noDA_earlyStop",
#     ]
#     rels = []
#     for c in candidates:
#         rels += [c, f"../{c}"]
#     dirs = _candidate_dirs(*rels)
#     return (dirs[0] if dirs else "")


# def auto_find_loc_dir():
#     rels = []
#     for c in DEFAULT_LOC_DIR_CANDIDATES:
#         rels += [c, f"../{c}", f"two_stage_pipeline/{c}", f"../two_stage_pipeline/{c}"]
#     dirs = _candidate_dirs(*rels)
#     return (dirs[0] if dirs else "")


# # =============================================================================
# # Main
# # =============================================================================
# def main():
#     ap = argparse.ArgumentParser()

#     ap.add_argument("--img_dir", default="")
#     ap.add_argument("--mask_dir", default="")
#     ap.add_argument("--gt_split", choices=["pre", "post"], default="post")

#     ap.add_argument("--test_ratio", type=float, default=0.1)
#     ap.add_argument("--seed", type=int, default=0)

#     ap.add_argument("--loc_dir", default="")
#     ap.add_argument("--loc_glob", default="*_idabd_ft_best.pth")
#     ap.add_argument("--loc_thresh", type=float, default=0.5)

#     ap.add_argument("--stage2_dir", default="")
#     ap.add_argument("--stage2_glob", default="*_idabd_stage2_ft_best.pth")
#     ap.add_argument("--stage2_max_models", type=int, default=12)

#     # fusion-only params (should match the ones used in training if you changed them)
#     ap.add_argument("--alpha_unsharp", type=float, default=0.7)
#     ap.add_argument("--unsharp_sigma", type=float, default=1.2)
#     ap.add_argument("--unsharp_amount", type=float, default=1.0)
#     ap.add_argument("--alpha_contrast", type=float, default=0.6)
#     ap.add_argument("--clahe_clip", type=float, default=2.0)
#     ap.add_argument("--clahe_grid", type=int, default=8)
#     ap.add_argument("--alpha_edge", type=float, default=0.6)
#     ap.add_argument("--canny_t1", type=int, default=50)
#     ap.add_argument("--canny_t2", type=int, default=150)
#     ap.add_argument("--edge_dilate", type=int, default=2)

#     ap.add_argument("--workers", type=int, default=2)
#     ap.add_argument("--amp", action="store_true")

#     ap.add_argument("--csv_path", default="idabd_pipeline_ENSEMBLE_LOC+DMG_UNCAL_ONLY_FUSION_ONLY_noDA.csv")
#     ap.add_argument("--overwrite_csv", action="store_true")

#     args = ap.parse_args()
#     if not (0.0 < args.test_ratio < 1.0):
#         raise ValueError("test_ratio must be in (0,1).")

#     # Auto dataset dirs
#     img_auto, mask_auto = auto_find_dataset_dirs()
#     if not args.img_dir or not path.isdir(args.img_dir):
#         args.img_dir = img_auto
#     if not args.mask_dir or not path.isdir(args.mask_dir):
#         args.mask_dir = mask_auto

#     # Auto loc + stage2 dirs
#     if not args.stage2_dir:
#         args.stage2_dir = auto_find_stage2_dir()
#     if not args.loc_dir:
#         args.loc_dir = auto_find_loc_dir()

#     print("\n[AUTO CONFIG]")
#     print(f"  img_dir    : {args.img_dir}")
#     print(f"  mask_dir   : {args.mask_dir}")
#     print(f"  loc_dir    : {args.loc_dir if args.loc_dir else '(NOT FOUND)'}")
#     print(f"  stage2_dir : {args.stage2_dir if args.stage2_dir else '(NOT FOUND)'}")

#     if not path.isdir(args.img_dir):
#         raise FileNotFoundError("Could not find img_dir. Pass --img_dir explicitly.")
#     if not path.isdir(args.mask_dir):
#         raise FileNotFoundError("Could not find mask_dir. Pass --mask_dir explicitly.")
#     if not args.loc_dir or not path.isdir(args.loc_dir):
#         raise FileNotFoundError("Could not find loc_dir. Pass --loc_dir explicitly.")
#     if not args.stage2_dir or not path.isdir(args.stage2_dir):
#         raise FileNotFoundError("Could not find stage2_dir. Pass --stage2_dir explicitly.")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"[DEVICE] {device}")

#     # Seeds
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(args.seed)

#     ensure_csv(args.csv_path, overwrite=args.overwrite_csv)
#     print(f"[CSV   ] Ready: {args.csv_path}")

#     # Build triplets and select test subset the same style as train scripts
#     triplets = build_stage2_triplets(args.img_dir, args.mask_dir, gt_split=args.gt_split)
#     if not triplets:
#         raise FileNotFoundError("No (pre, post, mask) triplets found.")

#     rng = random.Random(args.seed)
#     rng.shuffle(triplets)
#     n = len(triplets)

#     n_test = max(1, int(n * args.test_ratio))
#     n_test = min(n_test, n)
#     test_tr = triplets[:n_test]
#     print(f"[DATA  ] total={n} | test={len(test_tr)}")

#     fusion_cfg = dict(
#         alpha_unsharp=args.alpha_unsharp,
#         unsharp_sigma=args.unsharp_sigma,
#         unsharp_amount=args.unsharp_amount,
#         alpha_contrast=args.alpha_contrast,
#         clahe_clip=args.clahe_clip,
#         clahe_grid=args.clahe_grid,
#         alpha_edge=args.alpha_edge,
#         canny_t1=args.canny_t1,
#         canny_t2=args.canny_t2,
#         edge_dilate=args.edge_dilate,
#     )

#     test_ds = IdaBDEvalFusionOnly(test_tr, fusion_cfg=fusion_cfg)
#     test_ld = DataLoader(
#         test_ds,
#         batch_size=1,
#         shuffle=False,
#         num_workers=max(0, args.workers),
#         pin_memory=True
#     )

#     # Load models
#     loc_pack, loc_used = load_loc_pack(args.loc_dir, device=device, loc_glob=args.loc_glob)
#     stage2_models, stage2_used = load_stage2_models(
#         args.stage2_dir,
#         device=device,
#         stage2_glob=args.stage2_glob,
#         stage2_max_models=args.stage2_max_models
#     )

#     print(f"[LOC   ] models: {len(loc_used)}")
#     for w in loc_used:
#         print("   -", w)
#     print(f"[STAGE2] models: {len(stage2_used)}")
#     for w in stage2_used:
#         print("   -", w)

#     # Eval
#     acc5, locP, locR, locF, macroD, f1sD = eval_pipeline_ensemble_uncal(
#         test_ld,
#         device,
#         loc_pack=loc_pack,
#         stage2_models=stage2_models,
#         loc_thresh=args.loc_thresh,
#         use_amp=args.amp
#     )

#     print("\n==================== ENSEMBLED PIPELINE (UNCALIBRATED ONLY) ====================")
#     print("END-TO-END: Stage-1 loc gating -> Stage-2 damage (VALID pixels only; mask!=255)")
#     print(f"pipeline_acc(0..4)={acc5:.6f}")
#     print(f"F1 Localization (Building vs Background)={locF:.6f} (P={locP:.6f}, R={locR:.6f})")
#     print(f"macroF1(Damage 1..4, end-to-end conf)={macroD:.6f}")
#     print("Per-damage F1 (end-to-end conf):")
#     for i in range(4):
#         print(f"F1 {CLASS_NAMES_4[i]:>9s}: {f1sD[i]:.6f}")
#     print("===============================================================================\n")

#     row = {
#         "seed": args.seed,
#         "img_dir": args.img_dir,
#         "mask_dir": args.mask_dir,

#         "loc_dir": args.loc_dir,
#         "loc_models_used": len(loc_used),
#         "loc_glob": args.loc_glob,
#         "loc_thresh": args.loc_thresh,

#         "stage2_dir": args.stage2_dir,
#         "stage2_models_used": len(stage2_used),
#         "stage2_glob": args.stage2_glob,
#         "stage2_max_models": args.stage2_max_models,

#         **fusion_cfg,

#         "pipeline_acc_uncal": acc5,
#         "pipeline_loc_precision_uncal": locP,
#         "pipeline_loc_recall_uncal": locR,
#         "pipeline_loc_f1_uncal": locF,
#         "pipeline_macroF1_damage_uncal": macroD,
#         "pipeline_f1_no_damage_uncal": f1sD[0],
#         "pipeline_f1_minor_uncal": f1sD[1],
#         "pipeline_f1_major_uncal": f1sD[2],
#         "pipeline_f1_destroyed_uncal": f1sD[3],
#     }
#     append_csv_row(args.csv_path, row)
#     print(f"[CSV   ] Appended -> {args.csv_path}")


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
EVAL (FIXED): Stage-1 LOC on RAW PRE RGB + Stage-2 DAMAGE on FUSION-ONLY 6ch
- Stage-2 input: x6_fusion = [pre_fusion_3ch, post_fusion_3ch]
- Stage-1 input: pre_raw_rgb (ImageNet normalized), NOT fusion
- No DA for stage-2 (no calibration/vector scaling)
- Stage-1 may still use per-model Platt (if *_idabd_platt.npz exists)

RUN:
  python eval_idabd_pipeline_FUSION_ONLY_noDA_FIXEDLOC.py --stage2_dir idabd_stage2_damage_ft_checkpoints_FUSION_ONLY_noDA

================================================================================
"""

import os
from os import path
import sys
from pathlib import Path
import glob
import argparse
import random
import csv

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------
# Make sure "zoo" is importable
# ---------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
for cand in [THIS_DIR, THIS_DIR.parent, THIS_DIR.parent.parent]:
    if (cand / "zoo").is_dir():
        sys.path.insert(0, str(cand))
        break


IMG_DIR = "../idabd/images"
MASK_DIR = "../idabd/masks"

DEFAULT_STAGE2_DIR = "idabd_stage2_damage_ft_checkpoints_FUSION_ONLY_noDA"
DEFAULT_LOC_DIR_CANDIDATES = [
    "idabd_stage1_loc_ft_checkpoints",
    "idabd_stage1_loc_ft_checkpoints_secondplace",
    "stage1_loc_checkpoints",
    "loc_checkpoints",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IGNORE_LABEL = 255
CLASS_NAMES_4 = ["No-Damage", "Minor", "Major", "Destroyed"]
BUILD_TENSOR_CPU = torch.tensor([1, 2, 3, 4], dtype=torch.long)

CSV_FIELDS = [
    "seed",
    "img_dir",
    "mask_dir",

    "loc_dir",
    "loc_models_used",
    "loc_glob",
    "loc_thresh",

    "stage2_dir",
    "stage2_models_used",
    "stage2_glob",
    "stage2_max_models",

    "alpha_unsharp",
    "unsharp_sigma",
    "unsharp_amount",
    "alpha_contrast",
    "clahe_clip",
    "clahe_grid",
    "alpha_edge",
    "canny_t1",
    "canny_t2",
    "edge_dilate",

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


def ensure_csv(csv_path: str, overwrite: bool = False):
    if not csv_path:
        return
    out_dir = path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if overwrite and path.exists(csv_path):
        os.remove(csv_path)

    if not path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            f.flush()


def append_csv_row(csv_path: str, row: dict):
    if not csv_path:
        return
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writerow(row)
        f.flush()


# =============================================================================
# Triplets
# =============================================================================
def tile_id_from_name(fname: str) -> str:
    base = path.splitext(path.basename(fname))[0]
    base = base.replace("_pre_disaster", "").replace("_post_disaster", "")
    return base


def list_split_files(root: str, split: str):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    files = []
    for e in exts:
        files += glob.glob(path.join(root, e))
    files = sorted(files)
    if split == "pre":
        files = [f for f in files if "_pre_disaster" in path.basename(f)]
    elif split == "post":
        files = [f for f in files if "_post_disaster" in path.basename(f)]
    return files


def find_mask(mask_dir: str, tile_id: str, gt_split: str):
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        cand = path.join(mask_dir, f"{tile_id}_{gt_split}_disaster{ext}")
        if path.exists(cand):
            return cand
    return None


def build_stage2_triplets(img_dir: str, mask_dir: str, gt_split: str = "post"):
    pre_imgs  = list_split_files(img_dir, "pre")
    post_imgs = list_split_files(img_dir, "post")

    pre_map  = {tile_id_from_name(p): p for p in pre_imgs}
    post_map = {tile_id_from_name(p): p for p in post_imgs}

    tile_ids = sorted(set(pre_map.keys()) & set(post_map.keys()))
    triplets = []
    for tid in tile_ids:
        m = find_mask(mask_dir, tid, gt_split)
        if m is None:
            continue
        triplets.append((pre_map[tid], post_map[tid], m))
    return triplets


# =============================================================================
# Fusion-only ops
# =============================================================================
def blend_bgr(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0:
        return a
    if alpha >= 1:
        return b
    return cv2.addWeighted(a, 1.0 - alpha, b, alpha, 0.0)


def unsharp_bgr(img_bgr: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    sigma = float(max(0.01, sigma))
    amount = float(max(0.0, amount))
    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(img_bgr, 1.0 + amount, blur, -amount, 0.0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def clahe_contrast_bgr(img_bgr: np.ndarray, clip_limit: float, grid: int) -> np.ndarray:
    clip_limit = float(max(0.1, clip_limit))
    grid = int(max(2, grid))
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    L2 = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)


def edge_overlay_bgr(img_bgr: np.ndarray, t1: int, t2: int, dilate: int, alpha_edge: float) -> np.ndarray:
    t1 = int(max(1, t1))
    t2 = int(max(t1 + 1, t2))
    dilate = int(max(0, dilate))
    alpha_edge = float(max(0.0, alpha_edge))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(gray, t1, t2)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        e = cv2.dilate(e, k, iterations=1)
    e_bgr = cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(img_bgr, 1.0, e_bgr, alpha_edge, 0.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def fusion_only_bgr(
    img_bgr: np.ndarray,
    alpha_unsharp: float,
    unsharp_sigma: float,
    unsharp_amount: float,
    alpha_contrast: float,
    clahe_clip: float,
    clahe_grid: int,
    alpha_edge: float,
    canny_t1: int,
    canny_t2: int,
    edge_dilate: int,
) -> np.ndarray:
    u = unsharp_bgr(img_bgr, sigma=unsharp_sigma, amount=unsharp_amount)
    m = blend_bgr(img_bgr, u, alpha_unsharp)
    c = clahe_contrast_bgr(m, clip_limit=clahe_clip, grid=clahe_grid)
    mc = blend_bgr(m, c, alpha_contrast)
    return edge_overlay_bgr(mc, t1=canny_t1, t2=canny_t2, dilate=edge_dilate, alpha_edge=alpha_edge)


def preprocess_rgb_bgr_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)


def preprocess_6ch_fusion_only(pre_bgr: np.ndarray, post_bgr: np.ndarray, fusion_cfg: dict) -> torch.Tensor:
    pre_f = fusion_only_bgr(pre_bgr, **fusion_cfg)
    post_f = fusion_only_bgr(post_bgr, **fusion_cfg)
    return torch.cat([preprocess_rgb_bgr_to_tensor(pre_f), preprocess_rgb_bgr_to_tensor(post_f)], dim=0)


def pad_to_factor(img, factor=32):
    h, w = img.shape[:2]
    new_h = int(np.ceil(h / factor) * factor)
    new_w = int(np.ceil(w / factor) * factor)
    pad_h = new_h - h
    pad_w = new_w - w
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT_101)
    return padded, (pad_h, pad_w)


def load_mask_raw(mask_path: str) -> np.ndarray:
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(mask_path)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return m.astype(np.int64)


class IdaBDEvalFusionOnly(Dataset):
    """
    Returns BOTH:
      - x6_fusion: fusion-only 6ch for stage-2
      - pre_raw3 : raw pre RGB (ImageNet norm) for stage-1
    """
    def __init__(self, triplets, fusion_cfg):
        self.triplets = triplets
        self.fusion_cfg = fusion_cfg

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        pre_path, post_path, mask_path = self.triplets[idx]
        pre = cv2.imread(pre_path, cv2.IMREAD_COLOR)
        post = cv2.imread(post_path, cv2.IMREAD_COLOR)
        if pre is None:
            raise FileNotFoundError(pre_path)
        if post is None:
            raise FileNotFoundError(post_path)

        m = load_mask_raw(mask_path)

        pre, pad_hw = pad_to_factor(pre, 32)
        post, _ = pad_to_factor(post, 32)
        if pad_hw != (0, 0):
            m = np.pad(m, ((0, pad_hw[0]), (0, pad_hw[1])), mode="edge")

        # Stage-1 input MUST be raw pre RGB
        pre_raw3 = preprocess_rgb_bgr_to_tensor(pre)

        # Stage-2 input is fusion-only 6ch
        x6_fusion = preprocess_6ch_fusion_only(pre, post, self.fusion_cfg)

        y = torch.from_numpy(m).long()
        return x6_fusion.float(), pre_raw3.float(), y.long(), path.basename(post_path)


# =============================================================================
# Models
# =============================================================================
def strip_prefixes(sd: dict):
    prefixes = ["module.", "model.", "net."]
    out = {}
    for k, v in sd.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p):
                kk = kk[len(p):]
        out[kk] = v
    return out


def load_state_dict_any(weight_path: str):
    try:
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(weight_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    if not isinstance(sd, dict):
        raise ValueError(f"Checkpoint not a state_dict mapping: {weight_path}")
    return strip_prefixes(sd)


def detect_backbone_from_filename(fname_lower: str) -> str:
    for key in ["dpn92", "res34", "res50", "se154"]:
        if key in fname_lower:
            return key
    return ""


def build_damage_model_from_weight(weight_path: str):
    import zoo.models as zm
    fname = path.basename(weight_path).lower()
    bb = detect_backbone_from_filename(fname)
    if bb == "dpn92":
        model = zm.Dpn92_Unet_Double(pretrained=None)
    elif bb == "res34":
        model = zm.Res34_Unet_Double(pretrained=False)
    elif bb == "res50":
        model = zm.SeResNext50_Unet_Double(pretrained=None)
    elif bb == "se154":
        model = zm.SeNet154_Unet_Double(pretrained=None)
    else:
        raise ValueError(f"Unrecognized stage-2 backbone: {fname}")
    sd = load_state_dict_any(weight_path)
    model.load_state_dict(sd, strict=True)
    return model


def build_loc_model_from_weight(weight_path: str):
    import zoo.models as zm
    fname = path.basename(weight_path).lower()
    bb = detect_backbone_from_filename(fname)
    if bb == "dpn92":
        model = zm.Dpn92_Unet_Loc(pretrained=None)
    elif bb == "res34":
        model = zm.Res34_Unet_Loc(pretrained=False)
    elif bb == "res50":
        model = zm.SeResNext50_Unet_Loc(pretrained=None)
    elif bb == "se154":
        model = zm.SeNet154_Unet_Loc(pretrained=None)
    else:
        raise ValueError(f"Unrecognized stage-1 backbone: {fname}")
    sd = load_state_dict_any(weight_path)
    model.load_state_dict(sd, strict=True)
    return model


# =============================================================================
# Metrics + ensembles (uncal)
# =============================================================================
def _confusion_add(conf, gt_flat, pr_flat, ncls):
    idx = (gt_flat * ncls + pr_flat).astype(np.int64)
    binc = np.bincount(idx, minlength=ncls * ncls)
    conf += binc.reshape(ncls, ncls)


def f1s_from_conf(conf):
    n = conf.shape[0]
    f1s = []
    for c in range(n):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)
        f1s.append(float(f1))
    return f1s


def acc_from_conf(conf):
    return float(np.trace(conf) / (conf.sum() + 1e-9))


def prf_from_counts(tp, fp, fn):
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f = (2 * tp) / (2 * tp + fp + fn + 1e-9)
    return float(p), float(r), float(f)


def make_amp_ctx(device: torch.device, use_amp: bool):
    if use_amp and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        def _ctx():
            return torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"))
        return _ctx
    else:
        def _ctx():
            return torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda"))
        return _ctx


@torch.no_grad()
def predict_build_mask_ensemble(loc_pack, pre_raw3, thresh, amp_ctx):
    prob_sum = None
    for (m, a, b) in loc_pack:
        with amp_ctx():
            logit = m(pre_raw3)
            if isinstance(logit, (tuple, list)):
                logit = logit[0]
            if logit.ndim == 3:
                logit = logit.unsqueeze(1)
            if logit.ndim == 4 and logit.shape[1] > 1:
                logit = logit[:, 0:1, :, :]
            logit = a * logit + b
            prob = torch.sigmoid(logit)[:, 0, :, :]
        prob_sum = prob if prob_sum is None else (prob_sum + prob)

    prob_mean = prob_sum / float(len(loc_pack))
    return prob_mean >= thresh


@torch.no_grad()
def predict_damage_ensemble_uncal(stage2_models, x6_fusion, amp_ctx):
    prob_sum = None
    for m in stage2_models:
        with amp_ctx():
            logits = m(x6_fusion)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            z = logits[:, 1:5, :, :]
            probs = F.softmax(z, dim=1)
        prob_sum = probs if prob_sum is None else (prob_sum + probs)
    prob_mean = prob_sum / float(len(stage2_models))
    return torch.argmax(prob_mean, dim=1) + 1


@torch.no_grad()
def eval_pipeline_ensemble_uncal(loader, device, loc_pack, stage2_models, loc_thresh, use_amp: bool):
    amp_ctx = make_amp_ctx(device, use_amp)

    conf5 = np.zeros((5, 5), dtype=np.int64)
    loc_tp = 0.0
    loc_fp = 0.0
    loc_fn = 0.0
    build_tensor = BUILD_TENSOR_CPU.to(device)

    for x6_fusion, pre_raw3, raw, _ in loader:
        x6_fusion = x6_fusion.to(device, non_blocking=True)
        pre_raw3 = pre_raw3.to(device, non_blocking=True)
        raw = raw.to(device, non_blocking=True)

        build_mask = predict_build_mask_ensemble(loc_pack, pre_raw3, loc_thresh, amp_ctx=amp_ctx)
        pred_damage = predict_damage_ensemble_uncal(stage2_models, x6_fusion, amp_ctx=amp_ctx)

        pred_final = torch.zeros_like(pred_damage)
        pred_final[build_mask] = pred_damage[build_mask]

        valid = (raw != IGNORE_LABEL)
        gt_build = valid & torch.isin(raw, build_tensor)
        pr_build = valid & build_mask

        loc_tp += float((pr_build & gt_build).sum().item())
        loc_fp += float((pr_build & (~gt_build)).sum().item())
        loc_fn += float(((~pr_build) & gt_build).sum().item())

        gt_np = raw.detach().cpu().numpy().astype(np.int64)
        pr_np = pred_final.detach().cpu().numpy().astype(np.int64)

        for i in range(gt_np.shape[0]):
            g = gt_np[i]
            p = pr_np[i]
            v = (g != IGNORE_LABEL)
            if v.sum() == 0:
                continue
            _confusion_add(conf5, g[v], np.clip(p[v], 0, 4), 5)

    acc5 = acc_from_conf(conf5)
    f1s5 = f1s_from_conf(conf5)
    f1s_damage = [f1s5[c] for c in [1, 2, 3, 4]]
    macro_damage = float(np.mean(f1s_damage))
    loc_p, loc_r, loc_f1 = prf_from_counts(loc_tp, loc_fp, loc_fn)
    return acc5, loc_p, loc_r, loc_f1, macro_damage, f1s_damage


# =============================================================================
# Load packs
# =============================================================================
def strip_known_suffix(name_noext: str, suffixes):
    for suf in suffixes:
        if name_noext.endswith(suf):
            return name_noext[:-len(suf)]
    return name_noext


def load_loc_pack(loc_dir: str, device, loc_glob="*_idabd_ft_best.pth"):
    weights = sorted(glob.glob(path.join(loc_dir, loc_glob)))
    if not weights:
        raise FileNotFoundError(f"No Stage-1 loc weights in {loc_dir} with {loc_glob}")

    pack = []
    for w in weights:
        base = path.basename(w)
        noext = path.splitext(base)[0]
        stem = strip_known_suffix(noext, ["_idabd_ft_best"])
        platt = path.join(loc_dir, f"{stem}_idabd_platt.npz")

        model = build_loc_model_from_weight(w).to(device).eval()
        for p in model.parameters():
            p.requires_grad = False

        a, b = 1.0, 0.0
        if path.exists(platt):
            d = np.load(platt)
            a = float(d["a"])
            b = float(d["b"])

        pack.append((model, a, b))

    return pack, weights


def load_stage2_models(stage2_dir: str, device, stage2_glob="*_idabd_stage2_ft_best.pth", stage2_max_models: int = 12):
    ckpts = sorted(glob.glob(path.join(stage2_dir, stage2_glob)))
    if not ckpts:
        raise FileNotFoundError(f"No Stage-2 checkpoints in {stage2_dir} with {stage2_glob}")

    if stage2_max_models > 0 and len(ckpts) > stage2_max_models:
        ckpts = ckpts[:stage2_max_models]

    models = []
    for ck in ckpts:
        m = build_damage_model_from_weight(ck).to(device).eval()
        for p in m.parameters():
            p.requires_grad = False
        models.append(m)

    return models, ckpts


# =============================================================================
# Auto dirs
# =============================================================================
def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _candidate_dirs(*rel_paths):
    roots = [_script_dir(), _script_dir().parent, _script_dir().parent.parent, Path.cwd(), Path.cwd().parent]
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


def auto_find_dataset_dirs():
    img_dirs = _candidate_dirs("idabd/images", "../idabd/images", "data/idabd/images", "dataset/idabd/images")
    mask_dirs = _candidate_dirs("idabd/masks", "../idabd/masks", "data/idabd/masks", "dataset/idabd/masks")
    return (img_dirs[0] if img_dirs else IMG_DIR), (mask_dirs[0] if mask_dirs else MASK_DIR)


def auto_find_stage2_dir():
    rels = [DEFAULT_STAGE2_DIR, f"../{DEFAULT_STAGE2_DIR}"]
    dirs = _candidate_dirs(*rels)
    return (dirs[0] if dirs else "")


def auto_find_loc_dir():
    rels = []
    for c in DEFAULT_LOC_DIR_CANDIDATES:
        rels += [c, f"../{c}", f"two_stage_pipeline/{c}", f"../two_stage_pipeline/{c}"]
    dirs = _candidate_dirs(*rels)
    return (dirs[0] if dirs else "")


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--img_dir", default="")
    ap.add_argument("--mask_dir", default="")
    ap.add_argument("--gt_split", choices=["pre", "post"], default="post")

    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--loc_dir", default="")
    ap.add_argument("--loc_glob", default="*_idabd_ft_best.pth")
    ap.add_argument("--loc_thresh", type=float, default=0.5)

    ap.add_argument("--stage2_dir", default="")
    ap.add_argument("--stage2_glob", default="*_idabd_stage2_ft_best.pth")
    ap.add_argument("--stage2_max_models", type=int, default=12)

    # fusion-only params
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

    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--csv_path", default="idabd_eval_pipeline_FUSION_ONLY_noDA_FIXEDLOC.csv")
    ap.add_argument("--overwrite_csv", action="store_true")

    args = ap.parse_args()
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    img_auto, mask_auto = auto_find_dataset_dirs()
    if not args.img_dir or not path.isdir(args.img_dir):
        args.img_dir = img_auto
    if not args.mask_dir or not path.isdir(args.mask_dir):
        args.mask_dir = mask_auto

    if not args.stage2_dir:
        args.stage2_dir = auto_find_stage2_dir()
    if not args.loc_dir:
        args.loc_dir = auto_find_loc_dir()

    print("\n[AUTO CONFIG]")
    print(f"  img_dir    : {args.img_dir}")
    print(f"  mask_dir   : {args.mask_dir}")
    print(f"  loc_dir    : {args.loc_dir if args.loc_dir else '(NOT FOUND)'}")
    print(f"  stage2_dir : {args.stage2_dir if args.stage2_dir else '(NOT FOUND)'}")

    if not path.isdir(args.img_dir):
        raise FileNotFoundError("Could not find img_dir. Pass --img_dir explicitly.")
    if not path.isdir(args.mask_dir):
        raise FileNotFoundError("Could not find mask_dir. Pass --mask_dir explicitly.")
    if not args.loc_dir or not path.isdir(args.loc_dir):
        raise FileNotFoundError("Could not find loc_dir. Pass --loc_dir explicitly.")
    if not args.stage2_dir or not path.isdir(args.stage2_dir):
        raise FileNotFoundError("Could not find stage2_dir. Pass --stage2_dir explicitly.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ensure_csv(args.csv_path, overwrite=args.overwrite_csv)
    print(f"[CSV   ] Ready: {args.csv_path}")

    triplets = build_stage2_triplets(args.img_dir, args.mask_dir, gt_split=args.gt_split)
    if not triplets:
        raise FileNotFoundError("No (pre, post, mask) triplets found.")

    # -------------------------
    # Split (match training style)
    # Training printed: total=87 | train=71 | val=8 | test=8
    # Common split is:
    #   test = first n_test
    #   val  = next n_val
    #   train = rest
    #
    # BUT many scripts do:
    #   train = first
    #   val = next
    #   test = last
    #
    # Your training script likely does train/val/test, not "test only".
    # Below I use: train = first, val = next, test = last (MOST COMMON).
    # If your train script does test first, swap to test_tr = triplets[:n_test].
    # -------------------------
    rng = random.Random(args.seed)
    rng.shuffle(triplets)
    n = len(triplets)

    n_test = max(1, int(n * args.test_ratio))
    n_val  = max(1, int(n * args.val_ratio))
    if n_test + n_val >= n:
        n_test = max(1, min(n_test, n - 2))
        n_val  = max(1, min(n_val,  n - 1 - n_test))

    train_tr = triplets[: n - (n_val + n_test)]
    val_tr   = triplets[n - (n_val + n_test) : n - n_test]
    test_tr  = triplets[n - n_test :]

    print(f"[DATA  ] total={n} | train={len(train_tr)} | val={len(val_tr)} | test={len(test_tr)}")

    fusion_cfg = dict(
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

    test_ds = IdaBDEvalFusionOnly(test_tr, fusion_cfg=fusion_cfg)
    test_ld = DataLoader(test_ds, batch_size=1, shuffle=False,
                         num_workers=max(0, args.workers), pin_memory=True)

    loc_pack, loc_used = load_loc_pack(args.loc_dir, device=device, loc_glob=args.loc_glob)
    print(f"[LOC   ] models: {len(loc_used)}")

    stage2_models, stage2_used = load_stage2_models(
        args.stage2_dir, device=device,
        stage2_glob=args.stage2_glob,
        stage2_max_models=args.stage2_max_models
    )
    print(f"[STAGE2] models: {len(stage2_used)}")

    acc5, locP, locR, locF, macroD, f1sD = eval_pipeline_ensemble_uncal(
        test_ld, device,
        loc_pack=loc_pack,
        stage2_models=stage2_models,
        loc_thresh=args.loc_thresh,
        use_amp=args.amp
    )

    print("\n==================== ENSEMBLED PIPELINE (UNCALIBRATED ONLY) ====================")
    print("END-TO-END: Stage-1 loc gating (RAW pre RGB) -> Stage-2 damage (FUSION 6ch)")
    print(f"pipeline_acc(0..4)={acc5:.6f}")
    print(f"F1 Localization (Building vs Background)={locF:.6f} (P={locP:.6f}, R={locR:.6f})")
    print(f"macroF1(Damage 1..4)={macroD:.6f}")
    for i in range(4):
        print(f"F1 {CLASS_NAMES_4[i]:>9s}: {f1sD[i]:.6f}")
    print("===============================================================================\n")

    row = {
        "seed": args.seed,
        "img_dir": args.img_dir,
        "mask_dir": args.mask_dir,

        "loc_dir": args.loc_dir,
        "loc_models_used": len(loc_used),
        "loc_glob": args.loc_glob,
        "loc_thresh": args.loc_thresh,

        "stage2_dir": args.stage2_dir,
        "stage2_models_used": len(stage2_used),
        "stage2_glob": args.stage2_glob,
        "stage2_max_models": args.stage2_max_models,

        **fusion_cfg,

        "pipeline_acc_uncal": acc5,
        "pipeline_loc_precision_uncal": locP,
        "pipeline_loc_recall_uncal": locR,
        "pipeline_loc_f1_uncal": locF,
        "pipeline_macroF1_damage_uncal": macroD,
        "pipeline_f1_no_damage_uncal": f1sD[0],
        "pipeline_f1_minor_uncal": f1sD[1],
        "pipeline_f1_major_uncal": f1sD[2],
        "pipeline_f1_destroyed_uncal": f1sD[3],
    }
    append_csv_row(args.csv_path, row)
    print(f"[CSV   ] Appended -> {args.csv_path}")


if __name__ == "__main__":
    main()
