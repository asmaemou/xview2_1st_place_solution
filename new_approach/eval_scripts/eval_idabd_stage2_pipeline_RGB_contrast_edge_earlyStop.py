# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# ================================================================================
# EVAL: ENSEMBLE STAGE-1 LOCALIZATION + ENSEMBLE STAGE-2 DAMAGE (CALIBRATED ONLY)
# IDABD end-to-end pipeline — RGB + CONTRAST + EDGE ONLY

# RUNS WITH JUST:
#   python eval_idabd_stage2_pipeline_RGB_contrast_edge_only_earlyStop.py

# Input definition:
# - x6 = [pre_RGB, post_RGB_used]
# - post_RGB_used = post fused with:
#     (A) Contrast enhancement (CLAHE on L channel)
#     (B) Edge overlay (Canny edges)

# Ensembling:
# - Stage-1: average building probabilities across loc models
# - Stage-2: apply vector scaling per model, softmax, then average probabilities

# Split MUST match training (seed/val_ratio/test_ratio).
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


# IMG_DIR = "../idabd/images"
# MASK_DIR = "../idabd/masks"

# DEFAULT_STAGE2_DIR = "idabd_stage2_damage_ft_checkpoints_RGB_CONTRAST_EDGE_ONLY"

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
#     "loc_dir",
#     "loc_models_used",
#     "stage2_dir",
#     "stage2_models_used",
#     "loc_glob",
#     "stage2_glob",
#     "loc_thresh",

#     "alpha_contrast",
#     "alpha_edge",
#     "clahe_clip",
#     "clahe_grid",
#     "canny_t1",
#     "canny_t2",
#     "edge_dilate",

#     "pipeline_acc_cal",
#     "pipeline_loc_precision_cal",
#     "pipeline_loc_recall_cal",
#     "pipeline_loc_f1_cal",
#     "pipeline_macroF1_damage_cal",
#     "pipeline_f1_no_damage_cal",
#     "pipeline_f1_minor_cal",
#     "pipeline_f1_major_cal",
#     "pipeline_f1_destroyed_cal",
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
# # Pairing
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
# # Fusion (contrast+edge) + preprocess
# # =============================================================================
# def clahe_contrast_bgr(img_bgr: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
#     lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(grid), int(grid)))
#     l2 = clahe.apply(l)
#     lab2 = cv2.merge([l2, a, b])
#     out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
#     return out


# def canny_edges(img_bgr: np.ndarray, t1: int = 50, t2: int = 150, dilate: int = 1) -> np.ndarray:
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     e = cv2.Canny(gray, int(t1), int(t2))
#     if dilate and int(dilate) > 0:
#         k = int(dilate) * 2 + 1
#         ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
#         e = cv2.dilate(e, ker, iterations=1)
#     return e


# def blend_bgr(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
#     alpha = float(np.clip(alpha, 0.0, 1.0))
#     if alpha <= 0:
#         return a
#     if alpha >= 1:
#         return b
#     return cv2.addWeighted(a, 1.0 - alpha, b, alpha, 0.0)


# def overlay_edges_on_bgr(img_bgr: np.ndarray, edges_u8: np.ndarray, alpha_edge: float) -> np.ndarray:
#     alpha = float(np.clip(alpha_edge, 0.0, 1.0))
#     if alpha <= 0:
#         return img_bgr
#     edge_rgb = cv2.cvtColor(edges_u8, cv2.COLOR_GRAY2BGR)
#     return cv2.addWeighted(img_bgr, 1.0 - alpha, edge_rgb, alpha, 0.0)


# def fuse_contrast_edge_post(
#     post_bgr: np.ndarray,
#     alpha_contrast: float,
#     alpha_edge: float,
#     clahe_clip: float,
#     clahe_grid: int,
#     canny_t1: int,
#     canny_t2: int,
#     edge_dilate: int,
# ) -> np.ndarray:
#     post_c = clahe_contrast_bgr(post_bgr, clip_limit=clahe_clip, grid=clahe_grid)
#     post_m = blend_bgr(post_bgr, post_c, alpha_contrast)
#     e = canny_edges(post_m, t1=canny_t1, t2=canny_t2, dilate=edge_dilate)
#     return overlay_edges_on_bgr(post_m, e, alpha_edge=alpha_edge)


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


# def preprocess_rgb(img_bgr: np.ndarray) -> torch.Tensor:
#     img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
#     img = (img - IMAGENET_MEAN) / IMAGENET_STD
#     img = np.transpose(img, (2, 0, 1))
#     return torch.from_numpy(img)


# def preprocess_6ch(pre_bgr: np.ndarray, post_bgr: np.ndarray) -> torch.Tensor:
#     return torch.cat([preprocess_rgb(pre_bgr), preprocess_rgb(post_bgr)], dim=0)


# def load_mask_raw(mask_path: str) -> np.ndarray:
#     m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
#     if m is None:
#         raise FileNotFoundError(mask_path)
#     if m.ndim == 3:
#         m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
#     return m.astype(np.int64)


# class IdaBDEval(Dataset):
#     def __init__(
#         self,
#         triplets,
#         alpha_contrast=0.65,
#         alpha_edge=0.35,
#         clahe_clip=2.0,
#         clahe_grid=8,
#         canny_t1=50,
#         canny_t2=150,
#         edge_dilate=1,
#     ):
#         self.triplets = triplets
#         self.alpha_contrast = float(alpha_contrast)
#         self.alpha_edge = float(alpha_edge)
#         self.clahe_clip = float(clahe_clip)
#         self.clahe_grid = int(clahe_grid)
#         self.canny_t1 = int(canny_t1)
#         self.canny_t2 = int(canny_t2)
#         self.edge_dilate = int(edge_dilate)

#     def __len__(self):
#         return len(self.triplets)

#     def __getitem__(self, idx):
#         pre_path, post_path, mask_path = self.triplets[idx]

#         pre  = cv2.imread(pre_path, cv2.IMREAD_COLOR)
#         post = cv2.imread(post_path, cv2.IMREAD_COLOR)
#         if pre is None:
#             raise FileNotFoundError(pre_path)
#         if post is None:
#             raise FileNotFoundError(post_path)

#         m = load_mask_raw(mask_path)

#         # deterministic fusion
#         post = fuse_contrast_edge_post(
#             post,
#             alpha_contrast=self.alpha_contrast,
#             alpha_edge=self.alpha_edge,
#             clahe_clip=self.clahe_clip,
#             clahe_grid=self.clahe_grid,
#             canny_t1=self.canny_t1,
#             canny_t2=self.canny_t2,
#             edge_dilate=self.edge_dilate,
#         )

#         pre, pad_hw = pad_to_factor(pre, 32)
#         post, _ = pad_to_factor(post, 32)
#         if pad_hw != (0, 0):
#             m = np.pad(m, ((0, pad_hw[0]), (0, pad_hw[1])), mode="edge")

#         x = preprocess_6ch(pre, post)
#         y = torch.from_numpy(m).long()
#         return x.float(), y.long(), path.basename(post_path)


# # =============================================================================
# # Load models
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
#         raise ValueError(f"Unrecognized stage-2 backbone in filename: {fname}")

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
#         raise ValueError(f"Unrecognized stage-1 backbone in filename: {fname}")

#     sd = load_state_dict_any(weight_path)
#     model.load_state_dict(sd, strict=True)
#     return model


# # =============================================================================
# # Metrics + AMP helpers
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
# def predict_damage_ensemble_cal(stage2_pack, x6, amp_ctx):
#     prob_sum = None

#     for (m, W, b) in stage2_pack:
#         with amp_ctx():
#             logits = m(x6)
#             if isinstance(logits, (tuple, list)):
#                 logits = logits[0]
#             z = logits[:, 1:5, :, :]  # Bx4xHxW

#         with torch.cuda.amp.autocast(enabled=False):
#             z = z.float()
#             Wf = W.float()
#             bf = b.float()
#             z2 = z.permute(0, 2, 3, 1).contiguous()
#             z2 = torch.matmul(z2, Wf.t()) + bf
#             z2 = z2.permute(0, 3, 1, 2).contiguous()
#             probs = F.softmax(z2, dim=1)

#         prob_sum = probs if prob_sum is None else (prob_sum + probs)

#     prob_mean = prob_sum / float(len(stage2_pack))
#     return torch.argmax(prob_mean, dim=1) + 1  # 1..4


# @torch.no_grad()
# def eval_pipeline_ensemble_cal(loader, device, loc_pack, stage2_pack, loc_thresh, use_amp: bool, debug_destroyed: bool):
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
#         pred_damage = predict_damage_ensemble_cal(stage2_pack, x6, amp_ctx=amp_ctx)

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
#             g_valid = g[v]
#             p_valid = np.clip(p[v], 0, 4)
#             _confusion_add(conf5, g_valid, p_valid, 5)

#     acc5 = acc_from_conf(conf5)
#     f1s5 = f1s_from_conf(conf5)

#     f1s_damage = [f1s5[c] for c in [1, 2, 3, 4]]
#     macro_damage = float(np.mean(f1s_damage))
#     loc_p, loc_r, loc_f1 = prf_from_counts(loc_tp, loc_fp, loc_fn)

#     if debug_destroyed:
#         gt = int(conf5[4, :].sum())
#         pred = int(conf5[:, 4].sum())
#         tp = int(conf5[4, 4])
#         fp = pred - tp
#         fn = gt - tp
#         print(f"[DESTROYED DEBUG] GT={gt}  PRED={pred}  TP={tp}  FP={fp}  FN={fn}")

#     return acc5, loc_p, loc_r, loc_f1, macro_damage, f1s_damage


# def strip_known_suffix(name_noext: str, suffixes):
#     for suf in suffixes:
#         if name_noext.endswith(suf):
#             return name_noext[:-len(suf)]
#     return name_noext


# def load_loc_pack(loc_dir: str, device, loc_glob="*_idabd_ft_best.pth"):
#     weights = sorted(glob.glob(path.join(loc_dir, loc_glob)))
#     if not weights:
#         raise FileNotFoundError(f"No Stage-1 loc weights found in {loc_dir} with {loc_glob}")

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


# def find_stage2_calib(stage2_dir: str, ckpt_path: str) -> str:
#     base = path.basename(ckpt_path)
#     noext = path.splitext(base)[0]
#     stem = strip_known_suffix(noext, ["_idabd_stage2_ft_best", "_idabd_stage2_ft_last"])
#     cand = path.join(stage2_dir, f"{stem}_idabd_stage2_calib_vector_scaling.npz")
#     if path.exists(cand):
#         return cand
#     g = sorted(glob.glob(path.join(stage2_dir, f"{stem}*_calib*vector*scaling*.npz")))
#     return g[0] if g else ""


# def load_stage2_pack(stage2_dir: str, device, stage2_glob="*_idabd_stage2_ft_best.pth", stage2_max_models: int = 12):
#     ckpts = sorted(glob.glob(path.join(stage2_dir, stage2_glob)))
#     if not ckpts:
#         raise FileNotFoundError(f"No Stage-2 checkpoints found in {stage2_dir} ({stage2_glob})")

#     if stage2_max_models > 0 and len(ckpts) > stage2_max_models:
#         print(f"[STAGE2] WARNING: found {len(ckpts)} models; keeping first {stage2_max_models}.")
#         ckpts = ckpts[:stage2_max_models]

#     pack = []
#     for ck in ckpts:
#         calib = find_stage2_calib(stage2_dir, ck)
#         if not calib:
#             raise FileNotFoundError(f"Missing calibration for stage2 ckpt: {ck}")

#         model = build_damage_model_from_weight(ck).to(device).eval()
#         for p in model.parameters():
#             p.requires_grad = False

#         d = np.load(calib)
#         W = torch.from_numpy(d["W"].astype(np.float32)).to(device)
#         b = torch.from_numpy(d["b"].astype(np.float32)).to(device)
#         pack.append((model, W, b))

#     return pack, ckpts


# # =============================================================================
# # Auto helpers
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
#     return (img_dirs[0] if img_dirs else IMG_DIR), (mask_dirs[0] if mask_dirs else MASK_DIR), img_dirs, mask_dirs


# def auto_find_stage2_dir():
#     dirs = _candidate_dirs(DEFAULT_STAGE2_DIR, f"../{DEFAULT_STAGE2_DIR}")
#     return (dirs[0] if dirs else DEFAULT_STAGE2_DIR), dirs


# def auto_find_loc_dir():
#     rels = []
#     for c in DEFAULT_LOC_DIR_CANDIDATES:
#         rels += [c, f"../{c}", f"two_stage_pipeline/{c}", f"../two_stage_pipeline/{c}"]
#     dirs = _candidate_dirs(*rels)
#     return (dirs[0] if dirs else ""), dirs


# # =============================================================================
# # Main
# # =============================================================================
# def main():
#     ap = argparse.ArgumentParser()

#     ap.add_argument("--img_dir", default="")
#     ap.add_argument("--mask_dir", default="")
#     ap.add_argument("--gt_split", choices=["pre", "post"], default="post")

#     ap.add_argument("--val_ratio", type=float, default=0.1)
#     ap.add_argument("--test_ratio", type=float, default=0.1)
#     ap.add_argument("--seed", type=int, default=0)

#     ap.add_argument("--loc_dir", default="")
#     ap.add_argument("--loc_glob", default="*_idabd_ft_best.pth")
#     ap.add_argument("--stage2_dir", default="")
#     ap.add_argument("--stage2_glob", default="*_idabd_stage2_ft_best.pth")
#     ap.add_argument("--stage2_max_models", type=int, default=12)
#     ap.add_argument("--loc_thresh", type=float, default=0.5)

#     # Must match training definition
#     ap.add_argument("--alpha_contrast", type=float, default=0.65)
#     ap.add_argument("--alpha_edge", type=float, default=0.35)
#     ap.add_argument("--clahe_clip", type=float, default=2.0)
#     ap.add_argument("--clahe_grid", type=int, default=8)
#     ap.add_argument("--canny_t1", type=int, default=50)
#     ap.add_argument("--canny_t2", type=int, default=150)
#     ap.add_argument("--edge_dilate", type=int, default=1)

#     ap.add_argument("--workers", type=int, default=2)
#     ap.add_argument("--amp", action="store_true")

#     ap.add_argument("--csv_path", default="idabd_pipeline_ENSEMBLE_LOC+DMG_CAL_ONLY_RGB_CONTRAST_EDGE_ONLY.csv")
#     ap.add_argument("--overwrite_csv", action="store_true")
#     ap.add_argument("--debug_destroyed", action="store_true")

#     args = ap.parse_args()
#     if args.val_ratio + args.test_ratio >= 1.0:
#         raise ValueError("val_ratio + test_ratio must be < 1.0")

#     img_auto, mask_auto, img_tried, mask_tried = auto_find_dataset_dirs()
#     if not args.img_dir:
#         args.img_dir = img_auto
#     if not args.mask_dir:
#         args.mask_dir = mask_auto

#     if not args.stage2_dir:
#         s2_auto, s2_tried = auto_find_stage2_dir()
#         args.stage2_dir = s2_auto
#     else:
#         s2_tried = []

#     if not args.loc_dir:
#         loc_auto, loc_tried = auto_find_loc_dir()
#         args.loc_dir = loc_auto
#     else:
#         loc_tried = []

#     print("\n[AUTO CONFIG]")
#     print(f"  img_dir       : {args.img_dir}")
#     print(f"  mask_dir      : {args.mask_dir}")
#     print(f"  loc_dir       : {args.loc_dir if args.loc_dir else '(NOT FOUND)'}")
#     print(f"  stage2_dir    : {args.stage2_dir}")
#     print(f"  alpha_contrast: {args.alpha_contrast}")
#     print(f"  alpha_edge    : {args.alpha_edge}")
#     print(f"  CLAHE         : clip={args.clahe_clip}, grid={args.clahe_grid}")
#     print(f"  Canny         : t1={args.canny_t1}, t2={args.canny_t2}, dilate={args.edge_dilate}")

#     if not path.isdir(args.img_dir):
#         raise FileNotFoundError("Could not find img_dir. Pass --img_dir explicitly.")
#     if not path.isdir(args.mask_dir):
#         raise FileNotFoundError("Could not find mask_dir. Pass --mask_dir explicitly.")
#     if not args.loc_dir or not path.isdir(args.loc_dir):
#         if loc_tried:
#             print("\n[AUTO DEBUG] Tried loc_dir candidates:")
#             for d in loc_tried:
#                 print("  -", d)
#         raise FileNotFoundError("Could not find loc_dir. Pass --loc_dir explicitly.")
#     if not path.isdir(args.stage2_dir):
#         if s2_tried:
#             print("\n[AUTO DEBUG] Tried stage2_dir candidates:")
#             for d in s2_tried:
#                 print("  -", d)
#         raise FileNotFoundError("Could not find stage2_dir. Pass --stage2_dir explicitly.")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"[DEVICE] {device}")

#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(args.seed)

#     ensure_csv(args.csv_path, overwrite=args.overwrite_csv)
#     print(f"[CSV   ] Ready: {args.csv_path}")

#     triplets = build_stage2_triplets(args.img_dir, args.mask_dir, gt_split=args.gt_split)
#     if not triplets:
#         raise FileNotFoundError("No (pre, post, mask) triplets found.")

#     rng = random.Random(args.seed)
#     rng.shuffle(triplets)
#     n = len(triplets)

#     n_test = max(1, int(n * args.test_ratio))
#     n_val  = max(1, int(n * args.val_ratio))
#     if n_test + n_val >= n:
#         n_test = max(1, min(n_test, n - 2))
#         n_val  = max(1, min(n_val,  n - 1 - n_test))

#     test_tr  = triplets[:n_test]
#     print(f"[DATA  ] total={n} | test={len(test_tr)}")

#     test_ds = IdaBDEval(
#         test_tr,
#         alpha_contrast=args.alpha_contrast,
#         alpha_edge=args.alpha_edge,
#         clahe_clip=args.clahe_clip,
#         clahe_grid=args.clahe_grid,
#         canny_t1=args.canny_t1,
#         canny_t2=args.canny_t2,
#         edge_dilate=args.edge_dilate,
#     )
#     test_ld = DataLoader(test_ds, batch_size=1, shuffle=False,
#                          num_workers=max(0, args.workers), pin_memory=True)

#     loc_pack, loc_used = load_loc_pack(args.loc_dir, device=device, loc_glob=args.loc_glob)
#     print(f"[LOC   ] models: {len(loc_used)}")

#     stage2_pack, stage2_used = load_stage2_pack(
#         args.stage2_dir, device=device,
#         stage2_glob=args.stage2_glob,
#         stage2_max_models=args.stage2_max_models
#     )
#     print(f"[STAGE2] models: {len(stage2_used)}")

#     acc5, locP, locR, locF, macroD, f1sD = eval_pipeline_ensemble_cal(
#         test_ld, device,
#         loc_pack=loc_pack,
#         stage2_pack=stage2_pack,
#         loc_thresh=args.loc_thresh,
#         use_amp=args.amp,
#         debug_destroyed=args.debug_destroyed
#     )

#     print("\n==================== ENSEMBLED PIPELINE (CALIBRATED ONLY) ====================")
#     print("END-TO-END: Stage-1 ensemble gate -> Stage-2 calibrated ensemble damage")
#     print(f"pipeline_acc(0..4)={acc5:.6f}")
#     print(f"F1 Localization (Building vs Background)={locF:.6f}  (P={locP:.6f}, R={locR:.6f})")
#     print(f"macroF1(Damage 1..4, end-to-end conf)={macroD:.6f}")
#     for i in range(4):
#         print(f"F1 {CLASS_NAMES_4[i]:>9s}: {f1sD[i]:.6f}")
#     print("=============================================================================\n")

#     row = {
#         "seed": args.seed,
#         "loc_dir": args.loc_dir,
#         "loc_models_used": len(loc_used),
#         "stage2_dir": args.stage2_dir,
#         "stage2_models_used": len(stage2_used),
#         "loc_glob": args.loc_glob,
#         "stage2_glob": args.stage2_glob,
#         "loc_thresh": args.loc_thresh,

#         "alpha_contrast": args.alpha_contrast,
#         "alpha_edge": args.alpha_edge,
#         "clahe_clip": args.clahe_clip,
#         "clahe_grid": args.clahe_grid,
#         "canny_t1": args.canny_t1,
#         "canny_t2": args.canny_t2,
#         "edge_dilate": args.edge_dilate,

#         "pipeline_acc_cal": acc5,
#         "pipeline_loc_precision_cal": locP,
#         "pipeline_loc_recall_cal": locR,
#         "pipeline_loc_f1_cal": locF,
#         "pipeline_macroF1_damage_cal": macroD,
#         "pipeline_f1_no_damage_cal": f1sD[0],
#         "pipeline_f1_minor_cal": f1sD[1],
#         "pipeline_f1_major_cal": f1sD[2],
#         "pipeline_f1_destroyed_cal": f1sD[3],
#     }
#     append_csv_row(args.csv_path, row)
#     print(f"[CSV   ] Appended -> {args.csv_path}")


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
EVAL: ENSEMBLE STAGE-1 LOCALIZATION + ENSEMBLE STAGE-2 DAMAGE (CALIBRATED ONLY)
IDABD end-to-end pipeline — RGB (+ optional CONTRAST+EDGE)

RUNS WITH JUST:
  python eval_idabd_stage2_pipeline_RGB_contrast_edge_only_earlyStop.py

IMPORTANT (matches the updated training script fix):
- By default, EVAL uses RAW post RGB (NO fusion) because fusion is TRAIN augmentation only.
- If you want to evaluate with fusion, pass:  --apply_fusion 1

Input:
- x6 = [pre_RGB, post_RGB_used]
- post_RGB_used = post RGB (raw) OR fused with contrast+edge (optional)

Ensembling:
- Stage-1: average building probabilities across loc models
- Stage-2: apply vector scaling per model, softmax, then average probabilities

Split MUST match training (seed/val_ratio/test_ratio).
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

DEFAULT_STAGE2_DIR = "idabd_stage2_damage_ft_checkpoints_RGB_CONTRAST_EDGE_ONLY"

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
    "loc_dir",
    "loc_models_used",
    "stage2_dir",
    "stage2_models_used",
    "loc_glob",
    "stage2_glob",
    "loc_thresh",

    "apply_fusion",
    "alpha_contrast",
    "alpha_edge",
    "clahe_clip",
    "clahe_grid",
    "canny_t1",
    "canny_t2",
    "edge_dilate",

    "pipeline_acc_cal",
    "pipeline_loc_precision_cal",
    "pipeline_loc_recall_cal",
    "pipeline_loc_f1_cal",
    "pipeline_macroF1_damage_cal",
    "pipeline_f1_no_damage_cal",
    "pipeline_f1_minor_cal",
    "pipeline_f1_major_cal",
    "pipeline_f1_destroyed_cal",
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
# Pairing
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
# Fusion (contrast+edge) + preprocess
# =============================================================================
def clahe_contrast_bgr(img_bgr: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(grid), int(grid)))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out


def canny_edges(img_bgr: np.ndarray, t1: int = 80, t2: int = 200, dilate: int = 0) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(gray, int(t1), int(t2))
    if dilate and int(dilate) > 0:
        k = int(dilate) * 2 + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        e = cv2.dilate(e, ker, iterations=1)
    return e


def blend_bgr(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0:
        return a
    if alpha >= 1:
        return b
    return cv2.addWeighted(a, 1.0 - alpha, b, alpha, 0.0)


def overlay_edges_on_bgr(img_bgr: np.ndarray, edges_u8: np.ndarray, alpha_edge: float) -> np.ndarray:
    alpha = float(np.clip(alpha_edge, 0.0, 1.0))
    if alpha <= 0:
        return img_bgr
    edge_rgb = cv2.cvtColor(edges_u8, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, edge_rgb, alpha, 0.0)


def fuse_contrast_edge_post(
    post_bgr: np.ndarray,
    alpha_contrast: float,
    alpha_edge: float,
    clahe_clip: float,
    clahe_grid: int,
    canny_t1: int,
    canny_t2: int,
    edge_dilate: int,
) -> np.ndarray:
    post_c = clahe_contrast_bgr(post_bgr, clip_limit=clahe_clip, grid=clahe_grid)
    post_m = blend_bgr(post_bgr, post_c, alpha_contrast)
    e = canny_edges(post_m, t1=canny_t1, t2=canny_t2, dilate=edge_dilate)
    return overlay_edges_on_bgr(post_m, e, alpha_edge=alpha_edge)


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


def preprocess_rgb(img_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)


def preprocess_6ch(pre_bgr: np.ndarray, post_bgr: np.ndarray) -> torch.Tensor:
    return torch.cat([preprocess_rgb(pre_bgr), preprocess_rgb(post_bgr)], dim=0)


def load_mask_raw(mask_path: str) -> np.ndarray:
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(mask_path)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return m.astype(np.int64)


class IdaBDEval(Dataset):
    def __init__(
        self,
        triplets,
        apply_fusion: bool = False,
        alpha_contrast=0.45,
        alpha_edge=0.10,
        clahe_clip=2.0,
        clahe_grid=8,
        canny_t1=80,
        canny_t2=200,
        edge_dilate=0,
    ):
        self.triplets = triplets
        self.apply_fusion = bool(apply_fusion)
        self.alpha_contrast = float(alpha_contrast)
        self.alpha_edge = float(alpha_edge)
        self.clahe_clip = float(clahe_clip)
        self.clahe_grid = int(clahe_grid)
        self.canny_t1 = int(canny_t1)
        self.canny_t2 = int(canny_t2)
        self.edge_dilate = int(edge_dilate)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        pre_path, post_path, mask_path = self.triplets[idx]

        pre  = cv2.imread(pre_path, cv2.IMREAD_COLOR)
        post = cv2.imread(post_path, cv2.IMREAD_COLOR)
        if pre is None:
            raise FileNotFoundError(pre_path)
        if post is None:
            raise FileNotFoundError(post_path)

        m = load_mask_raw(mask_path)

        # IMPORTANT: default is RAW post. Fusion is optional via --apply_fusion 1
        if self.apply_fusion:
            post = fuse_contrast_edge_post(
                post,
                alpha_contrast=self.alpha_contrast,
                alpha_edge=self.alpha_edge,
                clahe_clip=self.clahe_clip,
                clahe_grid=self.clahe_grid,
                canny_t1=self.canny_t1,
                canny_t2=self.canny_t2,
                edge_dilate=self.edge_dilate,
            )

        pre, pad_hw = pad_to_factor(pre, 32)
        post, _ = pad_to_factor(post, 32)
        if pad_hw != (0, 0):
            m = np.pad(m, ((0, pad_hw[0]), (0, pad_hw[1])), mode="edge")

        x = preprocess_6ch(pre, post)
        y = torch.from_numpy(m).long()
        return x.float(), y.long(), path.basename(post_path)


# =============================================================================
# Load models
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
        raise ValueError(f"Unrecognized stage-2 backbone in filename: {fname}")

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
        raise ValueError(f"Unrecognized stage-1 backbone in filename: {fname}")

    sd = load_state_dict_any(weight_path)
    model.load_state_dict(sd, strict=True)
    return model


# =============================================================================
# Metrics + AMP helpers
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
def predict_build_mask_ensemble(loc_pack, x6, thresh, amp_ctx):
    pre = x6[:, 0:3, :, :]
    prob_sum = None

    for (m, a, b) in loc_pack:
        with amp_ctx():
            logit = m(pre)
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
def predict_damage_ensemble_cal(stage2_pack, x6, amp_ctx):
    prob_sum = None

    for (m, W, b) in stage2_pack:
        with amp_ctx():
            logits = m(x6)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            z = logits[:, 1:5, :, :]

        with torch.cuda.amp.autocast(enabled=False):
            z = z.float()
            Wf = W.float()
            bf = b.float()
            z2 = z.permute(0, 2, 3, 1).contiguous()
            z2 = torch.matmul(z2, Wf.t()) + bf
            z2 = z2.permute(0, 3, 1, 2).contiguous()
            probs = F.softmax(z2, dim=1)

        prob_sum = probs if prob_sum is None else (prob_sum + probs)

    prob_mean = prob_sum / float(len(stage2_pack))
    return torch.argmax(prob_mean, dim=1) + 1


@torch.no_grad()
def eval_pipeline_ensemble_cal(loader, device, loc_pack, stage2_pack, loc_thresh, use_amp: bool, debug_destroyed: bool):
    amp_ctx = make_amp_ctx(device, use_amp)

    conf5 = np.zeros((5, 5), dtype=np.int64)
    loc_tp = 0.0
    loc_fp = 0.0
    loc_fn = 0.0
    build_tensor = BUILD_TENSOR_CPU.to(device)

    for x6, raw, _ in loader:
        x6 = x6.to(device, non_blocking=True)
        raw = raw.to(device, non_blocking=True)

        build_mask = predict_build_mask_ensemble(loc_pack, x6, loc_thresh, amp_ctx=amp_ctx)
        pred_damage = predict_damage_ensemble_cal(stage2_pack, x6, amp_ctx=amp_ctx)

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
            g_valid = g[v]
            p_valid = np.clip(p[v], 0, 4)
            _confusion_add(conf5, g_valid, p_valid, 5)

    acc5 = acc_from_conf(conf5)
    f1s5 = f1s_from_conf(conf5)

    f1s_damage = [f1s5[c] for c in [1, 2, 3, 4]]
    macro_damage = float(np.mean(f1s_damage))
    loc_p, loc_r, loc_f1 = prf_from_counts(loc_tp, loc_fp, loc_fn)

    if debug_destroyed:
        gt = int(conf5[4, :].sum())
        pred = int(conf5[:, 4].sum())
        tp = int(conf5[4, 4])
        fp = pred - tp
        fn = gt - tp
        print(f"[DESTROYED DEBUG] GT={gt}  PRED={pred}  TP={tp}  FP={fp}  FN={fn}")

    return acc5, loc_p, loc_r, loc_f1, macro_damage, f1s_damage


def strip_known_suffix(name_noext: str, suffixes):
    for suf in suffixes:
        if name_noext.endswith(suf):
            return name_noext[:-len(suf)]
    return name_noext


def load_loc_pack(loc_dir: str, device, loc_glob="*_idabd_ft_best.pth"):
    weights = sorted(glob.glob(path.join(loc_dir, loc_glob)))
    if not weights:
        raise FileNotFoundError(f"No Stage-1 loc weights found in {loc_dir} with {loc_glob}")

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


def find_stage2_calib(stage2_dir: str, ckpt_path: str) -> str:
    base = path.basename(ckpt_path)
    noext = path.splitext(base)[0]
    stem = strip_known_suffix(noext, ["_idabd_stage2_ft_best", "_idabd_stage2_ft_last"])
    cand = path.join(stage2_dir, f"{stem}_idabd_stage2_calib_vector_scaling.npz")
    if path.exists(cand):
        return cand
    g = sorted(glob.glob(path.join(stage2_dir, f"{stem}*_calib*vector*scaling*.npz")))
    return g[0] if g else ""


def load_stage2_pack(stage2_dir: str, device, stage2_glob="*_idabd_stage2_ft_best.pth", stage2_max_models: int = 12):
    ckpts = sorted(glob.glob(path.join(stage2_dir, stage2_glob)))
    if not ckpts:
        raise FileNotFoundError(f"No Stage-2 checkpoints found in {stage2_dir} ({stage2_glob})")

    if stage2_max_models > 0 and len(ckpts) > stage2_max_models:
        print(f"[STAGE2] WARNING: found {len(ckpts)} models; keeping first {stage2_max_models}.")
        ckpts = ckpts[:stage2_max_models]

    pack = []
    for ck in ckpts:
        calib = find_stage2_calib(stage2_dir, ck)
        if not calib:
            raise FileNotFoundError(f"Missing calibration for stage2 ckpt: {ck}")

        model = build_damage_model_from_weight(ck).to(device).eval()
        for p in model.parameters():
            p.requires_grad = False

        d = np.load(calib)
        W = torch.from_numpy(d["W"].astype(np.float32)).to(device)
        b = torch.from_numpy(d["b"].astype(np.float32)).to(device)
        pack.append((model, W, b))

    return pack, ckpts


# =============================================================================
# Auto helpers
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
    return (img_dirs[0] if img_dirs else IMG_DIR), (mask_dirs[0] if mask_dirs else MASK_DIR), img_dirs, mask_dirs


def auto_find_stage2_dir():
    dirs = _candidate_dirs(DEFAULT_STAGE2_DIR, f"../{DEFAULT_STAGE2_DIR}")
    return (dirs[0] if dirs else DEFAULT_STAGE2_DIR), dirs


def auto_find_loc_dir():
    rels = []
    for c in DEFAULT_LOC_DIR_CANDIDATES:
        rels += [c, f"../{c}", f"two_stage_pipeline/{c}", f"../two_stage_pipeline/{c}"]
    dirs = _candidate_dirs(*rels)
    return (dirs[0] if dirs else ""), dirs


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
    ap.add_argument("--stage2_dir", default="")
    ap.add_argument("--stage2_glob", default="*_idabd_stage2_ft_best.pth")
    ap.add_argument("--stage2_max_models", type=int, default=12)
    ap.add_argument("--loc_thresh", type=float, default=0.5)

    # IMPORTANT: default NO fusion to match updated training/testing protocol
    ap.add_argument("--apply_fusion", type=int, default=0, help="0=raw post (default), 1=apply contrast+edge fusion")

    # Fusion params (only used if apply_fusion=1)
    ap.add_argument("--alpha_contrast", type=float, default=0.45)
    ap.add_argument("--alpha_edge", type=float, default=0.10)
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_grid", type=int, default=8)
    ap.add_argument("--canny_t1", type=int, default=80)
    ap.add_argument("--canny_t2", type=int, default=200)
    ap.add_argument("--edge_dilate", type=int, default=0)

    ap.add_argument("--workers", type=int, default=2)

    # AMP on demand (keep as flag to match your style)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--csv_path", default="idabd_pipeline_ENSEMBLE_LOC+DMG_CAL_ONLY_RGB_CONTRAST_EDGE_ONLY.csv")
    ap.add_argument("--overwrite_csv", action="store_true")
    ap.add_argument("--debug_destroyed", action="store_true")

    args = ap.parse_args()
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    img_auto, mask_auto, img_tried, mask_tried = auto_find_dataset_dirs()
    if not args.img_dir:
        args.img_dir = img_auto
    if not args.mask_dir:
        args.mask_dir = mask_auto

    if not args.stage2_dir:
        s2_auto, s2_tried = auto_find_stage2_dir()
        args.stage2_dir = s2_auto
    else:
        s2_tried = []

    if not args.loc_dir:
        loc_auto, loc_tried = auto_find_loc_dir()
        args.loc_dir = loc_auto
    else:
        loc_tried = []

    print("\n[AUTO CONFIG]")
    print(f"  img_dir      : {args.img_dir}")
    print(f"  mask_dir     : {args.mask_dir}")
    print(f"  loc_dir      : {args.loc_dir if args.loc_dir else '(NOT FOUND)'}")
    print(f"  stage2_dir   : {args.stage2_dir}")
    print(f"  apply_fusion : {bool(args.apply_fusion)}")
    if args.apply_fusion:
        print(f"  contrast     : alpha={args.alpha_contrast}, clahe_clip={args.clahe_clip}, clahe_grid={args.clahe_grid}")
        print(f"  edges        : alpha={args.alpha_edge}, t1={args.canny_t1}, t2={args.canny_t2}, dilate={args.edge_dilate}")

    if not path.isdir(args.img_dir):
        raise FileNotFoundError("Could not find img_dir. Pass --img_dir explicitly.")
    if not path.isdir(args.mask_dir):
        raise FileNotFoundError("Could not find mask_dir. Pass --mask_dir explicitly.")
    if not args.loc_dir or not path.isdir(args.loc_dir):
        if loc_tried:
            print("\n[AUTO DEBUG] Tried loc_dir candidates:")
            for d in loc_tried:
                print("  -", d)
        raise FileNotFoundError("Could not find loc_dir. Pass --loc_dir explicitly.")
    if not path.isdir(args.stage2_dir):
        if s2_tried:
            print("\n[AUTO DEBUG] Tried stage2_dir candidates:")
            for d in s2_tried:
                print("  -", d)
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

    rng = random.Random(args.seed)
    rng.shuffle(triplets)
    n = len(triplets)

    n_test = max(1, int(n * args.test_ratio))
    n_val  = max(1, int(n * args.val_ratio))
    if n_test + n_val >= n:
        n_test = max(1, min(n_test, n - 2))
        n_val  = max(1, min(n_val,  n - 1 - n_test))

    test_tr = triplets[:n_test]
    print(f"[DATA  ] total={n} | test={len(test_tr)}")

    test_ds = IdaBDEval(
        test_tr,
        apply_fusion=bool(args.apply_fusion),
        alpha_contrast=args.alpha_contrast,
        alpha_edge=args.alpha_edge,
        clahe_clip=args.clahe_clip,
        clahe_grid=args.clahe_grid,
        canny_t1=args.canny_t1,
        canny_t2=args.canny_t2,
        edge_dilate=args.edge_dilate,
    )
    test_ld = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=max(0, args.workers), pin_memory=True
    )

    loc_pack, loc_used = load_loc_pack(args.loc_dir, device=device, loc_glob=args.loc_glob)
    print(f"[LOC   ] models: {len(loc_used)}")

    stage2_pack, stage2_used = load_stage2_pack(
        args.stage2_dir, device=device,
        stage2_glob=args.stage2_glob,
        stage2_max_models=args.stage2_max_models
    )
    print(f"[STAGE2] models: {len(stage2_used)}")

    acc5, locP, locR, locF, macroD, f1sD = eval_pipeline_ensemble_cal(
        test_ld, device,
        loc_pack=loc_pack,
        stage2_pack=stage2_pack,
        loc_thresh=args.loc_thresh,
        use_amp=args.amp,
        debug_destroyed=args.debug_destroyed
    )

    print("\n==================== ENSEMBLED PIPELINE (CALIBRATED ONLY) ====================")
    print("END-TO-END: Stage-1 ensemble gate -> Stage-2 calibrated ensemble damage")
    print(f"pipeline_acc(0..4)={acc5:.6f}")
    print(f"F1 Localization={locF:.6f}  (P={locP:.6f}, R={locR:.6f})")
    print(f"macroF1(Damage 1..4)={macroD:.6f}")
    for i in range(4):
        print(f"F1 {CLASS_NAMES_4[i]:>9s}: {f1sD[i]:.6f}")
    print("=============================================================================\n")

    row = {
        "seed": args.seed,
        "loc_dir": args.loc_dir,
        "loc_models_used": len(loc_used),
        "stage2_dir": args.stage2_dir,
        "stage2_models_used": len(stage2_used),
        "loc_glob": args.loc_glob,
        "stage2_glob": args.stage2_glob,
        "loc_thresh": args.loc_thresh,

        "apply_fusion": int(bool(args.apply_fusion)),
        "alpha_contrast": args.alpha_contrast,
        "alpha_edge": args.alpha_edge,
        "clahe_clip": args.clahe_clip,
        "clahe_grid": args.clahe_grid,
        "canny_t1": args.canny_t1,
        "canny_t2": args.canny_t2,
        "edge_dilate": args.edge_dilate,

        "pipeline_acc_cal": acc5,
        "pipeline_loc_precision_cal": locP,
        "pipeline_loc_recall_cal": locR,
        "pipeline_loc_f1_cal": locF,
        "pipeline_macroF1_damage_cal": macroD,
        "pipeline_f1_no_damage_cal": f1sD[0],
        "pipeline_f1_minor_cal": f1sD[1],
        "pipeline_f1_major_cal": f1sD[2],
        "pipeline_f1_destroyed_cal": f1sD[3],
    }
    append_csv_row(args.csv_path, row)
    print(f"[CSV   ] Appended -> {args.csv_path}")


if __name__ == "__main__":
    main()
