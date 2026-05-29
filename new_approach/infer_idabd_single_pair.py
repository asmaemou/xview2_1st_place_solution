#!/usr/bin/env python3

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------
# Make sure xView2 project and new_approach are importable
# ---------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
XVIEW2_ROOT = CURRENT_DIR.parent

sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(XVIEW2_ROOT))

from eval_idabd_pipeline_ensemble_loc_and_damage_cal_only import (
    pad_to_factor,
    preprocess_6ch,
    load_loc_pack,
    load_stage2_pack,
    predict_build_mask_ensemble,
    predict_damage_ensemble_cal,
)


CLASS_NAMES = {
    1: "no_damage",
    2: "minor",
    3: "major",
    4: "destroyed",
}

CLASS_RGB = {
    0: (0, 0, 0),          # background
    1: (34, 197, 94),     # no damage - green
    2: (234, 179, 8),     # minor - yellow
    3: (249, 115, 22),    # major - orange
    4: (220, 38, 38),     # destroyed - red
}


def unsharp_bgr(img_bgr: np.ndarray, amount: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """
    Match the RGB + Unsharp experiment.

    Input:
      BGR uint8 image

    Output:
      BGR uint8 sharpened image

    Formula:
      sharp = image * (1 + amount) - blur * amount
    """
    if amount <= 0:
        return img_bgr

    blur = cv2.GaussianBlur(
        img_bgr,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
    )

    sharp = cv2.addWeighted(
        img_bgr,
        1.0 + float(amount),
        blur,
        -float(amount),
        0,
    )

    return sharp


def blend_bgr(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blend two BGR images.

    post_used = (1 - alpha) * post + alpha * unsharp(post)
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))

    if alpha <= 0:
        return a

    if alpha >= 1:
        return b

    return cv2.addWeighted(a, 1.0 - alpha, b, alpha, 0.0)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for class_id, rgb in CLASS_RGB.items():
        color[mask == class_id] = rgb

    return color


def normalize_ground_truth_mask(mask_path: str, target_shape) -> np.ndarray:
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    if gt_mask is None:
        raise FileNotFoundError(f"Could not read ground-truth mask: {mask_path}")

    if gt_mask.ndim == 3:
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)

    gt_mask = gt_mask.astype(np.uint8)

    if gt_mask.shape[:2] != target_shape:
        print(
            f"[WARNING] Ground-truth mask shape {gt_mask.shape[:2]} does not match "
            f"prediction shape {target_shape}. Resizing with nearest-neighbor.",
            flush=True,
        )

        gt_mask = cv2.resize(
            gt_mask,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    unique_values = set(np.unique(gt_mask).tolist())
    allowed_values = {0, 1, 2, 3, 4, 255}

    if not unique_values.issubset(allowed_values):
        print(
            "[WARNING] Ground-truth mask has unexpected values:",
            sorted(unique_values),
            flush=True,
        )
        print(
            "[WARNING] Expected class IDs: 0, 1, 2, 3, 4, 255.",
            flush=True,
        )

    if unique_values.issubset({0, 255}):
        print(
            "[WARNING] Ground-truth mask appears binary with values only 0/255. "
            "For per-class F1, the mask should contain class labels: "
            "0=background, 1=no_damage, 2=minor, 3=major, 4=destroyed, 255=ignore.",
            flush=True,
        )

    return gt_mask


def compute_f1_scores(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """
    pred_mask and gt_mask expected values:
      0   = background
      1   = no damage
      2   = minor damage
      3   = major damage
      4   = destroyed
      255 = ignore
    """

    class_names = {
        1: "no_damage_f1",
        2: "minor_f1",
        3: "major_f1",
        4: "destroyed_f1",
    }

    valid = gt_mask != 255

    scores = {}

    for class_id, metric_name in class_names.items():
        pred_class = (pred_mask == class_id) & valid
        gt_class = (gt_mask == class_id) & valid

        tp = np.logical_and(pred_class, gt_class).sum()
        fp = np.logical_and(pred_class, ~gt_class).sum()
        fn = np.logical_and(~pred_class, gt_class).sum()

        f1 = (2 * tp) / ((2 * tp) + fp + fn + 1e-9)
        scores[metric_name] = round(float(f1), 4)

    scores["macro_f1"] = round(
        float(
            np.mean(
                [
                    scores["no_damage_f1"],
                    scores["minor_f1"],
                    scores["major_f1"],
                    scores["destroyed_f1"],
                ]
            )
        ),
        4,
    )

    return scores


def compute_localization_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """
    Localization compares building vs background.
    Any class 1..4 is considered building.
    """

    valid = gt_mask != 255

    pred_building = (pred_mask > 0) & valid
    gt_building = np.isin(gt_mask, [1, 2, 3, 4]) & valid

    tp = np.logical_and(pred_building, gt_building).sum()
    fp = np.logical_and(pred_building, ~gt_building).sum()
    fn = np.logical_and(~pred_building, gt_building).sum()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = (2 * tp) / ((2 * tp) + fp + fn + 1e-9)

    return {
        "localization_precision": round(float(precision), 4),
        "localization_recall": round(float(recall), 4),
        "localization_f1": round(float(f1), 4),
    }


def summarize_by_connected_components(pred_mask: np.ndarray, min_area: int = 20) -> dict:
    """
    Count building instances approximately using connected components.

    pred_mask:
      0 = background
      1 = no_damage
      2 = minor
      3 = major
      4 = destroyed
    """

    building_binary = (pred_mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        building_binary,
        connectivity=8,
    )

    counts = {
        "no_damage": 0,
        "minor": 0,
        "major": 0,
        "destroyed": 0,
    }

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        component_pixels = pred_mask[labels == label_id]
        component_pixels = component_pixels[component_pixels > 0]

        if component_pixels.size == 0:
            continue

        class_id = int(np.bincount(component_pixels, minlength=5).argmax())
        class_name = CLASS_NAMES.get(class_id)

        if class_name:
            counts[class_name] += 1

    total = sum(counts.values())

    if total == 0:
        percentages = {
            "no_damage": 0,
            "minor": 0,
            "major": 0,
            "destroyed": 0,
        }
    else:
        percentages = {
            key: round((value / total) * 100, 2)
            for key, value in counts.items()
        }

    return {
        "total_buildings": total,
        "damage_counts": counts,
        "damage_percentages": percentages,
        "counting_method": "connected_components_majority_class",
    }


def save_csv(csv_path: str, summary: dict) -> None:
    counts = summary["damage_counts"]
    percentages = summary["damage_percentages"]
    metrics = summary.get("metrics", {})
    runtime = summary.get("runtime", {})
    model_info = summary.get("model_info", {})

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["section", "name", "value"])

        writer.writerow(["damage_count", "no_damage", counts["no_damage"]])
        writer.writerow(["damage_count", "minor", counts["minor"]])
        writer.writerow(["damage_count", "major", counts["major"]])
        writer.writerow(["damage_count", "destroyed", counts["destroyed"]])

        writer.writerow(["damage_percentage", "no_damage", percentages["no_damage"]])
        writer.writerow(["damage_percentage", "minor", percentages["minor"]])
        writer.writerow(["damage_percentage", "major", percentages["major"]])
        writer.writerow(["damage_percentage", "destroyed", percentages["destroyed"]])

        for key, value in metrics.items():
            writer.writerow(["metric", key, value])

        for key, value in runtime.items():
            writer.writerow(["runtime", key, value])

        for key, value in model_info.items():
            writer.writerow(["model_info", key, value])


def make_overlay(
    post_bgr: np.ndarray,
    color_mask_rgb: np.ndarray,
    pred_mask: np.ndarray,
    overlay_output: str,
) -> None:
    post_rgb = cv2.cvtColor(post_bgr, cv2.COLOR_BGR2RGB)

    if color_mask_rgb.shape[:2] != post_rgb.shape[:2]:
        color_mask_rgb = cv2.resize(
            color_mask_rgb,
            (post_rgb.shape[1], post_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    if pred_mask.shape[:2] != post_rgb.shape[:2]:
        pred_mask = cv2.resize(
            pred_mask,
            (post_rgb.shape[1], post_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_pixels = pred_mask > 0

    overlay = post_rgb.copy()

    if mask_pixels.any():
        overlay[mask_pixels] = (
            0.55 * post_rgb[mask_pixels] + 0.45 * color_mask_rgb[mask_pixels]
        ).astype(np.uint8)

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(overlay_output), overlay_bgr)


def get_gpu_runtime_info(device: torch.device) -> dict:
    if device.type != "cuda":
        return {
            "gpu_name": "CPU only",
            "gpu_peak_allocated_mb": 0,
            "gpu_peak_reserved_mb": 0,
        }

    gpu_index = torch.cuda.current_device()

    return {
        "gpu_name": torch.cuda.get_device_name(gpu_index),
        "gpu_peak_allocated_mb": round(
            float(torch.cuda.max_memory_allocated(gpu_index)) / (1024 ** 2),
            2,
        ),
        "gpu_peak_reserved_mb": round(
            float(torch.cuda.max_memory_reserved(gpu_index)) / (1024 ** 2),
            2,
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Ida-BD single pre/post image-pair inference and F1 evaluation."
    )

    parser.add_argument("--pre_image", required=True)
    parser.add_argument("--post_image", required=True)
    parser.add_argument("--gt_mask", required=True)

    parser.add_argument("--loc_dir", required=True)
    parser.add_argument("--stage2_dir", required=True)

    parser.add_argument("--building_mask_output", required=True)
    parser.add_argument("--damage_mask_output", required=True)
    parser.add_argument("--damage_index_output", required=True)
    parser.add_argument("--overlay_output", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--summary_csv", required=True)

    parser.add_argument("--loc_glob", default="*_idabd_ft_best.pth")
    parser.add_argument("--stage2_glob", default="*_idabd_stage2_ft_best.pth")
    parser.add_argument("--stage2_max_models", type=int, default=12)
    parser.add_argument("--loc_thresh", type=float, default=0.5)
    parser.add_argument("--min_area", type=int, default=20)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--model_name", default="RGB + Unsharp Model")

    # Must match your RGB + Unsharp experiment.
    parser.add_argument("--unsharp_alpha", type=float, default=0.6)
    parser.add_argument("--unsharp_amount", type=float, default=1.0)
    parser.add_argument("--unsharp_sigma", type=float, default=1.0)

    args = parser.parse_args()

    script_start_time = time.perf_counter()

    # ------------------------------------------------------------------
    # Image loading + preprocessing timer
    # ------------------------------------------------------------------
    preprocess_start_time = time.perf_counter()

    pre_image = cv2.imread(args.pre_image, cv2.IMREAD_COLOR)
    post_image = cv2.imread(args.post_image, cv2.IMREAD_COLOR)

    if pre_image is None:
        raise FileNotFoundError(f"Could not read pre-disaster image: {args.pre_image}")

    if post_image is None:
        raise FileNotFoundError(f"Could not read post-disaster image: {args.post_image}")

    if pre_image.shape[:2] != post_image.shape[:2]:
        print(
            "[WARNING] Pre and post images have different sizes. "
            "Resizing post image to match pre image.",
            flush=True,
        )
        post_image = cv2.resize(
            post_image,
            (pre_image.shape[1], pre_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    original_h, original_w = pre_image.shape[:2]

    # ------------------------------------------------------------------
    # Match the RGB + Unsharp experiment:
    # post_for_model = blend(post_image, unsharp(post_image), alpha)
    # ------------------------------------------------------------------
    post_unsharp = unsharp_bgr(
        post_image,
        amount=args.unsharp_amount,
        sigma=args.unsharp_sigma,
    )

    post_for_model = blend_bgr(
        post_image,
        post_unsharp,
        alpha=args.unsharp_alpha,
    )

    print("[PREPROCESSING] RGB + Unsharp", flush=True)
    print(f"[UNSHARP] alpha={args.unsharp_alpha}", flush=True)
    print(f"[UNSHARP] amount={args.unsharp_amount}", flush=True)
    print(f"[UNSHARP] sigma={args.unsharp_sigma}", flush=True)

    pre_padded, _ = pad_to_factor(pre_image, 32)
    post_padded, _ = pad_to_factor(post_for_model, 32)

    x6 = preprocess_6ch(pre_padded, post_padded).unsqueeze(0).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}", flush=True)

    x6 = x6.to(device)

    preprocess_seconds = time.perf_counter() - preprocess_start_time

    # ------------------------------------------------------------------
    # Model loading timer
    # ------------------------------------------------------------------
    model_load_start_time = time.perf_counter()

    loc_pack, loc_used = load_loc_pack(
        args.loc_dir,
        device=device,
        loc_glob=args.loc_glob,
    )

    stage2_pack, stage2_used = load_stage2_pack(
        args.stage2_dir,
        device=device,
        stage2_glob=args.stage2_glob,
        stage2_max_models=args.stage2_max_models,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    model_load_seconds = time.perf_counter() - model_load_start_time

    print(f"[LOC] Loaded {len(loc_used)} localization models", flush=True)
    for weight in loc_used:
        print(f"  - {weight}", flush=True)

    print(f"[STAGE2] Loaded {len(stage2_used)} damage models", flush=True)
    for weight in stage2_used:
        print(f"  - {weight}", flush=True)

    # ------------------------------------------------------------------
    # Inference timer + GPU peak memory
    # ------------------------------------------------------------------
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    inference_start_time = time.perf_counter()

    with torch.no_grad():
        build_mask = predict_build_mask_ensemble(
            loc_pack,
            x6,
            thresh=args.loc_thresh,
            use_amp=args.amp,
        )

        damage_pred = predict_damage_ensemble_cal(
            stage2_pack,
            x6,
            use_amp=args.amp,
        )

        pred_final = torch.zeros_like(damage_pred)
        pred_final[build_mask] = damage_pred[build_mask]

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_seconds = time.perf_counter() - inference_start_time

    # ------------------------------------------------------------------
    # Postprocessing timer
    # ------------------------------------------------------------------
    postprocess_start_time = time.perf_counter()

    pred_np = pred_final[0].detach().cpu().numpy().astype(np.uint8)
    pred_np = pred_np[:original_h, :original_w]

    gt_mask = normalize_ground_truth_mask(
        mask_path=args.gt_mask,
        target_shape=pred_np.shape[:2],
    )

    damage_metrics = compute_f1_scores(pred_np, gt_mask)
    localization_metrics = compute_localization_metrics(pred_np, gt_mask)

    metrics = {
        **localization_metrics,
        **damage_metrics,
        "metrics_note": "Image-specific F1 calculated using uploaded ground-truth post-disaster mask.",
    }

    building_mask = ((pred_np > 0).astype(np.uint8)) * 255
    color_mask_rgb = colorize_mask(pred_np)

    output_parent = Path(args.building_mask_output).parent
    output_parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(args.building_mask_output), building_mask)

    color_mask_bgr = cv2.cvtColor(color_mask_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(args.damage_mask_output), color_mask_bgr)

    cv2.imwrite(str(args.damage_index_output), pred_np)

    make_overlay(
        post_bgr=post_image,
        color_mask_rgb=color_mask_rgb,
        pred_mask=pred_np,
        overlay_output=args.overlay_output,
    )

    summary = summarize_by_connected_components(
        pred_mask=pred_np,
        min_area=args.min_area,
    )

    summary["metrics"] = metrics

    num_loc_models = len(loc_used)
    num_stage2_models = len(stage2_used)
    num_total_models = num_loc_models + num_stage2_models

    image_h = int(original_h)
    image_w = int(original_w)
    num_pixels = int(image_h * image_w)

    gpu_info = get_gpu_runtime_info(device)

    postprocess_seconds = time.perf_counter() - postprocess_start_time
    total_script_seconds = time.perf_counter() - script_start_time

    summary["model_info"] = {
        "model_name": args.model_name,
        "loc_models_used": num_loc_models,
        "stage2_models_used": num_stage2_models,
        "total_models_used": num_total_models,
        "loc_dir": args.loc_dir,
        "stage2_dir": args.stage2_dir,
        "loc_glob": args.loc_glob,
        "stage2_glob": args.stage2_glob,
        "loc_thresh": args.loc_thresh,
        "stage2_max_models": args.stage2_max_models,
        "preprocessing": "RGB_PLUS_UNSHARP",
        "unsharp_alpha": args.unsharp_alpha,
        "unsharp_amount": args.unsharp_amount,
        "unsharp_sigma": args.unsharp_sigma,
        "gt_mask_unique_values": sorted(np.unique(gt_mask).astype(int).tolist()),
    }

    summary["runtime"] = {
        "model_name": args.model_name,
        "device": str(device),
        "gpu_name": gpu_info["gpu_name"],
        "gpu_peak_allocated_mb": gpu_info["gpu_peak_allocated_mb"],
        "gpu_peak_reserved_mb": gpu_info["gpu_peak_reserved_mb"],
        "image_height": image_h,
        "image_width": image_w,
        "num_pixels": num_pixels,
        "localization_models": num_loc_models,
        "damage_models": num_stage2_models,
        "total_models": num_total_models,
        "preprocess_seconds": round(float(preprocess_seconds), 3),
        "model_load_seconds": round(float(model_load_seconds), 3),
        "model_inference_seconds": round(float(inference_seconds), 3),
        "postprocess_seconds": round(float(postprocess_seconds), 3),
        "total_script_seconds": round(float(total_script_seconds), 3),
        "time_complexity": "O((N_loc + N_damage) × H × W)",
        "space_complexity": "O(H × W × C + model_weights)",
        "complexity_note": (
            "Time and space complexity are theoretical estimates. "
            "Runtime and GPU memory values are measured during this prediction."
        ),
    }

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_csv(args.summary_csv, summary)

    print("[RUNTIME]", flush=True)
    print(json.dumps(summary["runtime"], indent=2), flush=True)

    print("[SUMMARY]", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()