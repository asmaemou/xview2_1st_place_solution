# xView2 → Ida-BD: Two-Stage Building Damage Mapping with Supervised Domain Adaptation

This repository contains code for **pixel-wise building damage mapping on Hurricane Ida (Ida-BD)** using a **two-stage pipeline** adapted from the **xView2 first-place solution**. The key contribution is **supervised domain adaptation (DA)**: we fine-tune xView2-pretrained localization and damage models on Ida-BD to reduce cross-domain performance degradation. We also include an optional **fusion augmentation** to enhance post-disaster imagery.

---

## Overview

### Task
Given a pre-disaster and post-disaster satellite image pair, predict a **5-class mask**:
- `0`: Background (non-building)
- `1`: No Damage
- `2`: Minor Damage
- `3`: Major Damage
- `4`: Destroyed
- `255`: Ignore / unlabeled (excluded from loss and metrics)

### Pipeline Summary

**Stage 1 — Building Localization (binary)**
- Input: **pre-disaster RGB**
- Output: building probability map → thresholded to a **binary building mask**
- Optional calibration: **Platt scaling** (`a, b`) applied to logits:
  \[
  p(\text{building}) = \sigma(a \cdot \text{logit} + b)
  \]

**Stage 2 — Damage Classification (multiclass)**
- Input: **6-channel** tensor `[pre_RGB ; post_RGB]`
- Output: damage logits for classes `{1..4}` (damage only)
- Final output mask:
  - If `building_mask == 0` → output `0`
  - Else → output predicted damage class `1..4`

---

## What is “Supervised Domain Adaptation” here?

Models trained on xView2/xBD often degrade on Ida-BD due to **domain shift** (different region, disaster type, sensor conditions). We address this by **supervised fine-tuning on labeled Ida-BD**, using xView2 weights as initialization:

- Stage 1: xView2 localization ensemble → **fine-tune on Ida-BD** building masks
- Stage 2: xView2 damage ensemble → **fine-tune on Ida-BD** damage masks

This improves robustness and avoids collapse on rare severe classes (especially **Destroyed**).

---

## Ensemble Definition (Where “12 models” comes from)

12 models come from four distinct model architectures, and for each architec-
ture they conducted three independent runs with different
random seeds, resulting in 4 × 3 = 12 trained models
in total. 
During training, we fine-tune **all 12 xView2-initialized Stage-2 damage models** spanning U-Net backbones:



- **DPN92**
- **ResNet34**
- **SE-ResNeXt50**
- **SENet154**

During inference, we ensemble the models by **averaging predictions** (mean over probabilities or logits, depending on the script).

---

## Key Additions Beyond xView2 First-Place Code

### 1) Destroyed-aware crop sampling (Stage 2 training)
Ida-BD has strong class imbalance; **Destroyed** is rare. To avoid learning to ignore class 4, Stage-2 training includes **Destroyed-focused sampling**:
- With probability `focus_destroyed_p`, sample crops that contain at least `min_destroyed_px` pixels of class 4.

### 2) Optional stronger class weighting for Destroyed
We optionally increase the loss weight for class 4:
- `destroyed_weight_mul` multiplies the computed class weight for Destroyed.

### 3) Multiclass calibration via Vector Scaling (Stage 2)
We calibrate Stage-2 logits on validation pixels using **Vector Scaling**:
- We collect validation logits using **class-balanced sampling** (up to `N` pixels per class).
- We learn a linear calibration layer:
  \[
  z' = Wz + b
  \]
- Calibration is applied **AMP-safe** (forced FP32 for matmul).

### 4) Optional Fusion Augmentation (post-disaster enhancement)
We optionally enrich the post-disaster image using a weighted fusion of enhancement cues such as:
- Edge enhancement
- Contrast enhancement
- Unsharp masking

This yields an augmented post image that can improve damage separability under domain shift.

---

## Repository Structure (recommended)

> If your repo still contains many long scripts, this section explains what each group of files does.

- `train_*stage1*`: train / fine-tune building localization networks
- `train_*stage2*`: train / fine-tune damage classification networks
- `eval_*stage2*`: evaluate Stage-2 damage only
- `eval_*pipeline*`: evaluate **end-to-end** pipeline (Stage-1 gating → Stage-2)

Checkpoints and outputs are written into folders such as:
- `idabd_stage1_loc_ft_checkpoints/`
- `idabd_stage2_damage_ft_checkpoints_.../`

---

## Setup

### Requirements
- Python 3.8+
- PyTorch (CUDA recommended)
- OpenCV (`cv2`)
- NumPy

Install typical dependencies:
```bash
pip install -r requirements.txt
