#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --job-name=idabd_ue_res50_0
#SBATCH --output=/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/Journal/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/Journal/logs/%x_%j.err

set -eo pipefail

REPO_DIR="/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution"
JOURNAL_DIR="${REPO_DIR}/Journal"

SCRIPT="${JOURNAL_DIR}/scripts/train_idabd_stage2_pipeline_RGB_unsharp_edge_earlyStop.py"

IMG_DIR="/homes/j244s673/documents/wsu/phd/idabd/PRJ-5770/Project--ian-bd-pre-and-post-disaster-high-resolution-aerial-imagery-for-building-damage-assessment-from-hurricane-ian/data/images"
MASK_DIR="/homes/j244s673/documents/wsu/phd/idabd/PRJ-5770/Project--ian-bd-pre-and-post-disaster-high-resolution-aerial-imagery-for-building-damage-assessment-from-hurricane-ian/data/masks"

WEIGHTS_DIR="${REPO_DIR}/new_approach/weights"
INIT_WEIGHT="${WEIGHTS_DIR}/res50_cls_cce_0_tuned_best.pth"

LOC_WEIGHT="${JOURNAL_DIR}/outputs/idabd_stage1_loc_ft_checkpoints/se154_loc_0_1_best_idabd_ft_best.pth"

OUT_DIR="${JOURNAL_DIR}/outputs/idabd_stage2_damage_ft_checkpoints_RGB_UNSHARP_EDGE_ONLY"
CSV_PATH="${JOURNAL_DIR}/results/idabd_stage2_pipeline_results_RGB_UNSHARP_EDGE_ONLY.csv"

ENV_PY="/homes/j244s673/miniforge3/envs/first_xview2/bin/python"

mkdir -p "${JOURNAL_DIR}/logs" "${JOURNAL_DIR}/results" "${OUT_DIR}"

cd "${REPO_DIR}"

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled

echo "START: $(date)"
echo "HOST:  $(hostname)"
echo "JOB:   ${SLURM_JOB_ID}"
echo "SCRIPT: ${SCRIPT}"
echo "INIT_WEIGHT: ${INIT_WEIGHT}"
echo "OUT_DIR: ${OUT_DIR}"
echo "CSV_PATH: ${CSV_PATH}"

"${ENV_PY}" -V
nvidia-smi || true

"${ENV_PY}" -u "${SCRIPT}" \
  --img_dir "${IMG_DIR}" \
  --mask_dir "${MASK_DIR}" \
  --weights_dir "${WEIGHTS_DIR}" \
  --gt_split post \
  --init_weight "${INIT_WEIGHT}" \
  --loc_weight "${LOC_WEIGHT}" \
  --loc_thresh 0.5 \
  --out_dir "${OUT_DIR}" \
  --csv_path "${CSV_PATH}" \
  --seed 0 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --epochs 10 \
  --batch 2 \
  --lr 1e-4 \
  --wd 1e-4 \
  --workers 8 \
  --crop 512 \
  --use_class_weights \
  --destroyed_weight_mul 3.0 \
  --focus_destroyed_p 0.6 \
  --min_destroyed_px 50 \
  --early_stop_patience 15 \
  --early_stop_warmup 5 \
  --early_stop_min_delta 0.0 \
  --calib_pixels_per_class 50000 \
  --calib_epochs 10

echo "END: $(date)"