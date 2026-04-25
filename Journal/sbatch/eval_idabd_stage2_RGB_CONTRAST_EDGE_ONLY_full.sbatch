#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=idabd_ce_full
#SBATCH --output=/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/Journal/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/Journal/logs/%x_%j.err

set -eo pipefail

REPO_DIR="/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution"
JOURNAL_DIR="${REPO_DIR}/Journal"

SCRIPT="${REPO_DIR}/new_approach/eval_idabd_stage2_pipeline_RGB_contrast_edge_earlyStop.py"

IMG_DIR="/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/Journal/idabd/images"
MASK_DIR="/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/Journal/idabd/masks"


LOC_DIR="${JOURNAL_DIR}/idabd_stage1_loc_ft_checkpoints"
STAGE2_DIR="${JOURNAL_DIR}/idabd_stage2_damage_ft_checkpoints_RGB_CONTRAST_EDGE_ONLY"

CSV_PATH="${JOURNAL_DIR}/results/eval_FULL_RGB_CONTRAST_EDGE_ONLY.csv"

ENV_PY="/homes/j244s673/miniforge3/envs/first_xview2/bin/python"

mkdir -p "${JOURNAL_DIR}/logs" "${JOURNAL_DIR}/results"

cd "${REPO_DIR}"

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled

echo "START: $(date)"
echo "HOST:  $(hostname)"
echo "JOB:   ${SLURM_JOB_ID}"
echo "PWD:   $(pwd)"
echo "SCRIPT: ${SCRIPT}"
echo "IMG_DIR: ${IMG_DIR}"
echo "MASK_DIR: ${MASK_DIR}"
echo "LOC_DIR: ${LOC_DIR}"
echo "STAGE2_DIR: ${STAGE2_DIR}"
echo "CSV_PATH: ${CSV_PATH}"

echo "===== ENV CHECK ====="
"${ENV_PY}" -V
nvidia-smi || true

echo "===== CHECK FILES ====="
ls -lh "${SCRIPT}"

echo "Stage-1 localization checkpoints:"
ls -lh "${LOC_DIR}"/*_idabd_ft_best.pth

echo "Stage-2 RGB CONTRAST EDGE best checkpoints:"
ls -lh "${STAGE2_DIR}"/*_idabd_stage2_ft_best.pth

echo "Stage-2 RGB CONTRAST EDGE calibration files:"
ls -lh "${STAGE2_DIR}"/*_idabd_stage2_calib_vector_scaling.npz

echo "Counts:"
echo -n "Stage-1 loc pth count: "
ls "${LOC_DIR}"/*_idabd_ft_best.pth | wc -l

echo -n "Stage-2 best pth count: "
ls "${STAGE2_DIR}"/*_idabd_stage2_ft_best.pth | wc -l

echo -n "Stage-2 calib npz count: "
ls "${STAGE2_DIR}"/*_idabd_stage2_calib_vector_scaling.npz | wc -l

"${ENV_PY}" -u "${SCRIPT}" \
  --img_dir "${IMG_DIR}" \
  --mask_dir "${MASK_DIR}" \
  --gt_split post \
  --loc_dir "${LOC_DIR}" \
  --loc_glob "*_idabd_ft_best.pth" \
  --stage2_dir "${STAGE2_DIR}" \
  --stage2_glob "*_idabd_stage2_ft_best.pth" \
  --stage2_max_models 12 \
  --loc_thresh 0.5 \
  --seed 0 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --workers 8 \
  --csv_path "${CSV_PATH}" \
  --overwrite_csv \
  --debug_destroyed

echo "END: $(date)"