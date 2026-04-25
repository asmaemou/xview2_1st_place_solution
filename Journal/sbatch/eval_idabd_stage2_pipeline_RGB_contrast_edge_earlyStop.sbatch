#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=idabd_s2_ce_eval
#SBATCH --output=/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/Journal/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/Journal/logs/%x_%j.err

set -eo pipefail

REPO_DIR="/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution"
JOURNAL_DIR="${REPO_DIR}/Journal"

# The script appears to be under new_approach in your screenshot.
SCRIPT="${REPO_DIR}/new_approach/eval_idabd_stage2_pipeline_RGB_contrast_edge_earlyStop.py"

# Fallback if you later copy it to Journal/scripts.
if [ ! -f "${SCRIPT}" ]; then
  SCRIPT="${JOURNAL_DIR}/scripts/eval_idabd_stage2_pipeline_RGB_contrast_edge_earlyStop.py"
fi

IMG_DIR="/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/Journal/idabd/images"
MASK_DIR="/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/Journal/idabd/masks"


LOC_DIR="${JOURNAL_DIR}/outputs/idabd_stage1_loc_ft_checkpoints"
STAGE2_DIR="${JOURNAL_DIR}/idabd_stage2_damage_ft_checkpoints_RGB_CONTRAST_EDGE_ONLY"

CSV_PATH="${JOURNAL_DIR}/results/idabd_stage2_pipeline_RGB_CONTRAST_EDGE_ONLY_EVAL.csv"

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

"${ENV_PY}" - <<'PY'
import sys, torch, numpy, cv2
print("python exe:", sys.executable)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("numpy:", numpy.__version__)
print("cv2:", cv2.__version__)

import zoo.models as zm
print("zoo.models import: OK")
print("Dpn92_Unet_Loc:", hasattr(zm, "Dpn92_Unet_Loc"))
print("Res34_Unet_Loc:", hasattr(zm, "Res34_Unet_Loc"))
print("SeResNext50_Unet_Loc:", hasattr(zm, "SeResNext50_Unet_Loc"))
print("SeNet154_Unet_Loc:", hasattr(zm, "SeNet154_Unet_Loc"))
print("Dpn92_Unet_Double:", hasattr(zm, "Dpn92_Unet_Double"))
print("Res34_Unet_Double:", hasattr(zm, "Res34_Unet_Double"))
print("SeResNext50_Unet_Double:", hasattr(zm, "SeResNext50_Unet_Double"))
print("SeNet154_Unet_Double:", hasattr(zm, "SeNet154_Unet_Double"))

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the Slurm job.")
PY

echo "===== CHECK FILES ====="
echo "Eval script:"
ls -lh "${SCRIPT}"

echo "Stage-1 loc best checkpoints:"
ls -lh "${LOC_DIR}"/*_idabd_ft_best.pth

echo "Stage-2 RGB CONTRAST EDGE best checkpoints:"
ls -lh "${STAGE2_DIR}"/*_idabd_stage2_ft_best.pth

echo "Stage-2 RGB CONTRAST EDGE calibration files:"
ls -lh "${STAGE2_DIR}"/*_idabd_stage2_calib_vector_scaling.npz

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