#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --job-name=idabd_stage2_rgb
#SBATCH --output=/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/Journal/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/Journal/logs/%x_%j.err

set -eo pipefail

REPO_DIR="/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution"
JOURNAL_DIR="${REPO_DIR}/Journal"

SCRIPT="${JOURNAL_DIR}/scripts/train_idabd_stage2_pipeline_RGB_baseline_earlyStop.py"

IMG_DIR="/homes/j244s673/documents/wsu/phd/idabd/PRJ-5770/Project--ian-bd-pre-and-post-disaster-high-resolution-aerial-imagery-for-building-damage-assessment-from-hurricane-ian/data/images"
MASK_DIR="/homes/j244s673/documents/wsu/phd/idabd/PRJ-5770/Project--ian-bd-pre-and-post-disaster-high-resolution-aerial-imagery-for-building-damage-assessment-from-hurricane-ian/data/masks"

# Stage-2 damage classification init weights
WEIGHTS_DIR="${REPO_DIR}/new_approach/weights"

# Stage-1 localization checkpoint to use for pipeline gating
# This one had val F1=0.8057 in your Stage-1 run.
LOC_WEIGHT="${JOURNAL_DIR}/outputs/idabd_stage1_loc_ft_checkpoints/se154_loc_0_1_best_idabd_ft_best.pth"

# I am leaving Platt calibration OFF by default because that same checkpoint had better uncalibrated val F1.
# To use Platt later, set:
# LOC_PLATT="${JOURNAL_DIR}/outputs/idabd_stage1_loc_ft_checkpoints/se154_loc_0_1_best_idabd_platt.npz"
LOC_PLATT=""

LOC_THRESH="0.5"

OUT_DIR="${JOURNAL_DIR}/outputs/idabd_stage2_damage_ft_checkpoints_RGB_baseline_earlyStop"
CSV_PATH="${JOURNAL_DIR}/results/idabd_stage2_pipeline_RGB_baseline_earlyStop.csv"

ENV_PY="/homes/j244s673/miniforge3/envs/first_xview2/bin/python"

mkdir -p "${JOURNAL_DIR}/logs" "${JOURNAL_DIR}/outputs" "${JOURNAL_DIR}/results" "${OUT_DIR}"

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
echo "WEIGHTS_DIR: ${WEIGHTS_DIR}"
echo "LOC_WEIGHT: ${LOC_WEIGHT}"
echo "LOC_PLATT: ${LOC_PLATT:-none}"
echo "LOC_THRESH: ${LOC_THRESH}"
echo "OUT_DIR: ${OUT_DIR}"
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
print("Dpn92_Unet_Double:", hasattr(zm, "Dpn92_Unet_Double"))
print("Res34_Unet_Double:", hasattr(zm, "Res34_Unet_Double"))
print("SeResNext50_Unet_Double:", hasattr(zm, "SeResNext50_Unet_Double"))
print("SeNet154_Unet_Double:", hasattr(zm, "SeNet154_Unet_Double"))
print("Dpn92_Unet_Loc:", hasattr(zm, "Dpn92_Unet_Loc"))
print("Res34_Unet_Loc:", hasattr(zm, "Res34_Unet_Loc"))
print("SeResNext50_Unet_Loc:", hasattr(zm, "SeResNext50_Unet_Loc"))
print("SeNet154_Unet_Loc:", hasattr(zm, "SeNet154_Unet_Loc"))

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the Slurm job.")
PY

LOC_PLATT_ARGS=()
if [ -n "${LOC_PLATT}" ]; then
  LOC_PLATT_ARGS=(--loc_platt "${LOC_PLATT}")
fi

"${ENV_PY}" -u "${SCRIPT}" \
  --img_dir "${IMG_DIR}" \
  --mask_dir "${MASK_DIR}" \
  --weights_dir "${WEIGHTS_DIR}" \
  --out_dir "${OUT_DIR}" \
  --csv_path "${CSV_PATH}" \
  --overwrite_csv \
  --gt_split post \
  --loc_weight "${LOC_WEIGHT}" \
  "${LOC_PLATT_ARGS[@]}" \
  --loc_thresh "${LOC_THRESH}" \
  --epochs 10 \
  --batch 2 \
  --lr 1e-4 \
  --wd 1e-4 \
  --workers 8 \
  --crop 512 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --use_class_weights \
  --destroyed_weight_mul 3.0 \
  --focus_destroyed_p 0.6 \
  --min_destroyed_px 50 \
  --early_stop_patience 15 \
  --early_stop_warmup 5 \
  --early_stop_min_delta 0.0 \
  --calib_pixels_per_class 50000 \
  --calib_epochs 10 \
  --amp

echo "END: $(date)"