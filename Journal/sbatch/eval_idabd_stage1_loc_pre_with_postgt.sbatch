#!/bin/bash
#SBATCH -p wsu_soc_kshirsagar.q
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=idabd_loc_eval
#SBATCH --output=/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/Journal/logs/%x_%j.out
#SBATCH --error=/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution/Journal/logs/%x_%j.err

set -eo pipefail

REPO_DIR="/homes/j244s673/documents/wsu/phd/xview2_1st_place_solution"
JOURNAL_DIR="${REPO_DIR}/Journal"
SCRIPT="${JOURNAL_DIR}/scripts/eval_idabd_stage1_loc_pre_with_postgt.py"

IMG_DIR="/homes/j244s673/documents/wsu/phd/idabd/PRJ-5770/Project--ian-bd-pre-and-post-disaster-high-resolution-aerial-imagery-for-building-damage-assessment-from-hurricane-ian/data/images"
MASK_DIR="/homes/j244s673/documents/wsu/phd/idabd/PRJ-5770/Project--ian-bd-pre-and-post-disaster-high-resolution-aerial-imagery-for-building-damage-assessment-from-hurricane-ian/data/masks"

# Fine-tuned checkpoints from your current training job.
WEIGHTS_DIR="${JOURNAL_DIR}/outputs/idabd_stage1_loc_ft_checkpoints"

OUT_DIR="${JOURNAL_DIR}/outputs/pred_idabd_loc_stage1_ft"

ENV_PY="/homes/j244s673/miniforge3/envs/first_xview2/bin/python"

mkdir -p "${JOURNAL_DIR}/logs" "${JOURNAL_DIR}/outputs" "${OUT_DIR}"

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
echo "OUT_DIR: ${OUT_DIR}"

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

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the Slurm job.")
PY

"${ENV_PY}" -u "${SCRIPT}" \
  --img_dir "${IMG_DIR}" \
  --mask_dir "${MASK_DIR}" \
  --weights_dir "${WEIGHTS_DIR}" \
  --out_dir "${OUT_DIR}" \
  --input_split pre \
  --gt_split post \
  --thresh 0.1 \
  --thresh_sweep \
  --debug \
  --debug_first 5 \
  --debug_every 20

echo "END: $(date)"