#!/bin/bash -l
#SBATCH --job-name=DL_symbols
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=0-01:00:00 #DD-HH:MM:SS
#SBATCH --array=1-1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module --force purge
module load env/development/2025a
module load lang/Python/3.13.1-GCCcore-14.2.0


source $HOME/deep_learning/DL_env_latest/bin/activate

PROJECT_DIR="$HOME/deep_learning/deepfake_HW"
CONFIG_FILE="$PROJECT_DIR/configs.csv"
DATA_DIR="$PROJECT_DIR/symbols"
RUNS_DIR="$PROJECT_DIR/runs"

mkdir -p "$RUNS_DIR" logs

cd "$PROJECT_DIR"

CONFIG_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_FILE")

IFS=',' read -r RUN_NAME HIDDEN_SIZE LR WEIGHT_DECAY SEQ_MAX_LEN BATCH_SIZE DROPOUT SEED EPOCHS <<< "$CONFIG_LINE"

MODEL_FILE="$RUNS_DIR/${RUN_NAME}.pth"

echo "======================================"
echo "SLURM job id:        ${SLURM_JOB_ID}"
echo "SLURM array task:    ${SLURM_ARRAY_TASK_ID}"
echo "Host:                $(hostname)"
echo "Date:                $(date)"
echo "Project dir:         ${PROJECT_DIR}"
echo "Data dir:            ${DATA_DIR}"
echo "Run name:            ${RUN_NAME}"
echo "Model file:          ${MODEL_FILE}"
echo "Hidden size:         ${HIDDEN_SIZE}"
echo "LR:                  ${LR}"
echo "Weight decay:        ${WEIGHT_DECAY}"
echo "Seq max len:         ${SEQ_MAX_LEN}"
echo "Batch size:          ${BATCH_SIZE}"
echo "Epochs:              ${EPOCHS}"
echo "Dropout:             ${DROPOUT}"
echo "Seed:                ${SEED}"
echo "CUDA available:"
python - <<'PY'
import torch
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
echo "======================================"

srun python -u train.py \
    --data-dir "$DATA_DIR" \
    --model-file "$MODEL_FILE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --seq-max-len "$SEQ_MAX_LEN" \
    --hidden-size "$HIDDEN_SIZE" \
    --patience 15 \
    --min-delta 1e-4 \
    --dropout "$DROPOUT" \
    --seed "$SEED"

echo "Finished run: ${RUN_NAME}"
echo "Saved model: ${MODEL_FILE}"




