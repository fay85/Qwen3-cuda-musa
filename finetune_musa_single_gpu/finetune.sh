#!/usr/bin/env bash
# ============================================================
#  finetune.sh  —  Qwen3-4B LoRA finetuning on NuminaMath
#  MUSA (Moore Threads) single-GPU version.
#  Run on Ubuntu from the script's directory:
#      cd /path/to/Qwen3/finetune
#      bash finetune.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configuration ────────────────────────────────────────────
# Set MODEL_PATH to a local directory if you already have the weights.
# Set DOWNLOAD_MODEL=true to pull Qwen3-4B from ModelScope first.
DOWNLOAD_MODEL=true
MODEL_ID="Qwen/Qwen3-4B"                          # ModelScope model ID (modelscope.cn namespace/name)
MODEL_PATH="./models/qwen3-4b"                    # local weights path

DATASET_DIR="./datasets/numinamath"
OUTPUT_DIR="./output/qwen3-4b-numinamath"

MAX_LENGTH=2048
NUM_EPOCHS=3
BATCH_SIZE=1          # per-device train batch
EVAL_BATCH_SIZE=1     # per-device eval batch
GRAD_ACCUM=16         # effective batch = BATCH_SIZE * GRAD_ACCUM * num_gpus
LEARNING_RATE="1e-4"
LORA_RANK=16
LORA_ALPHA=32

# Greedy-decode N samples per epoch for \boxed{} exact-match accuracy.
# Reduce if accuracy eval is too slow; set to 0 to disable.
EVAL_ACC_SAMPLES=64
EVAL_ACC_BATCH=4

# Optional: cap train set size for quick experiments (comment out for full dataset)
# TRAIN_SAMPLES=5000

# ── Select GPU ────────────────────────────────────────────────
export MUSA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# ── Check dataset ─────────────────────────────────────────────
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory not found: $DATASET_DIR"
    echo "Download with:"
    echo "  modelscope download --dataset AI-ModelScope/NuminaMath-CoT --local_dir $DATASET_DIR"
    exit 1
fi

# ── (Optional) Download model from ModelScope ─────────────────
if [ "$DOWNLOAD_MODEL" = true ]; then
    echo ""
    echo "[Step 1/4] Downloading Qwen3-4B from ModelScope ..."
    mkdir -p "$MODEL_PATH"
    # Use Python snapshot_download — more reliable than the CLI for model repos
    python - <<EOF
from modelscope import snapshot_download
snapshot_download('${MODEL_ID}', local_dir='${MODEL_PATH}')
EOF
else
    echo ""
    echo "[Step 1/4] Skipping model download (DOWNLOAD_MODEL=false)"
    echo "           MODEL_PATH = $MODEL_PATH"
fi

# ── Install Python dependencies ───────────────────────────────
echo ""
echo "[Step 2/4] Installing Python dependencies ..."
python3 -m pip install -r requirements.txt -q

# ── Build training command ────────────────────────────────────
TRAIN_CMD=(
    python3 train_qwen3.py
    --model_path                  "$MODEL_PATH"
    --dataset_dir                 "$DATASET_DIR"
    --output_dir                  "$OUTPUT_DIR"
    --max_length                  "$MAX_LENGTH"
    --num_train_epochs            "$NUM_EPOCHS"
    --per_device_train_batch_size "$BATCH_SIZE"
    --per_device_eval_batch_size  "$EVAL_BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --learning_rate               "$LEARNING_RATE"
    --lora_rank                   "$LORA_RANK"
    --lora_alpha                  "$LORA_ALPHA"
    --eval_accuracy_samples       "$EVAL_ACC_SAMPLES"
    --eval_accuracy_batch         "$EVAL_ACC_BATCH"
)

# Append optional train sample cap if set
if [ -n "${TRAIN_SAMPLES+x}" ]; then
    TRAIN_CMD+=(--train_samples "$TRAIN_SAMPLES")
fi

# ── Train ─────────────────────────────────────────────────────
echo ""
echo "[Step 3/4] Starting finetuning ..."
echo "${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"

# ── Plot curves ───────────────────────────────────────────────
METRICS_FILE="$OUTPUT_DIR/epoch_metrics.json"
if [ -f "$METRICS_FILE" ]; then
    echo ""
    echo "[Step 4/4] Plotting loss & accuracy curves ..."
    python3 plot_curves.py \
        --metrics_file "$METRICS_FILE" \
        --output_dir   "$OUTPUT_DIR/plots"
    echo ""
    echo "Plots saved to: $OUTPUT_DIR/plots"
else
    echo "WARNING: Metrics file not found ($METRICS_FILE) — skipping plot."
fi

echo ""
echo "All done!  Model checkpoint: $OUTPUT_DIR/final"
