#!/usr/bin/env bash
# ============================================================
#  Qwen3-8B Multi-GPU LoRA SFT on Moore Threads MUSA
#  Megatron-LM style training with FSDP + MCCL
#
#  Launch: bash finetune.sh
# ============================================================

set -euo pipefail

# -- Resolve script directory so relative paths always work ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -- Configuration -------------------------------------------
MODEL_PATH="./qwen3-8b-full"        # local path to Qwen3-8B weights
DATA_PATH="./datasets/qwen3-sft/qwen3_32b_distill_1k.jsonl"   # SFT training data
EVAL_DATA_PATH=""                    # optional eval data
OUTPUT_DIR="./output/qwen3-8b-sft"

NUM_GPUS=4                           # number of MUSA GPUs
TENSOR_PARALLEL_SIZE=1               # TP=1 means pure FSDP/DP

BATCH_SIZE=2                         # per-GPU micro batch
GRAD_ACCUM=8                         # gradient accumulation steps
MAX_LENGTH=2048
NUM_EPOCHS=3
LEARNING_RATE="2e-5"

LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.05

FSDP_SHARDING="FULL_SHARD"          # FULL_SHARD | SHARD_GRAD_OP | NO_SHARD
CPU_OFFLOAD=""                       # set to "--cpu_offload" if OOM

SEED=1234
LOG_INTERVAL=10
SAVE_INTERVAL=500
EVAL_INTERVAL=200

# -- Make all MUSA GPUs visible --------------------------------
export MUSA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# -- MUSA coredump management (keep for debugging, cap at 100MB) --
MUDMP_DIR="${OUTPUT_DIR}/mudmp"
MUDMP_MAX_MB=100
mkdir -p "$MUDMP_DIR"
export MUSA_COREDUMP_FILE="${MUDMP_DIR}/core_%p.mudmp"
export MUSA_COREDUMP_PIPE=0
_cleanup_mudmp() {
    [ -d "$MUDMP_DIR" ] || return 0
    local total_kb
    total_kb=$(du -sk "$MUDMP_DIR" 2>/dev/null | awk '{print $1}') || true
    local max_kb=$((MUDMP_MAX_MB * 1024))
    if [ "${total_kb:-0}" -gt "$max_kb" ]; then
        echo "[WARN] mudmp directory (${total_kb}KB) exceeds ${MUDMP_MAX_MB}MB limit, removing oldest dumps..."
        local oldest
        while [ "$(du -sk "$MUDMP_DIR" 2>/dev/null | awk '{print $1}' || echo 0)" -gt "$max_kb" ]; do
            oldest=$(find "$MUDMP_DIR" -name "*.mudmp" -printf '%T+ %p\n' 2>/dev/null | sort | head -1 | awk '{print $2}') || true
            [ -z "$oldest" ] && break
            echo "  Removing: $oldest"
            rm -f "$oldest"
        done
    fi
}
_cleanup_mudmp

# -- Install dependencies ------------------------------------
echo "[Step 1] Installing dependencies ..."
python3 -m pip install -r requirements.txt -q

# -- Create output directory ----------------------------------
mkdir -p "$OUTPUT_DIR"

# -- Build launch command ------------------------------------
TRAIN_ARGS=(
    --model_path                "$MODEL_PATH"
    --data_path                 "$DATA_PATH"
    --output_dir                "$OUTPUT_DIR"
    --max_length                "$MAX_LENGTH"
    --num_epochs                "$NUM_EPOCHS"
    --batch_size                "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --learning_rate             "$LEARNING_RATE"
    --lora_rank                 "$LORA_RANK"
    --lora_alpha                "$LORA_ALPHA"
    --lora_dropout              "$LORA_DROPOUT"
    --num_gpus                  "$NUM_GPUS"
    --tensor_parallel_size      "$TENSOR_PARALLEL_SIZE"
    --fsdp_sharding             "$FSDP_SHARDING"
    --seed                      "$SEED"
    --log_interval              "$LOG_INTERVAL"
    --save_interval             "$SAVE_INTERVAL"
    --eval_interval             "$EVAL_INTERVAL"
    --gradient_checkpointing
    --loss_log                  "$OUTPUT_DIR/loss_log.json"
)

if [ -n "$EVAL_DATA_PATH" ]; then
    TRAIN_ARGS+=(--eval_data_path "$EVAL_DATA_PATH")
fi

if [ -n "$CPU_OFFLOAD" ]; then
    TRAIN_ARGS+=($CPU_OFFLOAD)
fi

# -- Launch multi-GPU training with torchrun ------------------
echo ""
echo "[Step 2] Launching training on $NUM_GPUS MUSA GPUs ..."
echo "  Model:           $MODEL_PATH"
echo "  Data:            $DATA_PATH"
echo "  Effective batch: $((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM))"
echo "  LoRA rank:       $LORA_RANK"
echo "  FSDP:            $FSDP_SHARDING"
echo ""

python3 -m torch.distributed.run \
    --nproc_per_node="$NUM_GPUS" \
    --master_addr=localhost \
    --master_port=29500 \
    train_qwen3_8b.py "${TRAIN_ARGS[@]}"

_cleanup_mudmp
echo ""
echo "Done! Checkpoints saved to: $OUTPUT_DIR"
