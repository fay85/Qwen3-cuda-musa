# Qwen3 Finetuning (CUDA + MUSA)

LoRA finetuning of Qwen3 models on NVIDIA CUDA and Moore Threads MUSA GPUs.

## Projects

| Folder | GPU | Model | Parallelism | Training API |
|--------|-----|-------|-------------|--------------|
| `finetune/` | CUDA | Qwen3-4B / 8B | Single-GPU | HF Trainer |
| `finetune_musa_single_gpu/` | MUSA | Qwen3-4B | Single-GPU | HF Trainer |
| `finetune_musa/` | MUSA | Qwen3-8B | Multi-GPU (FSDP + MCCL) | Megatron-style loop |

## Quick Start

### CUDA (single GPU)

```bash
cd finetune
bash finetune.sh          # Qwen3-4B (default)
bash finetune.sh 8b       # Qwen3-8B
```

### MUSA single GPU

```bash
cd finetune_musa_single_gpu
bash finetune.sh
```

### MUSA multi-GPU

```bash
cd finetune_musa
bash finetune.sh
```

## Dataset

All single-GPU projects use NuminaMath-CoT (~860K math problems). Download with:

```bash
modelscope download --dataset AI-ModelScope/NuminaMath-CoT --local_dir ./datasets/numinamath
```

The multi-GPU project (`finetune_musa/`) uses a general SFT dataset in JSONL format.

## Model Weights

Weights are auto-downloaded from ModelScope when `DOWNLOAD_MODEL=true` (default).
To use local weights, set `DOWNLOAD_MODEL=false` and update `MODEL_PATH` in `finetune.sh`.