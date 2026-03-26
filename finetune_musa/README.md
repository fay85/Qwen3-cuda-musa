# Qwen3-8B Multi-GPU LoRA SFT on Moore Threads MUSA

Megatron-LM style multi-GPU finetuning of Qwen3-8B using FSDP + MCCL
on Moore Threads S4000 GPUs.

## Architecture

```
torchrun (multi-process)
   |
   +-- initialize_distributed()    # Megatron-style: MCCL backend,
   |                                # model-parallel + data-parallel groups
   +-- AutoModelForCausalLM         # Load Qwen3-8B from HuggingFace
   |
   +-- get_peft_model(LoRA)         # Apply LoRA adapters (rank=64)
   |
   +-- FSDP(model)                  # Shard model across MUSA GPUs
   |
   +-- Training Loop                # Megatron-style: forward_step,
       |                            # gradient accumulation, cosine LR
       +-- forward_step()           # Compute loss on micro-batch
       +-- clip_grad_norm           # Gradient clipping
       +-- optimizer.step()         # AdamW update
       +-- save_checkpoint()        # Periodic LoRA checkpoint saves
```

## Quick Start

1. Prepare SFT data in JSONL format (OpenAI messages or Alpaca format):

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

2. Edit `finetune.sh` to set `MODEL_PATH`, `DATA_PATH`, and `NUM_GPUS`.

3. Launch:

```bash
bash finetune.sh
```

## Files

| File | Purpose |
|------|---------|
| `train_qwen3_8b.py` | Main training entry with Megatron-style loop |
| `megatron_utils.py` | Distributed init, process groups, FSDP utilities |
| `sft_dataset.py` | SFT data loading and tokenization |
| `finetune.sh` | Multi-GPU launch script (torchrun + MCCL) |
| `requirements.txt` | Python dependencies |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_gpus` | 4 | Number of MUSA GPUs |
| `--tensor_parallel_size` | 1 | TP size (1 = FSDP only) |
| `--fsdp_sharding` | FULL_SHARD | FSDP strategy |
| `--batch_size` | 2 | Per-GPU micro batch |
| `--gradient_accumulation_steps` | 8 | Grad accum steps |
| `--lora_rank` | 64 | LoRA rank |
| `--lora_alpha` | 128 | LoRA alpha |
| `--max_length` | 2048 | Max sequence length |
| `--learning_rate` | 2e-5 | Learning rate |
| `--cpu_offload` | off | FSDP CPU offload (saves VRAM) |

## Memory Estimates (Qwen3-8B, bf16, LoRA rank=64)

| GPUs | FSDP Strategy | Per-GPU VRAM | Notes |
|------|--------------|-------------|-------|
| 4 | FULL_SHARD | ~18 GB | Recommended |
| 4 | SHARD_GRAD_OP | ~24 GB | Faster, more memory |
| 2 | FULL_SHARD | ~30 GB | Tight fit |
| 4 | FULL_SHARD + CPU offload | ~12 GB | Slower, minimum VRAM |

## Requirements

- Moore Threads S4000 GPU (MCCL support required)
- torch_musa (with MCCL enabled: `USE_MCCL=1`)
- transformers >= 4.51.0
- peft >= 0.12.0

## Differences from Qwen3.5 Single-GPU Finetune

| Aspect | Qwen3.5 (single-GPU) | This project (multi-GPU) |
|--------|---------------------|--------------------------|
| Model | Qwen3.5-0.8B | Qwen3-8B |
| Parallelism | None | FSDP + MCCL |
| Training API | HF Trainer | Custom Megatron-style loop |
| Device | Single MUSA | Multi-MUSA (torchrun) |
| Efficiency | LoRA (rank 16) | LoRA (rank 64) |
| Communication | N/A | MCCL (all-reduce, reduce-scatter) |
