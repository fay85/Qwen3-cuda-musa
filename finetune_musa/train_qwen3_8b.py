"""
Qwen3-8B Multi-GPU LoRA SFT on Moore Threads MUSA
===================================================
Megatron-LM style training with FSDP + MCCL on MUSA GPUs.

This script implements:
  - Megatron-style distributed initialization (process groups, seeding)
  - FSDP model sharding across multiple MUSA GPUs
  - LoRA (PEFT) for parameter-efficient finetuning
  - Custom training loop with gradient accumulation and mixed precision
  - SFT data pipeline with chat-template tokenization

Launch with torchrun:
    torchrun --nproc_per_node=4 train_qwen3_8b.py \\
        --model_path Qwen/Qwen3-8B \\
        --data_path data/sft_train.jsonl \\
        --num_gpus 4

Architecture reference (Qwen3-8B):
    hidden_size=4096, num_layers=36, num_heads=32, num_kv_heads=8,
    intermediate_size=14336, vocab_size=151936, max_position=131072
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path
from functools import partial

import torch
import torch_musa
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

from megatron_utils import (
    initialize_distributed,
    print_rank_0,
    set_random_seed,
    is_rank_0,
    get_global_rank,
    get_local_rank,
    get_world_size,
    barrier,
    destroy_distributed,
    get_fsdp_wrap_policy,
)
from sft_dataset import SFTDataset, sft_collate_fn


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-8B Multi-GPU SFT (MUSA)")

    p.add_argument("--model_path", type=str, default="./qwen3-8b-full",
                   help="Local path or HuggingFace model ID")
    p.add_argument("--data_path", type=str, default="./datasets/qwen3-sft/qwen3_32b_distill_1k.jsonl",
                   help="Path to SFT training data (JSONL or JSON)")
    p.add_argument("--eval_data_path", type=str, default=None,
                   help="Optional eval data path")
    p.add_argument("--output_dir", type=str, default="./output/qwen3-8b-sft")

    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2,
                   help="Per-device micro batch size")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])

    p.add_argument("--num_gpus", type=int, default=4)
    p.add_argument("--tensor_parallel_size", type=int, default=1,
                   help="TP size (1 = pure FSDP/DP, no TP)")
    p.add_argument("--fsdp_sharding", type=str, default="FULL_SHARD",
                   choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"],
                   help="FSDP sharding strategy")
    p.add_argument("--cpu_offload", action="store_true",
                   help="Offload FSDP parameters to CPU (saves GPU memory)")

    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)
    p.add_argument("--loss_log", type=str, default=None)

    return p.parse_args()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Megatron-style cosine LR schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_and_prepare_model(args):
    """Load Qwen3-8B, apply LoRA, prepare for FSDP."""
    print_rank_0(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print_rank_0(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print_rank_0("Gradient checkpointing enabled")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16)
    if is_rank_0():
        model.print_trainable_parameters()

    return model, tokenizer, lora_config


def wrap_model_with_fsdp(model, args):
    """Wrap model with FSDP for multi-GPU sharding."""
    sharding_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }

    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    cpu_offload = CPUOffload(offload_params=True) if args.cpu_offload else None
    wrap_policy = get_fsdp_wrap_policy(model)

    model = FSDP(
        model,
        sharding_strategy=sharding_map[args.fsdp_sharding],
        mixed_precision=bf16_policy,
        auto_wrap_policy=wrap_policy,
        cpu_offload=cpu_offload,
        device_id=torch.device("musa", get_local_rank()),
        limit_all_gathers=True,
        use_orig_params=True,
    )

    print_rank_0(f"FSDP wrapped: strategy={args.fsdp_sharding}, "
                 f"cpu_offload={args.cpu_offload}")
    return model


def create_dataloaders(args, tokenizer):
    """Create distributed-aware train (and optional eval) dataloaders."""
    train_dataset = SFTDataset(args.data_path, tokenizer, args.max_length)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        shuffle=True,
        seed=args.seed,
    )

    collate = partial(
        sft_collate_fn,
        pad_token_id=tokenizer.pad_token_id,
        max_length=args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    eval_loader = None
    if args.eval_data_path:
        eval_dataset = SFTDataset(args.eval_data_path, tokenizer, args.max_length)
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=get_world_size(),
            rank=get_global_rank(),
            shuffle=False,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            sampler=eval_sampler,
            collate_fn=collate,
            num_workers=2,
            pin_memory=True,
        )

    return train_loader, eval_loader


def forward_step(model, batch, device):
    """
    Megatron-style forward step.
    Returns (loss, token_accuracy) for this micro-batch.
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    token_acc = 0.0
    try:
        with torch.no_grad():
            shift_preds = outputs.logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            mask = shift_labels != -100
            if mask.sum() > 0:
                token_acc = (shift_preds == shift_labels)[mask].float().mean().item()
            del shift_preds, shift_labels, mask
    except RuntimeError:
        pass

    return outputs.loss, token_acc


def evaluate(model, eval_loader, device):
    """Run evaluation and return (avg_loss, token_accuracy, perplexity)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_steps = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            total_steps += 1

            try:
                shift_preds = outputs.logits[..., :-1, :].argmax(dim=-1)
                shift_labels = labels[..., 1:]
                mask = shift_labels != -100
                total_correct += (shift_preds == shift_labels)[mask].sum().item()
                total_tokens += mask.sum().item()
                del shift_preds, shift_labels, mask
            except (RuntimeError,):
                pass

    if dist.is_initialized():
        stats = torch.tensor([total_loss, total_steps, total_correct, total_tokens],
                             device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = stats[0].item()
        total_steps = int(stats[1].item())
        total_correct = int(stats[2].item())
        total_tokens = int(stats[3].item())

    model.train()
    avg_loss = total_loss / max(total_steps, 1)
    accuracy = total_correct / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))
    return avg_loss, accuracy, perplexity


def save_checkpoint(model, tokenizer, lora_config, output_dir, step):
    """Save LoRA adapter checkpoint (rank 0 only)."""
    save_dir = os.path.join(output_dir, f"checkpoint-{step}")
    if is_rank_0():
        os.makedirs(save_dir, exist_ok=True)

    barrier()

    if isinstance(model, FSDP):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
            state_dict = model.state_dict()
    else:
        state_dict = model.state_dict()

    if is_rank_0():
        lora_state = {k: v for k, v in state_dict.items() if "lora_" in k}
        torch.save(lora_state, os.path.join(save_dir, "adapter_model.bin"))
        lora_config.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print_rank_0(f"Checkpoint saved: {save_dir} ({len(lora_state)} LoRA tensors)")

    barrier()


def train(args):
    """
    Main training function -- Megatron-LM style training loop.

    Structure follows Megatron conventions:
      1. Initialize distributed (process groups, seeds)
      2. Build model, optimizer, scheduler
      3. Build data iterators
      4. Training loop with gradient accumulation
      5. Periodic eval and checkpointing
    """
    # --- 1. Initialize distributed ---
    if args.tensor_parallel_size != 1:
        raise ValueError(
            "tensor_parallel_size > 1 is not supported. This project uses FSDP for "
            "model sharding; TP groups would be created but unused. "
            "Set --tensor_parallel_size 1 and use --fsdp_sharding instead."
        )

    initialize_distributed(
        tensor_model_parallel_size=args.tensor_parallel_size,
        backend="mccl",
    )
    set_random_seed(args.seed)
    device = torch.device("musa", get_local_rank())

    # --- 2. Build model ---
    model, tokenizer, lora_config = load_and_prepare_model(args)
    model = wrap_model_with_fsdp(model, args)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print_rank_0(f"Parameters: {total_params:,} total, {trainable_params:,} trainable "
                 f"({100*trainable_params/total_params:.2f}%)")

    # --- 3. Build optimizer and scheduler ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )

    train_loader, eval_loader = create_dataloaders(args, tokenizer)

    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print_rank_0(f"Training config:")
    print_rank_0(f"  Epochs:             {args.num_epochs}")
    print_rank_0(f"  Batch/GPU:          {args.batch_size}")
    print_rank_0(f"  Grad accum:         {args.gradient_accumulation_steps}")
    print_rank_0(f"  Effective batch:    {args.batch_size * get_world_size() * args.gradient_accumulation_steps}")
    print_rank_0(f"  Steps/epoch:        {steps_per_epoch}")
    print_rank_0(f"  Total steps:        {total_steps}")
    print_rank_0(f"  Warmup steps:       {warmup_steps}")
    print_rank_0(f"  LR:                 {args.learning_rate}")
    print_rank_0(f"  LoRA rank:          {args.lora_rank}")
    print_rank_0(f"  Max length:         {args.max_length}")
    print_rank_0(f"  GPUs:               {get_world_size()}")
    print_rank_0(f"  FSDP sharding:      {args.fsdp_sharding}")
    print_rank_0("=" * 70)

    # --- 4. Training loop (Megatron-style) ---
    os.makedirs(args.output_dir, exist_ok=True)
    loss_log = []
    global_step = 0
    model.train()

    for epoch in range(args.num_epochs):
        train_loader.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_acc = 0.0
        micro_step = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            loss, token_acc = forward_step(model, batch, device)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            micro_step += 1

            step_loss = loss.item() * args.gradient_accumulation_steps
            epoch_loss += step_loss
            epoch_acc += token_acc

            if micro_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_interval == 0:
                    avg_loss = epoch_loss / micro_step
                    avg_acc = epoch_acc / micro_step
                    ppl = math.exp(min(avg_loss, 100))
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    samples_per_sec = (micro_step * args.batch_size * get_world_size()) / elapsed
                    print_rank_0(
                        f"Epoch {epoch+1}/{args.num_epochs} | "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {step_loss:.4f} | Avg: {avg_loss:.4f} | "
                        f"PPL: {ppl:.2f} | Acc: {avg_acc:.4f} | "
                        f"LR: {lr:.2e} | {samples_per_sec:.0f} samples/s"
                    )
                    loss_log.append({
                        "type": "train",
                        "epoch": epoch + 1,
                        "step": global_step,
                        "loss": round(step_loss, 6),
                        "avg_loss": round(avg_loss, 6),
                        "perplexity": round(ppl, 4),
                        "accuracy": round(avg_acc, 6),
                        "lr": lr,
                        "samples_per_sec": round(samples_per_sec, 1),
                    })

                if eval_loader and global_step % args.eval_interval == 0:
                    eval_loss, eval_acc, eval_ppl = evaluate(model, eval_loader, device)
                    print_rank_0(
                        f"  [Eval] step={global_step} loss={eval_loss:.4f} "
                        f"ppl={eval_ppl:.2f} acc={eval_acc:.4f}"
                    )
                    loss_log.append({
                        "type": "eval",
                        "epoch": epoch + 1,
                        "step": global_step,
                        "eval_loss": round(eval_loss, 6),
                        "eval_perplexity": round(eval_ppl, 4),
                        "eval_accuracy": round(eval_acc, 6),
                    })

                if global_step % args.save_interval == 0:
                    save_checkpoint(model, tokenizer, lora_config, args.output_dir, global_step)

        epoch_avg = epoch_loss / max(micro_step, 1)
        epoch_avg_acc = epoch_acc / max(micro_step, 1)
        epoch_ppl = math.exp(min(epoch_avg, 100))
        elapsed = time.time() - t0
        print_rank_0(
            f"--- Epoch {epoch+1} done | Steps: {global_step} | "
            f"Loss: {epoch_avg:.4f} | PPL: {epoch_ppl:.2f} | "
            f"Acc: {epoch_avg_acc:.4f} | Time: {elapsed:.1f}s ---"
        )
        loss_log.append({
            "type": "epoch_end",
            "epoch": epoch + 1,
            "step": global_step,
            "avg_loss": round(epoch_avg, 6),
            "perplexity": round(epoch_ppl, 4),
            "accuracy": round(epoch_avg_acc, 6),
            "time_seconds": round(elapsed, 1),
        })

        if eval_loader:
            eval_loss, eval_acc, eval_ppl = evaluate(model, eval_loader, device)
            print_rank_0(
                f"  [Eval] epoch={epoch+1} loss={eval_loss:.4f} "
                f"ppl={eval_ppl:.2f} acc={eval_acc:.4f}"
            )
            loss_log.append({
                "type": "eval",
                "epoch": epoch + 1,
                "step": global_step,
                "eval_loss": round(eval_loss, 6),
                "eval_perplexity": round(eval_ppl, 4),
                "eval_accuracy": round(eval_acc, 6),
            })

    # --- 5. Final save ---
    save_checkpoint(model, tokenizer, lora_config, args.output_dir, global_step)

    if is_rank_0() and args.loss_log:
        with open(args.loss_log, "w") as f:
            json.dump(loss_log, f, indent=2)
        print_rank_0(f"Loss log saved to {args.loss_log}")

    print_rank_0("Training complete.")
    destroy_distributed()


if __name__ == "__main__":
    args = parse_args()
    train(args)
