"""
Qwen3 LoRA finetuning on NuminaMath-CoT (local parquet files).

Logs training loss every `--logging_steps` optimizer steps; eval loss still runs
each epoch (`eval_strategy=epoch`). Per-epoch summary: mean train loss over that
epoch, eval_loss, eval_accuracy (boxed-answer exact match).

Metrics: {output_dir}/epoch_metrics.json (per epoch),
         {output_dir}/step_loss.jsonl (one line per training log with loss).
"""

import os
import json
import re
import math
import argparse
from pathlib import Path
from typing import Optional

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",  type=str, default="./models/qwen3-3b",
                   help="Local model path or HuggingFace/ModelScope model ID")
    p.add_argument("--dataset_dir", type=str, default="./datasets/numinamath")
    p.add_argument("--output_dir",  type=str, default="./output/qwen3-3b-numinamath")
    p.add_argument("--max_length",  type=int, default=2048)
    p.add_argument("--num_train_epochs",            type=int,   default=3)
    p.add_argument("--per_device_train_batch_size", type=int,   default=1)
    p.add_argument("--per_device_eval_batch_size",  type=int,   default=1)
    p.add_argument("--gradient_accumulation_steps", type=int,   default=16)
    p.add_argument("--learning_rate",  type=float, default=1e-4)
    p.add_argument("--lora_rank",      type=int,   default=16)
    p.add_argument("--lora_alpha",     type=int,   default=32)
    # Accuracy eval: greedy-decode N samples per epoch and check \boxed{} answer
    p.add_argument("--eval_accuracy_samples", type=int, default=64,
                   help="Samples to greedy-decode per epoch for accuracy. 0 = skip.")
    p.add_argument("--eval_accuracy_batch",   type=int, default=4)
    p.add_argument("--train_samples", type=int, default=None,
                   help="Cap training set size (None = full dataset)")
    p.add_argument("--logging_steps", type=int, default=100,
                   help="Log training loss every N optimizer steps (0 = disable step logging)")
    p.add_argument(
        "--map_num_proc",
        type=int,
        default=1,
        help="datasets.map() worker processes for tokenisation. Use 1 to limit host RAM (large parquet + many workers can OOM the machine).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_numinamath(dataset_dir: str):
    """
    Recursively locate all .parquet files under dataset_dir.
    Files with 'train' in the name -> train split.
    Files with 'test' or 'valid' in the name -> eval split.
    Falls back to a 98/2 random split if no test files are found.
    """
    base = Path(dataset_dir)
    all_pq = sorted(base.rglob("*.parquet"))
    if not all_pq:
        raise FileNotFoundError(f"No .parquet files found under {dataset_dir}")

    print(f"Parquet files found ({len(all_pq)}):")
    for f in all_pq:
        print(f"  {f.relative_to(base)}")

    train_files = [f for f in all_pq if "train" in f.stem.lower()]
    test_files  = [f for f in all_pq if any(k in f.stem.lower() for k in ("test", "valid"))]

    if not train_files:
        # No naming hint — use all files as train source
        train_files = all_pq

    train_df = pd.concat([pd.read_parquet(f) for f in train_files], ignore_index=True)

    if test_files:
        eval_df = pd.concat([pd.read_parquet(f) for f in test_files], ignore_index=True)
    else:
        split = int(len(train_df) * 0.98)
        train_df, eval_df = train_df.iloc[:split], train_df.iloc[split:]

    print(f"Train: {len(train_df)} rows  |  Eval: {len(eval_df)} rows")
    return Dataset.from_pandas(train_df.reset_index(drop=True)), \
           Dataset.from_pandas(eval_df.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def build_messages(example):
    """Return (messages, gt_answer_or_None) from whatever format the row uses."""
    if "messages" in example and example["messages"]:
        msgs = example["messages"]
        gt = extract_boxed(msgs[-1]["content"] if msgs else "")
        return msgs, gt
    if "problem" in example:
        solution = example.get("solution", "")
        msgs = [
            {"role": "user",      "content": example["problem"]},
            {"role": "assistant", "content": solution},
        ]
        return msgs, extract_boxed(solution)
    return None, None


def tokenize_example(example, tokenizer, max_length: int):
    messages, _ = build_messages(example)
    if messages is None:
        return {"input_ids": [], "attention_mask": [], "labels": [], "valid": False}

    try:
        # Full sequence: compute loss on assistant turn only
        full_text   = tokenizer.apply_chat_template(messages,    tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

        full_ids   = tokenizer(full_text,   add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [IGNORE_INDEX] * prompt_len + full_ids[prompt_len:]

        return {
            "input_ids":      full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels":         labels[:len(full_ids)],
            "valid":          True,
        }
    except Exception as e:
        return {"input_ids": [], "attention_mask": [], "labels": [], "valid": False}


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")

def extract_boxed(text: str) -> Optional[str]:
    """Return the last \\boxed{...} content in text, or None."""
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


# ---------------------------------------------------------------------------
# Per-epoch accuracy callback
# ---------------------------------------------------------------------------

class StepLossJsonlCallback(TrainerCallback):
    """Append one JSON object per training log that includes `loss` (for step-wise curves)."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self._path = os.path.join(output_dir, "step_loss.jsonl")

    def on_train_begin(self, args, state, control, **kwargs):
        os.makedirs(self.output_dir, exist_ok=True)
        open(self._path, "w", encoding="utf-8").close()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        row = {
            "step": state.global_step,
            "epoch": logs.get("epoch"),
            "loss": float(logs["loss"]),
            "learning_rate": logs.get("learning_rate"),
        }
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


class EpochMetricsCallback(TrainerCallback):
    """
    After every epoch:
      1. Mean train_loss over step logs collected in that epoch (see on_log).
      2. Latest eval_loss from Trainer's log history (after epoch eval).
      3. Greedy-decodes `eval_accuracy_samples` eval examples in batches.
      4. Compares extracted \\boxed{} answers (exact string match).
      5. Appends entry to {output_dir}/epoch_metrics.json.
    """

    def __init__(self, eval_dataset_raw, tokenizer, output_dir,
                 eval_accuracy_samples=64, eval_batch_size=4):
        self.tokenizer   = tokenizer
        self.output_dir  = output_dir
        self.eval_batch  = eval_batch_size
        self.metrics_log = []
        self._epoch_train_losses: list[float] = []

        n = min(eval_accuracy_samples, len(eval_dataset_raw))
        self.skip_accuracy = (n == 0)

        if not self.skip_accuracy:
            # Pre-compute prompts and ground-truth answers once
            subset = eval_dataset_raw.select(range(n))
            self.prompts    = []
            self.gt_answers = []
            for ex in subset:
                msgs, gt = build_messages(ex)
                if gt is None or msgs is None:
                    continue
                prompt_text = tokenizer.apply_chat_template(
                    msgs[:-1], tokenize=False, add_generation_prompt=True
                )
                self.prompts.append(prompt_text)
                self.gt_answers.append(gt)
            print(f"[Callback] {len(self.prompts)} eval samples with \\boxed{{}} answers prepared.")

    # ------------------------------------------------------------------

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs and "eval_loss" not in logs:
            self._epoch_train_losses.append(float(logs["loss"]))

    # ------------------------------------------------------------------

    def _accuracy(self, model) -> tuple[float, int]:
        tokenizer = self.tokenizer
        orig_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"   # required for batched causal generation

        model.eval()
        device = next(model.parameters()).device
        correct = 0
        total   = len(self.prompts)

        for i in range(0, total, self.eval_batch):
            batch_prompts = self.prompts[i : i + self.eval_batch]
            batch_gts     = self.gt_answers[i : i + self.eval_batch]

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=192,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            in_len = inputs["input_ids"].shape[1]
            for seq, gt in zip(out, batch_gts):
                generated = tokenizer.decode(seq[in_len:], skip_special_tokens=True)
                pred = extract_boxed(generated)
                if pred is not None and pred.strip() == gt.strip():
                    correct += 1

        tokenizer.padding_side = orig_padding_side
        model.train()
        return (correct / total if total > 0 else 0.0), total

    # ------------------------------------------------------------------

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl,
                     model=None, **kwargs):
        epoch = round(state.epoch)

        if self._epoch_train_losses:
            train_loss = sum(self._epoch_train_losses) / len(self._epoch_train_losses)
            self._epoch_train_losses.clear()
        else:
            train_loss = None

        eval_loss = None
        for log in reversed(state.log_history):
            if "eval_loss" in log:
                eval_loss = log["eval_loss"]
                break

        accuracy, n_evaluated = (0.0, 0)
        if not self.skip_accuracy and model is not None:
            print(f"\n[Epoch {epoch}] Running accuracy eval on {len(self.prompts)} samples ...")
            accuracy, n_evaluated = self._accuracy(model)

        entry = {
            "epoch":               epoch,
            "train_loss":          round(train_loss, 6) if train_loss else None,
            "eval_loss":           round(eval_loss,  6) if eval_loss  else None,
            "eval_accuracy":       round(accuracy,   6),
            "eval_samples_scored": n_evaluated,
        }
        self.metrics_log.append(entry)

        os.makedirs(self.output_dir, exist_ok=True)
        metrics_path = os.path.join(self.output_dir, "epoch_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_log, f, indent=2)

        tl = f"{train_loss:.4f}" if train_loss is not None else "n/a"
        el = f"{eval_loss:.4f}" if eval_loss is not None else "n/a"
        print(
            f"[Epoch {epoch}] "
            f"train_loss={tl}  "
            f"eval_loss={el}  "
            f"eval_accuracy={accuracy:.4f} ({int(accuracy*n_evaluated)}/{n_evaluated})  "
            f"→ {metrics_path}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _cuda_bf16_supported() -> bool:
    """Ampere+ generally supports BF16; V100 / Pascal do not — use FP16 there."""
    if not torch.cuda.is_available():
        return False
    fn = getattr(torch.cuda, "is_bf16_supported", None)
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and _cuda_bf16_supported()
    use_fp16 = use_cuda and not use_bf16
    model_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_cuda else torch.float32)
    print(
        f"Mixed precision: bf16_training={use_bf16}, fp16_training={use_fp16}, "
        f"model_dtype={model_dtype} (V100 and older GPUs should use fp16, not bf16)"
    )

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False   # required when using gradient checkpointing

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    # PEFT + gradient checkpointing: ensure non-LoRA inputs get grads
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ── Dataset ────────────────────────────────────────────────────────────
    train_raw, eval_raw = load_numinamath(args.dataset_dir)

    if args.train_samples is not None:
        n = min(args.train_samples, len(train_raw))
        train_raw = train_raw.select(range(n))
        print(f"Train capped at {n} samples")

    def preprocess(ex):
        return tokenize_example(ex, tokenizer, args.max_length)

    _mp = max(1, args.map_num_proc)
    print("Tokenising train split ...")
    train_ds = train_raw.map(
        preprocess, remove_columns=train_raw.column_names, num_proc=_mp, desc="train"
    )
    train_ds = train_ds.filter(lambda x: x["valid"] and len(x["input_ids"]) > 10)
    train_ds = train_ds.remove_columns(["valid"])

    print("Tokenising eval split ...")
    eval_ds = eval_raw.map(
        preprocess, remove_columns=eval_raw.column_names, num_proc=_mp, desc="eval"
    )
    eval_ds = eval_ds.filter(lambda x: x["valid"] and len(x["input_ids"]) > 10)
    eval_ds = eval_ds.remove_columns(["valid"])

    print(f"After tokenisation — train: {len(train_ds)}  eval: {len(eval_ds)}")

    # ── Steps per epoch (optimizer steps = one loss backward+accum cycle) ──
    # Micro-batches per epoch ≈ ceil(N / (per_device_bs * world_size)).
    # One optimizer step every gradient_accumulation_steps micro-batches.
    try:
        _ws = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        _ws = 1
    _n = len(train_ds)
    _bs = args.per_device_train_batch_size
    _ga = args.gradient_accumulation_steps
    _micro = math.ceil(_n / max(1, _bs * _ws))
    _opt_per_epoch = math.ceil(_micro / max(1, _ga))
    print(
        f"Steps per epoch (approx., {_ws} process(es)): "
        f"micro-batches ≈ {_micro}, optimizer steps ≈ {_opt_per_epoch} "
        f"(N={_n}, per_device_bs={_bs}, grad_accum={_ga})"
    )

    # ── Training arguments ─────────────────────────────────────────────────
    _log_steps = args.logging_steps
    if _log_steps <= 0:
        raise ValueError("--logging_steps must be positive, or omit to use default 100")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_strategy="steps",
        logging_steps=_log_steps,
        logging_first_step=True,
        eval_strategy="epoch",       # full eval (eval_loss) every epoch
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    callback = EpochMetricsCallback(
        eval_dataset_raw=eval_raw,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        eval_accuracy_samples=args.eval_accuracy_samples,
        eval_batch_size=args.eval_accuracy_batch,
    )
    step_loss_cb = StepLossJsonlCallback(output_dir=args.output_dir)

    # 'tokenizer' was renamed to 'processing_class' in transformers >= 4.46
    import transformers as _tv
    _trainer_tok_kwarg = (
        "processing_class"
        if tuple(int(x) for x in _tv.__version__.split(".")[:2]) >= (4, 46)
        else "tokenizer"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        **{_trainer_tok_kwarg: tokenizer},
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=IGNORE_INDEX,
            pad_to_multiple_of=8,
        ),
        callbacks=[step_loss_cb, callback],
    )

    # Refine steps/epoch using the real dataloader (matches Trainer internals).
    _dl_len = len(trainer.get_train_dataloader())
    _opt_exact = math.ceil(_dl_len / max(1, args.gradient_accumulation_steps))
    print(
        f"Trainer dataloader: len(train_dataloader)={_dl_len} → "
        f"optimizer steps per epoch ≈ {_opt_exact}"
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print("\nStarting training ...")
    trainer.train()

    # ── Save ───────────────────────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved to {final_dir}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n=== Per-epoch summary ===")
    for e in callback.metrics_log:
        print(f"  Epoch {e['epoch']:2d} | "
              f"train_loss={e['train_loss']}  "
              f"eval_loss={e['eval_loss']}  "
              f"accuracy={e['eval_accuracy']:.4f}")
    print(f"\nMetrics file: {os.path.join(args.output_dir, 'epoch_metrics.json')}")
    print(f"Step-wise loss (every {args.logging_steps} optimizer steps): "
          f"{os.path.join(args.output_dir, 'step_loss.jsonl')}")


if __name__ == "__main__":
    main()
