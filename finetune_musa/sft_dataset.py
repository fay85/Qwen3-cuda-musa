"""
SFT Dataset for Qwen3 finetuning.

Supports two formats:
  1. OpenAI messages format (JSONL with "messages" field)
  2. Alpaca format (JSONL with "instruction", "input", "output")

Tokenization uses the model's chat template to build input_ids/labels,
masking the prompt portion with IGNORE_INDEX so only the assistant
response contributes to the loss.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_json_or_jsonl(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    text = p.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return load_jsonl(path)


def convert_to_messages(example: Dict) -> Optional[List[Dict]]:
    """Convert various formats to OpenAI messages format."""
    if "messages" in example:
        return example["messages"]
    if "conversations" in example:
        msgs = []
        for turn in example["conversations"]:
            role = turn.get("role") or ("user" if turn.get("from") in ("human", "user") else "assistant")
            content = turn.get("content") or turn.get("value", "")
            msgs.append({"role": role, "content": content})
        return msgs
    if "instruction" in example:
        user_content = example["instruction"]
        if example.get("input"):
            user_content += "\n" + example["input"]
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example.get("output", "")},
        ]
    return None


class SFTDataset(Dataset):
    """
    Supervised Fine-Tuning dataset.

    Each sample is tokenized using the model's chat template.
    The prompt tokens are masked (IGNORE_INDEX) in labels so the loss
    is computed only on the assistant's response.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        raw_data = load_json_or_jsonl(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        skipped = 0
        for example in raw_data:
            messages = convert_to_messages(example)
            if messages is None or len(messages) < 2:
                skipped += 1
                continue

            result = self._tokenize(messages)
            if result is not None:
                self.samples.append(result)
            else:
                skipped += 1

        print(f"[SFTDataset] Loaded {len(self.samples)} samples from {data_path} "
              f"(skipped {skipped})")

    def _tokenize(self, messages: List[Dict]):
        try:
            full_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            prompt_text = self.tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True)

            full_ids = self.tokenizer(
                full_text, add_special_tokens=False,
                truncation=True, max_length=self.max_length)["input_ids"]
            prompt_ids = self.tokenizer(
                prompt_text, add_special_tokens=False)["input_ids"]

            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = [IGNORE_INDEX] * prompt_len + full_ids[prompt_len:]
            labels = labels[:len(full_ids)]

            if len(full_ids) < 4:
                return None

            return {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        except Exception:
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def sft_collate_fn(batch, pad_token_id: int = 0, max_length: int = 2048):
    """Collate function that pads to the longest sequence in the batch."""
    max_len = min(max(len(s["input_ids"]) for s in batch), max_length)
    max_len = ((max_len + 7) // 8) * 8

    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []

    for sample in batch:
        ids = sample["input_ids"][:max_len]
        mask = sample["attention_mask"][:max_len]
        labs = sample["labels"][:max_len]

        pad_len = max_len - len(ids)
        input_ids_batch.append(ids + [pad_token_id] * pad_len)
        attention_mask_batch.append(mask + [0] * pad_len)
        labels_batch.append(labs + [IGNORE_INDEX] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
        "labels": torch.tensor(labels_batch, dtype=torch.long),
    }
