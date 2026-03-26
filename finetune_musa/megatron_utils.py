"""
Megatron-LM style distributed initialization for Moore Threads MUSA.

Provides:
  - initialize_distributed(): sets up process groups with MCCL backend
  - get_model_parallel_group / get_data_parallel_group
  - print_rank_0(): only prints on rank 0
  - set_random_seed(): deterministic seeding across ranks
  - FSDP wrapping utilities for Qwen3 model sharding

This follows Megatron-LM conventions (megatron.core.parallel_state) but
uses PyTorch-native APIs compatible with MUSA via torch_musa.
"""

import os
import random
import datetime
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch_musa

_MODEL_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP = None
_LOCAL_RANK = 0
_GLOBAL_RANK = 0
_WORLD_SIZE = 1
_IS_INITIALIZED = False


def initialize_distributed(
    tensor_model_parallel_size: int = 1,
    backend: str = "mccl",
    timeout_minutes: int = 30,
):
    """
    Megatron-style distributed initialization.

    Sets up:
      - torch.distributed process group with MCCL backend
      - Model-parallel and data-parallel sub-groups
      - MUSA device assignment based on local rank

    Args:
        tensor_model_parallel_size: number of GPUs for tensor parallelism
        backend: distributed backend ("mccl" for MUSA, "nccl" for CUDA)
        timeout_minutes: timeout for init_process_group
    """
    global _MODEL_PARALLEL_GROUP, _DATA_PARALLEL_GROUP
    global _LOCAL_RANK, _GLOBAL_RANK, _WORLD_SIZE, _IS_INITIALIZED

    if _IS_INITIALIZED:
        return

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(minutes=timeout_minutes),
        )

    _GLOBAL_RANK = dist.get_rank()
    _WORLD_SIZE = dist.get_world_size()
    _LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

    torch.musa.set_device(_LOCAL_RANK)

    assert _WORLD_SIZE % tensor_model_parallel_size == 0, \
        f"world_size ({_WORLD_SIZE}) must be divisible by TP size ({tensor_model_parallel_size})"
    data_parallel_size = _WORLD_SIZE // tensor_model_parallel_size

    for dp_idx in range(data_parallel_size):
        tp_ranks = list(range(
            dp_idx * tensor_model_parallel_size,
            (dp_idx + 1) * tensor_model_parallel_size))
        group = dist.new_group(tp_ranks)
        if _GLOBAL_RANK in tp_ranks:
            _MODEL_PARALLEL_GROUP = group

    for tp_idx in range(tensor_model_parallel_size):
        dp_ranks = list(range(tp_idx, _WORLD_SIZE, tensor_model_parallel_size))
        group = dist.new_group(dp_ranks)
        if _GLOBAL_RANK in dp_ranks:
            _DATA_PARALLEL_GROUP = group

    _IS_INITIALIZED = True
    print_rank_0(f"Distributed initialized: world_size={_WORLD_SIZE}, "
                 f"TP={tensor_model_parallel_size}, DP={data_parallel_size}, "
                 f"backend={backend}")


def get_model_parallel_group():
    return _MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    return _DATA_PARALLEL_GROUP


def get_tensor_model_parallel_world_size():
    if _MODEL_PARALLEL_GROUP is None:
        return 1
    return dist.get_world_size(group=_MODEL_PARALLEL_GROUP)


def get_data_parallel_world_size():
    if _DATA_PARALLEL_GROUP is None:
        return 1
    return dist.get_world_size(group=_DATA_PARALLEL_GROUP)


def get_global_rank():
    return _GLOBAL_RANK


def get_local_rank():
    return _LOCAL_RANK


def get_world_size():
    return _WORLD_SIZE


def is_rank_0():
    return _GLOBAL_RANK == 0


def print_rank_0(msg: str):
    if is_rank_0():
        print(msg, flush=True)


def set_random_seed(seed: int):
    """Set deterministic random seed across all ranks (Megatron convention)."""
    seed = seed + get_global_rank() * 131
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.musa.manual_seed(seed)
    print_rank_0(f"Random seed set to {seed} (base + rank*131)")


def barrier():
    if dist.is_initialized():
        dist.barrier()


def destroy_distributed():
    global _IS_INITIALIZED
    if dist.is_initialized():
        dist.destroy_process_group()
    _IS_INITIALIZED = False


def get_fsdp_wrap_policy(model):
    """
    Returns an FSDP wrapping policy that shards each transformer layer
    individually. This achieves similar memory savings to Megatron TP
    but works with any HuggingFace model on MUSA.
    """
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    import functools

    transformer_layer_cls = set()
    for module in model.modules():
        cls_name = type(module).__name__
        if "DecoderLayer" in cls_name or "Block" in cls_name:
            transformer_layer_cls.add(type(module))
            break

    if not transformer_layer_cls:
        print_rank_0("[WARN] Could not auto-detect transformer layer class for FSDP. "
                     "Falling back to size-based wrapping.")
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        return functools.partial(size_based_auto_wrap_policy, min_num_params=1e7)

    print_rank_0(f"FSDP wrapping on: {[c.__name__ for c in transformer_layer_cls]}")
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls,
    )
