"""
Probe max micro_batch_size from low to high until OOM.

Finds the largest batch size that fits in GPU memory by probing
ascending sizes (4, 8, 16, 32, 64, ...). Outputs:
- max_micro_batch_size: largest that fits
- suggested grad_accumulation_steps for target global batch (Tiny/Small/Medium)
"""

import argparse
import gc
import os
import sys

import torch
from torch.nn import CrossEntropyLoss
from contextlib import nullcontext

import yaml
from collections import namedtuple

from models import construct_model


def load_config_simple(path: str):
    """Load config from YAML without FLAGS dependency."""
    with open(path) as f:
        config_dict = yaml.safe_load(f)
    config_dict.pop("neps_search_space", None)
    Config = namedtuple("Config", config_dict.keys())
    return Config(**config_dict)


def get_recommended_num_gpus(target_global_batch: int, max_micro_batch_size: int) -> int:
    """
    Minimum number of GPUs needed to reach target_global_batch without grad accumulation.
    Use this to assign different #GPUs per model when scheduling jobs.
    """
    if max_micro_batch_size <= 0:
        return 1
    return max(1, (target_global_batch + max_micro_batch_size - 1) // max_micro_batch_size)


def get_target_global_batch_for_tokens(n_tokens: int) -> int:
    """Target global batch size from number of tokens and model param count.
    """
    if n_tokens < 100_000_000:
        return 16
    if n_tokens < 500_000_000:
        return 32
    if n_tokens < 1_000_000_000:
        return 64
    if n_tokens < 3_000_000_000:
        return 128
    return 256


def dummy_batch(bs: int, seq_len: int, vocab_size: int, device: str):
    """Create a dummy batch matching scan_llm_bench format."""
    input_ids = torch.randint(
        0, vocab_size, (bs, seq_len + 1), device=device, dtype=torch.long)
    return {"input_ids": input_ids}


def train_one_step(model, loss_fn, inputs, targets, device, dtype, attn_mask=None):
    """Run one forward+backward step."""
    model.train()
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32,
               "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(
        device_type=device_type, dtype=ptdtype)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)

    with ctx:
        output = model(inputs, attn_mask)
        logits = getattr(output, "logits", output)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)),
                       targets.reshape(-1))

    loss.backward()
    optimizer.step()
    return loss.detach()


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def find_max_batch_size_ascending(
    cfg,
    model,
    device,
    seq_len: int,
    vocab_size: int,
    probe_sizes: list[int] | None = None,
    max_bs: int = 2048,
    max_memory_fraction: float = 0.88,
) -> int:
    """
    Probe from LOW to HIGH batch sizes until OOM.
    Returns the largest batch size that fits within max_memory_fraction of GPU memory
    (to leave headroom for DDP/torch.compile and reduce fragmentation OOMs).
    """
    if probe_sizes is None:
        probe_sizes = []
        bs = 4
        while bs <= max_bs:
            probe_sizes.append(bs)
            bs *= 2

    loss_fn = CrossEntropyLoss()
    dtype = getattr(cfg, "dtype", "bfloat16")

    total_gb = 0.0
    if torch.cuda.is_available():
        idx = device.index if hasattr(
            device, "index") else torch.cuda.current_device()
        total_gb = torch.cuda.get_device_properties(idx).total_memory / 1024**3

    working_bs = None
    last_peak_gb = 0.0

    for bs in probe_sizes:
        try:
            print(f"  Probing micro_batch_size={bs}...", flush=True)
            _cleanup()

            batch = dummy_batch(bs, seq_len, vocab_size, device)
            inputs = batch["input_ids"][:, :seq_len]
            targets = batch["input_ids"][:, 1: (seq_len + 1)]

            _ = train_one_step(
                model=model,
                loss_fn=loss_fn,
                inputs=inputs,
                targets=targets,
                device=device,
                dtype=dtype,
            )

            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            last_peak_gb = peak_gb
            # Enforce safety margin: real training uses DDP + torch.compile + persistent
            # optimizer, so probe can be optimistic. Cap at max_memory_fraction of GPU.
            if total_gb > 0 and peak_gb > total_gb * max_memory_fraction:
                print(
                    f"    batch {bs} | peak VRAM: {peak_gb:.2f} GB > {max_memory_fraction*100:.0f}% of {total_gb:.2f} GB; using previous batch size.",
                    flush=True,
                )
                _cleanup()
                break
            working_bs = bs
            print(f"    batch {bs} | peak VRAM: {peak_gb:.2f} GB", flush=True)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    batch {bs} OOM. Stopping.", flush=True)
                _cleanup()
                break
            raise
        finally:
            _cleanup()

    if working_bs is None:
        raise RuntimeError(
            "All batch sizes caused OOM. Try smaller probe_sizes (e.g. start from 2)."
        )

    return working_bs


def get_batch_config(
    cfg,
    model,
    device: str,
    num_gpus: int,
    probe_sizes: list[int] | None = None,
    max_bs: int = 2048,
    max_memory_fraction: float = 0.88,
) -> tuple[int, int, int]:
    """
    Probe max micro_batch_size and compute grad_accumulation_steps from model size.

    Uses model.count_params() to choose target global batch (<5M→16, 5-20M→64,
    20-70M→128, 70M+→256), then probes until OOM and derives grad_accum to hit that target.

    Args:
        cfg: Config namedtuple (seq_len, vocab_size, dtype, etc.).
        model: Model already on `device` (will be used for probe).
        device: Device string (e.g. 'cuda:0').
        num_gpus: World size for grad_accumulation_steps.
        probe_sizes: Optional list of batch sizes to probe; default 4,8,16,... .
        max_bs: Max batch size when probe_sizes is auto-generated.

    Returns:
        (micro_batch_size, grad_accumulation_steps)
    """
    seq_len = cfg.seq_len
    vocab_size = cfg.vocab_size
    target_global = get_target_global_batch_for_tokens(cfg.n_tokens)

    max_micro_bs = find_max_batch_size_ascending(
        cfg=cfg,
        model=model,
        device=device,
        seq_len=seq_len,
        vocab_size=vocab_size,
        probe_sizes=probe_sizes,
        max_bs=max_bs,
        max_memory_fraction=max_memory_fraction,
    )
    grad_accum = max(
        1,
        (target_global + max_micro_bs * num_gpus - 1) // (max_micro_bs * num_gpus),
    )
    # Only use grad_accum when we can't reach target with available GPUs; else 1
    recommended_gpus = get_recommended_num_gpus(target_global, max_micro_bs)
    if num_gpus >= recommended_gpus:
        grad_accum = 1
    _cleanup()
    return max_micro_bs, grad_accum, recommended_gpus


def print_vram(prefix: str = ""):
    """Print current VRAM usage."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    print(
        f"{prefix} | total: {total:.2f} GB | allocated: {allocated:.2f} GB | "
        f"reserved: {reserved:.2f} GB",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(
        description="Probe max batch size from low to high")
    p.add_argument("--config", required=True, help="Path to config YAML")
    p.add_argument(
        "--model_size",
        choices=["tiny", "small", "medium"],
        default=None,
        help="Optional override: tiny=16, small=64, medium=256. If unset, target is derived from model.count_params().",
    )
    p.add_argument(
        "--probe_sizes",
        type=str,
        default="4,8,16,32,64,128,256",
        help="Comma-separated batch sizes to probe, e.g. '4,8,16,32,64,128,256'",
    )
    p.add_argument("--num_gpus", type=int, default=2,
                   help="Number of GPUs for grad_accum suggestion")
    args = p.parse_args()

    cfg = load_config_simple(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print_vram("Before model load")
    model, _, _ = construct_model(cfg)
    model = model.to(device)
    print_vram("After model load")

    seq_len = cfg.seq_len
    vocab_size = cfg.vocab_size

    probe_sizes = None
    if args.probe_sizes:
        probe_sizes = [int(x.strip()) for x in args.probe_sizes.split(",")]

    max_bs = find_max_batch_size_ascending(
        cfg=cfg,
        model=model,
        device=device,
        seq_len=seq_len,
        vocab_size=vocab_size,
        probe_sizes=probe_sizes,
    )

    target_global_batch = get_target_global_batch_for_tokens(cfg.n_tokens)
    
    grad_accum = max(
        1, (target_global_batch + max_bs *
            args.num_gpus - 1) // (max_bs * args.num_gpus)
    )
    actual_global = max_bs * grad_accum * args.num_gpus

    print("")
    print(f"=== Result: max_micro_batch_size = {max_bs} ===")
    print(
        f"  Token Budget: {cfg.n_tokens:,} (target global batch {target_global_batch})")
    print(
        f"  Suggested grad_accumulation_steps: {grad_accum} (for target ~{target_global_batch})")
    print(f"  Actual global batch size: {actual_global}")

    print("")
    print("Add to your config YAML:")
    print(f"  micro_batch_size: {max_bs}")
    print(f"  grad_accumulation_steps: {grad_accum}")


if __name__ == "__main__":
    main()
