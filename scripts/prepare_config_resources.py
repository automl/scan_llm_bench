"""Prepare NePS configs with micro_batch_size, grad_accumulation_steps, and
recommended_num_gpus before launching training.

- Loads each config_*/config.yaml under a NePS configs directory
- Builds the model
- Uses the existing probe logic (scripts.probe_batch_size.get_batch_config)
  on a single GPU to find:
    * max micro_batch_size that fits on the given amount of gpus
    * grad_accumulation_steps to hit a target global batch
- Derives recommended_num_gpus from target_global_batch, max_micro_batch_size and max_gpus_cap
- Writes these values back into the YAML:
    micro_batch_size
    grad_accumulation_steps
    recommended_num_gpus
    auto_batch_size: False   # so train.py won't re-probe

Run this once before submitting training jobs so that Slurm scripts
can allocate the right number of GPUs per config.
"""

import argparse
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from models import construct_model
from utils import get_config_paths
from scripts.probe_batch_size import (
    get_batch_config,
    get_recommended_num_gpus,
    get_target_global_batch_for_tokens,
    load_config_simple,
)


def get_max_tokens_from_configs(config_paths: List[Path]) -> int:
    """Iterate over all configs and return the maximum n_tokens found."""
    max_tokens = 0
    for path in config_paths:
        try:
            with path.open("r") as f:
                cfg = yaml.safe_load(f)
                n_tokens = cfg.get("n_tokens", 0)
                if n_tokens > max_tokens:
                    max_tokens = n_tokens
        except Exception as e:
            print(f"Warning: Could not read n_tokens from {path}: {e}")
    return int(max_tokens)


def prepare_single_config(
    config_path: Path,
    device: str,
    max_tokens_budget: int | None,
    max_gpus_cap: int,
) -> None:
    """Probe batch config and write fields back into a single YAML config."""
    cfg = load_config_simple(str(config_path))
    if max_gpus_cap is None:
        raise RuntimeError("Max gpu limit needs to be set")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available; prepare_config_resources.py must run on a GPU node "
            "so it can probe micro_batch_size."
        )

    model, _, _ = construct_model(cfg)
    model = model.to(device)

    # Probe on a single GPU: this finds the largest micro_batch_size that fits.
    micro_bs, _, initial_recommended_gpus = get_batch_config(
        cfg,
        model,
        device=device,
        num_gpus=max_gpus_cap,
    )

    target_global = get_target_global_batch_for_tokens(cfg.n_tokens)
    
    print("Initial recommended_num_gpus based on target_global_batch and micro_batch_size:", initial_recommended_gpus)
    recommended_gpus = max(1, min(initial_recommended_gpus, max_gpus_cap))
    print(f"Recommended_num_gpus after applying max_gpus_cap of {max_gpus_cap}:", recommended_gpus)

    grad_accum = max(
        1,
        (target_global + micro_bs * recommended_gpus - 1) // (micro_bs * recommended_gpus),
    )
    print(f"Newly calculated grad acc after changes in number of GPUs: {grad_accum}")

    with config_path.open("r") as f:
        config_dict = yaml.safe_load(f)

    config_dict["micro_batch_size"] = int(micro_bs)
    config_dict["grad_accumulation_steps"] = int(grad_accum)
    config_dict["recommended_num_gpus"] = int(recommended_gpus)
    # Disable in-train probing so train.py just uses these values
    config_dict["auto_batch_size"] = False

    if max_tokens_budget is not None and max_tokens_budget > 0:
        config_dict["max_tokens_budget"] = int(max_tokens_budget)

    with config_path.open("w") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print(
        f"[{config_path}] micro_batch_size={micro_bs}, "
        f"grad_accumulation_steps={grad_accum}, "
        f"recommended_num_gpus={recommended_gpus}, "
        f"target_global_batch={target_global}"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepare configs with micro_batch_size, grad_accumulation_steps, and recommended_num_gpus.",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--neps_configs_dir",
        type=str,
        help="Path to NePS configs directory (e.g. neps/neps_configs/MyExp/configs)",
    )
    group.add_argument(
        "--config_path",
        type=str,
        help="Path to a single config.yaml file.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for probing (default: cuda if available, else error).",
    )
    p.add_argument(
        "--max-gpus-cap",
        type=int,
        default=1,
        help="Upper cap for recommended_num_gpus (default: 1).",
    )
    args = p.parse_args()

    if args.device is not None:
        device = args.device
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available and no --device was specified. "
                "Run this on a GPU node or pass --device cuda:0 explicitly."
            )
        device = "cuda"

    if args.config_path is not None:
        cfg_path = Path(args.config_path)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        
        max_tokens_budget = get_max_tokens_from_configs([cfg_path])

        prepare_single_config(
            cfg_path,
            device=device,
            max_tokens_budget=max_tokens_budget,
            max_gpus_cap=args.max_gpus_cap,
        )
        return

    # Directory (NePS) mode
    config_paths = get_config_paths(args.neps_configs_dir)
    if config_paths is None:
        raise Exception("No configs found")

    max_tokens_budget = get_max_tokens_from_configs(config_paths)
    if max_tokens_budget > 0:
        print(f"Auto-detected max_tokens_budget={max_tokens_budget} across {len(config_paths)} configs")

    print(f"Found {len(config_paths)} configs under {args.neps_configs_dir}")
    for cfg_path in config_paths:
        try:
            prepare_single_config(
                cfg_path,
                device=device,
                max_tokens_budget=max_tokens_budget,
                max_gpus_cap=args.max_gpus_cap,
            )
        except Exception as e:
            print(f"Skipping {cfg_path} due to error: {e}")


if __name__ == "__main__":
    main()

