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
from models.params_flops import total_training_flops, format_flops, flops_from_config



def add_total_params_and_flops(config_paths: List[Path]) -> int:
    """Iterate over all configs and return the maximum n_tokens found."""
    for path in config_paths:
        try:
            with path.open("r") as f:
                cfg = yaml.safe_load(f)
                n_tokens_cfg = cfg.get("n_tokens", 0)
                flops_info = flops_from_config(cfg, n_tokens=n_tokens_cfg)
                cfg['total_params'] = flops_info['total_params']
                cfg['total_training_flops'] = flops_info.get('total_training_flops')
                print(f"Wrote total params: {flops_info['total_params']} and FLOPs: {flops_info.get('total_training_flops')}")
        except Exception as e:
            print(f"Warning: Could not read n_tokens from {path}: {e}")
    return int(1)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepare configs with micro_batch_size, grad_accumulation_steps, and recommended_num_gpus.",
    )
    p.add_argument(
        "--neps_configs_dir",
        type=str,
        default="C:\Freiburg\Tema\Scaling_Law\scan_llm_bench\\neps\\neps_configs\Biggest_Manual_Configs",
        help="Path to NePS configs directory (e.g. neps/neps_configs/MyExp/configs)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for probing (default: cuda if available, else error).",
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

    # Directory (NePS) mode
    config_paths = get_config_paths(args.neps_configs_dir)
    if config_paths is None:
        raise Exception("No configs found")

    t = add_total_params_and_flops(config_paths)
    return


if __name__ == "__main__":
    main()

