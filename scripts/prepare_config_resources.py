"""Prepare NePS configs with micro_batch_size, grad_accumulation_steps, and
recommended_num_gpus before launching training.

- Loads each config_*/config.yaml under a NePS configs directory
- Builds the model
- Uses the existing probe logic (scripts.probe_batch_size.get_batch_config)
  on a single GPU to find:
    * max micro_batch_size that fits
    * grad_accumulation_steps to hit a target global batch
- Derives recommended_num_gpus from target_global_batch and max_micro_batch_size
- Writes these values back into the YAML:
    micro_batch_size
    grad_accumulation_steps
    recommended_num_gpus
    auto_batch_size: False   # so train.py won't re-probe
- Optionally sets max_tokens_budget for checkpoint scheduling.

Run this once before submitting training jobs so that Slurm scripts
can allocate the right number of GPUs per config.
"""

from pathlib import Path
import argparse
import yaml
import torch

from models import construct_model
from scripts.probe_batch_size import (
    load_config_simple,
    get_batch_config,
    get_target_global_batch_for_tokens,
    get_recommended_num_gpus,
)


def prepare_single_config(
    config_path: Path,
    device: str,
    max_tokens_budget: int | None,
    max_gpus_cap: int,
) -> None:
    """Probe batch config and write fields back into a single YAML config."""
    cfg = load_config_simple(str(config_path))

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available; prepare_config_resources.py must run on a GPU node "
            "so it can probe micro_batch_size."
        )

    model, _, _ = construct_model(cfg)
    model = model.to(device)

    # Probe on a single GPU: this finds the largest micro_batch_size that fits.
    micro_bs, grad_accum = get_batch_config(
        cfg,
        model,
        device=device,
        num_gpus=1,
    )

    target_global = get_target_global_batch_for_tokens(cfg.n_tokens)
    recommended_gpus = get_recommended_num_gpus(target_global, micro_bs)
    if max_gpus_cap is not None and max_gpus_cap > 0:
        recommended_gpus = max(1, min(recommended_gpus, max_gpus_cap))

    with config_path.open("r") as f:
        config_dict = yaml.safe_load(f)

    config_dict["micro_batch_size"] = int(micro_bs)
    config_dict["grad_accumulation_steps"] = int(grad_accum)
    config_dict["recommended_num_gpus"] = int(recommended_gpus)
    # Disable in-train probing so train.py just uses these values
    config_dict["auto_batch_size"] = False

    if max_tokens_budget is not None and max_tokens_budget > 0:
        config_dict.setdefault("max_tokens_budget", int(max_tokens_budget))

    with config_path.open("w") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print(
        f"[{config_path}] micro_batch_size={micro_bs}, "
        f"grad_accumulation_steps={grad_accum}, "
        f"recommended_num_gpus={recommended_gpus}"
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
        "--max-tokens-budget",
        type=float,
        default=None,
        help="Optional global max token budget (e.g. 5e9) used for checkpoint scheduling.",
    )
    p.add_argument(
        "--max-gpus-cap",
        type=int,
        default=8,
        help="Upper cap for recommended_num_gpus (default: 8).",
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

    max_tokens_budget = int(args.max_tokens_budget) if args.max_tokens_budget is not None else None

    # Single config mode
    if args.config_path is not None:
        cfg_path = Path(args.config_path)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        prepare_single_config(
            cfg_path,
            device=device,
            max_tokens_budget=max_tokens_budget,
            max_gpus_cap=args.max_gpus_cap,
        )
        return

    # Directory (NePS) mode
    root = Path(args.neps_configs_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    config_paths = []
    for config_dir in sorted(root.iterdir()):
        if not config_dir.is_dir() or not config_dir.name.startswith("config_"):
            continue
        cfg_path = config_dir / "config.yaml"
        if cfg_path.is_file():
            config_paths.append(cfg_path)

    if not config_paths:
        print(f"No config_*/config.yaml files found under {root}")
        return

    print(f"Found {len(config_paths)} configs under {root}")
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

