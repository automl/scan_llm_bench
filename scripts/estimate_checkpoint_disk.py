"""
Estimate disk space for checkpoints across a NePS configs directory.

Uses the same checkpoint schedule as training (every 10% of max_tokens_budget,
at least 2 saves per config). You can provide checkpoint size per billion params
or bytes per param once you have measured sizes.

Usage:
  python scripts/estimate_checkpoint_disk.py --neps_configs_dir neps/neps_configs/MyExp/configs
  python scripts/estimate_checkpoint_disk.py --neps_configs_dir ... --max-tokens-budget 5e9 --bytes-per-param 6
"""

import argparse
from pathlib import Path

from plot_utils import get_params_from_neps_configs
from plot_utils import _load_config_from_path, format_params
from checkpoint_utils import get_checkpoint_save_steps
from models.params_flops import flops_from_config


def estimate_num_saves_per_config(
    config_path: Path,
    max_tokens_budget: int,
    default_global_batch: int = 256,
) -> int:
    """
    For a single config YAML, compute how many checkpoints would be saved
    under the 10%-milestone schedule (and at least 2 for small budgets).
    """
    cfg = _load_config_from_path(config_path)
    n_tokens = getattr(cfg, "n_tokens", None)
    if n_tokens is None:
        return 0
    seq_len = getattr(cfg, "seq_len", 2048)
    tokens_per_step = seq_len * default_global_batch
    if tokens_per_step <= 0:
        return 0
    steps_budget = max(1, n_tokens // tokens_per_step)
    save_steps_set = get_checkpoint_save_steps(
        n_tokens=n_tokens,
        steps_budget=steps_budget,
        tokens_per_step=tokens_per_step,
        max_tokens_budget=max_tokens_budget,
    )
    if save_steps_set is None:
        # Fallback: assume save every 10% of steps
        return max(2, min(11, 1 + steps_budget // max(1, steps_budget // 10)))
    return len(save_steps_set)


def main():
    p = argparse.ArgumentParser(
        description="Estimate disk space for checkpoints from configs."
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
        "--max-tokens-budget",
        type=float,
        default=5e9,
        help="Max token budget in search space (default 5B). Used for 10%% save intervals.",
    )
    p.add_argument(
        "--default-global-batch",
        type=int,
        default=256,
        help="Assumed global batch size for steps_budget (default 256)",
    )
    p.add_argument(
        "--bytes-per-param",
        type=float,
        default=None,
        help="Bytes per parameter per checkpoint (e.g. 2 for bf16 model only, 6 for model+optim). Set when you have measured sizes.",
    )
    p.add_argument(
        "--size-per-billion-params-gb",
        type=float,
        default=None,
        help="Checkpoint size in GB per 1B params (overrides --bytes-per-param if set).",
    )
    args = p.parse_args()

    max_tokens_budget = int(args.max_tokens_budget)

    # Single config mode
    if args.config_path is not None:
        cfg_path = Path(args.config_path)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        cfg = _load_config_from_path(cfg_path)
        info = flops_from_config(cfg, n_tokens=None)
        n_params = info["total_params"]
        n_saves = estimate_num_saves_per_config(
            cfg_path,
            max_tokens_budget=max_tokens_budget,
            default_global_batch=args.default_global_batch,
        )

        print("Configs: 1 (single config)")
        print(f"Max tokens budget: {max_tokens_budget / 1e9:.1f}B")
        print(
            f"Parameter count: {format_params(n_params)} "
            f"({n_params:,} total params)"
        )
        print(f"Checkpoint saves for this config: {n_saves}")
    else:
        # Directory (NePS) mode
        root = Path(args.neps_configs_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")

        params_data = get_params_from_neps_configs(root)
        if not params_data:
            print("No configs found or all failed to load.")
            return

        # Per-config: config_id, n_params, n_saves
        rows = []
        for config_dir in sorted(root.iterdir()):
            if not config_dir.is_dir() or not config_dir.name.startswith("config_"):
                continue
            config_path = config_dir / "config.yaml"
            if not config_path.is_file():
                continue
            try:
                n_params = next((p for cid, p in params_data if cid == config_dir.name), None)
                if n_params is None:
                    continue
                n_saves = estimate_num_saves_per_config(
                    config_path,
                    max_tokens_budget=max_tokens_budget,
                    default_global_batch=args.default_global_batch,
                )
                rows.append((config_dir.name, n_params, n_saves))
            except Exception:
                continue

        print(f"Configs: {len(rows)}")
        print(f"Max tokens budget: {max_tokens_budget / 1e9:.1f}B")
        print(f"Parameter count range: {format_params(min(r[1] for r in rows))} .. {format_params(max(r[1] for r in rows))}")
        total_saves = sum(r[2] for r in rows)
        print(f"Total checkpoint saves (all configs): {total_saves}")

    if args.size_per_billion_params_gb is not None:
        # 2 GB per 1B params -> 2 bytes/param
        bytes_per_param = args.size_per_billion_params_gb * (1024**3) / 1e9
    else:
        bytes_per_param = args.bytes_per_param

    if bytes_per_param is not None and bytes_per_param > 0:
        total_bytes = 0
        for config_id, n_params, n_saves in rows:
            total_bytes += n_saves * n_params * bytes_per_param
        total_gb = total_bytes / (1024**3)
        print(f"Assumed size: {bytes_per_param} bytes/param")
        print(f"Estimated total disk for checkpoints: {total_gb:.2f} GB")
    else:
        print("Set --bytes-per-param or --size-per-billion-params-gb to get size estimate (e.g. after measuring a few checkpoints).")


if __name__ == "__main__":
    main()
