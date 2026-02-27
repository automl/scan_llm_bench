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
from scripts.prepare_config_resources import get_max_tokens_from_configs


def _tokens_per_step_from_config(cfg) -> int:
    seq_len = getattr(cfg, "seq_len", 2048)
    micro_batch = getattr(cfg, "micro_batch_size", None)
    grad_accum = getattr(cfg, "grad_accumulation_steps", None)
    global_batch = None
    if micro_batch is not None and grad_accum is not None:
        try:
            global_batch = int(micro_batch) * int(grad_accum)
        except Exception:
            global_batch = None
    return max(1, seq_len * global_batch)


def estimate_num_saves_per_config(
    config_path: Path,
    max_tokens_budget: int,
) -> int:
    """
    For a single config YAML, compute how many checkpoints would be saved
    under the 10%-milestone schedule (and at least 2 for small budgets).
    """
    cfg = _load_config_from_path(config_path)
    n_tokens = getattr(cfg, "n_tokens", None)
    if n_tokens is None:
        return 0
    tokens_per_step = _tokens_per_step_from_config(cfg)
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
        # Fallback: save every 10% of steps
        return max(2, min(11, 1 + steps_budget // max(1, steps_budget // 10)))
    print(f"Save steps: {len(save_steps_set)} ({save_steps_set}) for config {config_path.name} with n_tokens={n_tokens}, steps_budget={steps_budget}, tokens_per_step={tokens_per_step}")
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

    today_rows: list[tuple[str, int, int]] = []
    if args.config_path is not None:
        cfg_path = Path(args.config_path)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        cfg = _load_config_from_path(cfg_path)
        info = flops_from_config(cfg, n_tokens=None)
        max_tokens_budget = get_max_tokens_from_configs([cfg_path])
        n_params = getattr(cfg, "total_params", None)
        n_saves = estimate_num_saves_per_config(
            cfg_path,
            max_tokens_budget=max_tokens_budget,
            default_global_batch=args.default_global_batch,
        )
        today_rows.append((cfg_path.name, n_params, n_saves))

        print("Configs: 1 (single config)")
        print(f"Max tokens budget: {max_tokens_budget / 1e9:.1f}B")
        print(
            f"Parameter count: {format_params(n_params)} "
            f"({n_params:,} total params)"
        )
        print(f"Checkpoint saves for this config: {n_saves}")
    else:
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
        params_data = get_params_from_neps_configs(root)
        if not params_data:
            print("No configs found or all failed to load.")
            return

        print("Found params for configs. Estimating saves...")
        
        max_tokens_budget = get_max_tokens_from_configs(config_paths)
        print(f"Max tokens budget {max_tokens_budget}")
        
        for cfg_path in config_paths:
            try:
                n_params = next((p for cid, p in params_data if cid == cfg_path.parent.name), None)
                print(f"Params for {cfg_path.parent.name}: {n_params}")
                if n_params is None:
                    continue
                
                n_saves = estimate_num_saves_per_config(
                    cfg_path,
                    max_tokens_budget=max_tokens_budget,
                )
                today_rows.append((cfg_path.parent.name, n_params, n_saves))
            except Exception as e:
                print(f"Error processing {cfg_path}: {e}")
                continue

        if not today_rows:
            print("No valid configs processed.")
            return

        print(f"Configs: {len(today_rows)}")
        print(f"Max tokens budget: {max_tokens_budget / 1e9:.1f}B")
        print(
            f"Parameter count range: {format_params(min(r[1] for r in today_rows))} .. "
            f"{format_params(max(r[1] for r in today_rows))}"
        )
        total_saves = sum(r[2] for r in today_rows)
        print(f"Total checkpoint saves (all configs): {total_saves}")

    rows = today_rows
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
