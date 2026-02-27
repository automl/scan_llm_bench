"""
Plot utilities for NePS configs: FLOPs distribution and FLOPs per GPU hour.

Use these to analyze FLOPs sampled from a NePS config directory (e.g. neps/neps_configs/<exp>/configs).
FLOPs are computed via the same formula as in training (total_training_flops from config fields),
without needing to initialize the model.
"""

from pathlib import Path
from typing import List, Optional

import yaml
from utils import dict_to_config
from models.params_flops import total_training_flops, format_flops, flops_from_config
import neps


# L40 GPU: ~181 TFLOPS FP16/BF16.
# A100 GPU: ~312 TFLOPS FP16/BF16.
# H100 GPU: ~989 TFLOPS FP16/BF16.
# H200 GPU: ~989 TFLOPS FP16/BF16
GPU_PROFILES = {
    "L40": 181e12,
    "A100": 312e12,
    "H100": 989e12,
    "H200": 989e12,
}

DEFAULT_GPU = "L40"
EXPECTED_MFU = 0.32

def format_params(n: int) -> str:
    """Format parameter count for display (e.g. 12_345_678 -> '12.3M')."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return f"{n:_}"


def get_flops_per_gpu_hour(gpu_name: str) -> float:
    """Returns peak FLOPs per hour for a specific GPU type."""
    peak_flops_s = GPU_PROFILES.get(gpu_name, GPU_PROFILES[DEFAULT_GPU])
    return peak_flops_s * 3600


def _parse_expand(expand_val) -> float:
    """Parse expand (e.g. '8/3' or 2.666) to float."""
    if expand_val is None:
        return 8 / 3
    if isinstance(expand_val, (int, float)):
        return float(expand_val)
    if isinstance(expand_val, str) and "/" in expand_val:
        a, b = expand_val.split("/")
        return float(a) / float(b)
    return float(expand_val)


def _load_config_from_path(config_path: Path):
    """Load config as namedtuple from a YAML path."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config_dict.pop("neps_search_space", None)
    cfg, _ = dict_to_config(config_dict)
    return cfg


def _compute_flops_from_config(cfg) -> Optional[int]:
    """Compute total training FLOPs from config (same logic as construct_model/train)."""
    n_tokens = getattr(cfg, "n_tokens", None)
    if n_tokens is None:
        return None
    try:
        return total_training_flops(
            n_layers=cfg.n_layers,
            d=cfg.d_model,
            vocab_size=cfg.vocab_size,
            seq_len=cfg.seq_len,
            n_tokens=n_tokens,
            mlp_class=getattr(cfg, "mlp_class", "glu"),
            expand=_parse_expand(getattr(cfg, "expand", "8/3")),
        )
    except Exception:
        return None


def get_flops_from_neps_configs(
    neps_configs_dir: str | Path,
    config_yaml_name: str = "config.yaml",
) -> list[tuple[str, int]]:
    """
    Load all configs from a NePS configs directory and compute FLOPs for each.

    Scans `<neps_configs_dir>/config_*/config.yaml` (e.g. neps/neps_configs/test_neps/configs),
    loads each config, and computes total training FLOPs using the same formula as
    `construct_model` / `total_training_flops` (no model instantiation needed).

    Args:
        neps_configs_dir: Path to the NePS configs directory (e.g. .../neps_configs/MyExp/configs).
        config_yaml_name: Name of the config file inside each config_XX folder (default: config.yaml).

    Returns:
        List of (config_id, flops) where config_id is e.g. "config_1", and flops is total
        training FLOPs (int). Configs that fail to load or have no n_tokens are skipped.
    """
    root = Path(neps_configs_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    results = []
    for config_dir in sorted(root.iterdir()):
        if not config_dir.is_dir() or not config_dir.name.startswith("config_"):
            continue
        config_path = config_dir / config_yaml_name
        if not config_path.is_file():
            continue
        try:
            cfg = _load_config_from_path(config_path)
            total_flops = getattr(cfg, "total_flops", None)
            if total_flops is not None:
                results.append((config_dir.name, total_flops))
        except Exception:
            continue
    return results


def get_params_from_neps_configs(
    neps_configs_dir: str | Path,
    config_yaml_name: str = "config.yaml",
) -> list[tuple[str, int]]:
    """
    Load all configs from a NePS configs directory and compute total parameter count for each.

    Uses the same formula as params_flops.tot_params.
    Configs that fail to load are skipped.

    Returns:
        List of (config_id, total_params) e.g. ("config_1", 12_345_678).
    """
    root = Path(neps_configs_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    results = []
    for config_dir in sorted(root.iterdir()):
        if not config_dir.is_dir() or not config_dir.name.startswith("config_"):
            continue
        config_path = config_dir / config_yaml_name
        if not config_path.is_file():
            continue
        try:
            cfg = _load_config_from_path(config_path)
            total_params = getattr(cfg, "total_params", None)
            if total_params is not None:
                results.append((config_dir.name, total_params))
        except Exception:
            continue
    return results


def get_tokens_from_neps_configs(
    neps_configs_dir: str | Path,
    config_yaml_name: str = "config.yaml",
) -> list[tuple[str, int]]:
    """
    Scan NePS configs and extract the `n_tokens` budget from each.

    Similar to the other helper functions, this walks through
    `<neps_configs_dir>/config_*/config.yaml`, loads each config, and
    returns the value of ``cfg.n_tokens`` when present. Any configs that
    fail to load or that lack the field are skipped.

    Args:
        neps_configs_dir: Directory containing the individually numbered
            config folders.
        config_yaml_name: Filename inside each config folder (default
            ``config.yaml``).

    Returns:
        A list of tuples ``(config_id, n_tokens)``.
    """
    root = Path(neps_configs_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    results: list[tuple[str, int]] = []
    for config_dir in sorted(root.iterdir()):
        if not config_dir.is_dir() or not config_dir.name.startswith("config_"):
            continue
        config_path = config_dir / config_yaml_name
        if not config_path.is_file():
            continue
        try:
            cfg = _load_config_from_path(config_path)
            n_tokens = getattr(cfg, "n_tokens", None)
            if n_tokens is not None:
                results.append((config_dir.name, n_tokens))
        except Exception:
            continue
    return results


def plot_flops_distribution(
    flops_data: list[tuple[str, int]],
    *,
    title: Optional[str] = "FLOPs distribution sampled from NePS",
    xlabel: Optional[str] = "Total training FLOPs",
    log_scale: bool = True,
    ax=None,
    **kwargs,
):
    """
    Plot the distribution of FLOPs from NePS configs (e.g. from get_flops_from_neps_configs).

    Args:
        flops_data: List of (config_id, flops) as returned by get_flops_from_neps_configs.
        title: Plot title.
        xlabel: X-axis label.
        log_scale: If True, use log scale for FLOPs (default True).
        ax: Matplotlib axes to plot on. If None, uses current axes or creates a new figure.
        **kwargs: Passed to matplotlib hist (e.g. bins=30, density=True, edgecolor='black').
    """
    import matplotlib.pyplot as plt

    flops_list = [f for _, f in flops_data]
    if not flops_list:
        raise ValueError("flops_data is empty; nothing to plot.")

    if ax is None:
        fig, ax = plt.subplots()
    if log_scale:
        import numpy as np

        flops_arr = np.array(flops_list, dtype=float)
        flops_arr = np.log10(flops_arr)
        ax.hist(flops_arr, **kwargs)
        ax.set_xlabel(f"{xlabel} (log10)")
    else:
        ax.hist(flops_list, **kwargs)
        ax.set_xlabel(xlabel)
    if title:
        ax.set_ylabel("Count")
        ax.set_title(title)
    return ax


def plot_flops_per_gpu_hour(
        flops_data: list[tuple[str, int]],
        *,
        flops_per_gpu_hour: Optional[float] = None,
        num_gpus: int = 1,
        gpu_name: str = "L40",
        title: Optional[str] = None,
        log_scale: bool = True,
        ax=None,
        **kwargs,
):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

    # Calculate throughput based on selected GPU and MFU
    if flops_per_gpu_hour is None:
        base_flops_per_hour = get_flops_per_gpu_hour(gpu_name)
        # Total throughput = (Peak per GPU * num_gpus) * Efficiency
        total_throughput_per_hour = base_flops_per_hour * num_gpus * EXPECTED_MFU
    else:
        total_throughput_per_hour = flops_per_gpu_hour

    gpu_hours = [flops / total_throughput_per_hour for _, flops in flops_data]

    if not gpu_hours:
        raise ValueError("flops_data is empty; nothing to plot.")

    total_hours = np.sum(gpu_hours)
    print(f"--- Projection for {num_gpus}x {gpu_name} ---")
    print(f"Total projected training time: {total_hours:.2f} hours (at {EXPECTED_MFU * 100}% MFU)")

    if ax is None:
        fig, ax = plt.subplots()

    if log_scale:
        hours_arr = np.array(gpu_hours, dtype=float)
        hours_arr = np.log10(np.clip(hours_arr, 1e-12, None))
        ax.hist(hours_arr, **kwargs)

        def format_log_hours(x, pos):
            actual_hours = 10 ** x
            return f"{actual_hours:.1f}h" if actual_hours >= 1 else f"{actual_hours:.2f}h"

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_log_hours))
        ax.set_xlabel(f"Training time on {num_gpus}x {gpu_name} (GPU hours, log10)")
    else:
        ax.hist(gpu_hours, **kwargs)
        ax.set_xlabel(f"Training time on {num_gpus}x {gpu_name} (GPU hours)")

    if title is None:
        title = f"Projected Time: {gpu_name} Cluster (MFU={EXPECTED_MFU * 100}%)"
    ax.set_ylabel("Count")
    ax.set_title(title)
    return ax


def get_metrics_from_neps_reports(
        neps_configs_dir: str | Path,
        report_yaml_name: str = "report.yaml",
        num_gpus: int = 1,
        gpu_name: str = "L40",
) -> List[dict]:
    """
    Scans NePS directory for report.yaml files and extracts
    FLOPs (cost) and actual time taken.
    """
    root = Path(neps_configs_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    peak_flops_per_sec = GPU_PROFILES.get(gpu_name, GPU_PROFILES["L40"])

    results = []
    for config_dir in sorted(root.iterdir()):
        if not config_dir.is_dir() or not config_dir.name.startswith("config_"):
            continue
        report_path = config_dir / report_yaml_name
        if not report_path.is_file():
            continue

        try:
            with open(report_path, "r") as f:
                report = yaml.safe_load(f)

            extra = report.get("extra", {})
            total_flops = report.get("cost")
            duration = extra.get("time_duration_seconds")

            if total_flops and duration:
                # achieved_flops_per_sec = Total FLOPs / time in seconds
                achieved_flops_per_sec = (total_flops / duration)

                # MFU = Achieved / (Peak_Per_GPU * Num_GPUs)
                mfu = (achieved_flops_per_sec / (peak_flops_per_sec * num_gpus)) * 100

                results.append({
                    "config_id": config_dir.name,
                    "total_flops": total_flops,
                    "duration": duration,
                    "mfu": mfu,
                    "achieved_tflops_s": achieved_flops_per_sec / 1e12
                })
        except Exception as e:
            print(f"Skipping {config_dir.name} due to error: {e}")
            continue

    return results

def plot_mfu_distribution(
    metrics_data: List[dict],
    title: str = "Model FLOPs Utilization (MFU) Distribution",
    ax=None,
    **kwargs
):
    """Plots a histogram of MFU values across all sampled configs."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    import numpy as np

    mfu_values = [d["mfu"] for d in metrics_data]
    
    # Standard styling
    kwargs.setdefault("bins", 20)
    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("alpha", 0.7)
    kwargs.setdefault("color", "skyblue")

    ax.hist(mfu_values, **kwargs)
    ax.axvline(np.mean(mfu_values), color='red', linestyle='--', label=f'Mean: {np.mean(mfu_values):.1f}%')
    
    ax.set_xlabel("MFU (%)")
    ax.set_ylabel("Number of Configs")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    return ax

def plot_flops_vs_mfu(metrics_data: List[dict], ax=None):
    """
    Scatter plot to see if larger models (more FLOPs) 
    result in better or worse GPU utilization.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    flops = [d["total_flops"] for d in metrics_data]
    mfu = [d["mfu"] for d in metrics_data]

    sc = ax.scatter(flops, mfu, alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel("Total Theoretical FLOPs (Log Scale)")
    ax.set_ylabel("MFU (%)")
    ax.set_title("Model Size vs. Efficiency")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    return ax

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot FLOPs distribution and FLOPs per GPU hour from NePS configs."
    )
    parser.add_argument(
        "--neps_configs_dir",
        default="neps/neps_configs/Possible_1k_Neps_Full_Scale_Search_Space/configs",
        type=str,
        help="Path to NePS configs directory (e.g. neps/neps_configs/MyExp/configs)",
    )
    parser.add_argument(
        "--out-dist",
        default="plots/test.png",
        type=str,
        help="Save FLOPs distribution figure to this path",
    )
    parser.add_argument(
        "--out-gpu-hours",
        default="plots/h200/gpu_hours.png",
        type=str,
        help="Save FLOPs per GPU hour figure to this path",
    )
    parser.add_argument(
        "--out-mfu-distribution",
        default="plots/test2.png",
        type=str,
        help="Save MFU distribution figure to this path",
    )
    parser.add_argument(
        "--out-flops-vs-mfu",
        default="plots/test3.png",
        type=str,
        help="Save FLOPs vs MFU figure to this path",
    )
    parser.add_argument(
        "--gpu-name",
        type=str,
        default="H200",
        choices=["L40", "A100", "H100", "H200"],
        help="GPU type for time calculation"
    )
    parser.add_argument(
        "--flops-per-gpu-hour",
        type=float,
        default=None,
        help=f"FLOPs per GPU-hour (default: L40)",
    )
    parser.add_argument("--no-log-scale", action="store_true", help="Use linear scale")
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    flops_data = get_flops_from_neps_configs(args.neps_configs_dir)
    # Always plot the hours for single-gpu hours since we will use variable gpu's depending on the config needs
    num_gpus = 1
    if not flops_data:
        print("No configs with valid FLOPs found.")
        exit(1)
    print(f"Loaded {len(flops_data)} configs.")
    print(f"FLOPs range: {format_flops(min(f for _, f in flops_data))} .. {format_flops(max(f for _, f in flops_data))}")

    params_data = get_params_from_neps_configs(args.neps_configs_dir)
    if params_data:
        min_params = min(p for _, p in params_data)
        max_params = max(p for _, p in params_data)
        print(f"Parameter count range: {format_params(min_params)} .. {format_params(max_params)} (min {min_params:,} / max {max_params:,} total params)")

    tokens_data = get_tokens_from_neps_configs(args.neps_configs_dir)
    if tokens_data:
        min_tokens = min(t for _, t in tokens_data)
        max_tokens = max(t for _, t in tokens_data)
        print(f"Token budget range: {min_tokens:,} .. {max_tokens:,} tokens")


    if args.out_dist or not (args.out_dist or args.out_gpu_hours):
        fig, ax = plt.subplots()
        plot_flops_distribution(
            flops_data, log_scale=not args.no_log_scale, bins=30, edgecolor="black", alpha=0.7
        )
        if args.out_dist:
            plt.savefig(args.out_dist, bbox_inches="tight")
            print(f"Saved distribution plot to {args.out_dist}")
        else:
            plt.show()

    if args.out_gpu_hours or not (args.out_dist or args.out_gpu_hours):
        fig, ax = plt.subplots()
        plot_flops_per_gpu_hour(
            flops_data,
            flops_per_gpu_hour=args.flops_per_gpu_hour,
            num_gpus=num_gpus,
            gpu_name=args.gpu_name,
            log_scale=not args.no_log_scale,
            bins=30,
            edgecolor="black",
            alpha=0.7,
        )
        if args.out_gpu_hours:
            plt.savefig(args.out_gpu_hours, bbox_inches="tight")
            print(f"Saved GPU hours plot to {args.out_gpu_hours}")
        else:
            plt.show()

    results = get_metrics_from_neps_reports(
        args.neps_configs_dir,
        num_gpus=num_gpus,
        gpu_name=args.gpu_name
    )

    if args.out_mfu_distribution:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_mfu_distribution(results)
        plt.savefig(args.out_mfu_distribution, bbox_inches="tight")

    if args.out_flops_vs_mfu:
        plot_flops_vs_mfu(results)
        plt.savefig(args.out_flops_vs_mfu, bbox_inches="tight")

