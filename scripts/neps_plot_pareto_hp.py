from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

ROOT_DIRECTORY = Path(__file__).resolve().parents[1] / "neps" / "neps_configs" / "neps_1000_configs"
DEFAULT_SUMMARY_DIR = ROOT_DIRECTORY / "summary"

HP_LABELS = {
    "beta1": "Beta 1",
    "beta2": "Beta 2",
    "cooldown_steps": "Cooldown steps",
    "d_model": "Embedding size",
    "lr": "Learning rate",
    "micro_batch_size": "Micro batch size",
    "n_heads": "Number of heads",
    "n_layers": "Number of layers",
    "n_tokens": "Training tokens",
    "warmup_steps": "Warmup steps",
    "weight_decay": "Weight decay",
    "grad_accumulation_steps": "Gradient accumulation steps",
}

HP_NAMES_TO_PLOT = [
    "lr",
    "weight_decay",
    "beta1",
    "beta2",
    "warmup_steps",
    "cooldown_steps",
    "n_tokens",
    "d_model",
    "n_layers",
    "n_heads",
    "micro_batch_size",
    "grad_accumulation_steps",
]


def _safe_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _parse_pareto_configs(pareto_csv_path: Path) -> pd.DataFrame:
    """Parse pareto CSV and extract config JSON into separate columns."""
    pareto_df = pd.read_csv(pareto_csv_path)
    if "config" not in pareto_df.columns:
        return pareto_df

    parsed_configs = []
    for raw_cfg in pareto_df["config"]:
        if not isinstance(raw_cfg, str):
            parsed_configs.append({})
            continue
        try:
            parsed_configs.append(json.loads(raw_cfg))
        except Exception:
            parsed_configs.append({})

    cfg_df = pd.DataFrame(parsed_configs)
    for col in cfg_df.columns:
        pareto_df[f"hp.{col}"] = cfg_df[col]
    return pareto_df


def _find_loss_column(df: pd.DataFrame) -> tuple[str | None, str]:
    """Find test loss or validation loss column.
    
    Returns:
        Tuple of (column_name, description)
    """
    candidates = [
        ("extra.test_loss", "test_loss"),
        ("test_loss", "test_loss"),
        ("extra.last_test_loss", "last_test_loss"),
        ("extra.valid_loss", "valid_loss"),
        ("valid_loss", "valid_loss"),
        ("val_loss", "val_loss"),
        ("validation_loss", "validation_loss"),
        ("extra.last_valid_loss", "last_valid_loss"),
        ("extra.train_loss", "train_loss"),
    ]
    
    for col, desc in candidates:
        if col in df.columns:
            return col, desc
    
    return None, "unknown_loss"


def _hp_bounds_from_full_csv(full_csv_path: Path, hp_names: Iterable[str]) -> dict[str, tuple[float, float]]:
    if not full_csv_path.exists():
        return {}

    full_df = pd.read_csv(full_csv_path)
    bounds: dict[str, tuple[float, float]] = {}
    for hp_name in hp_names:
        col_name = f"config.{hp_name}"
        if col_name not in full_df.columns:
            continue
        values = _safe_numeric(full_df[col_name])
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        bounds[hp_name] = (float(np.min(values)), float(np.max(values)))
    return bounds


def _should_use_log_x(values: np.ndarray) -> tuple[bool, str]:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return False, "too_few_points"
    if np.any(finite <= 0):
        return False, "contains_non_positive"

    q05, q95 = np.quantile(finite, [0.05, 0.95])
    if not np.isfinite(q05) or not np.isfinite(q95) or q05 <= 0 or q95 <= 0:
        return False, "invalid_quantiles"

    spread_ratio = q95 / q05
    if spread_ratio < 30:
        return False, f"low_spread_ratio={spread_ratio:.2f}"

    as_int = np.allclose(finite, np.round(finite))
    if as_int and finite.max() <= 256:
        return False, "small_integer_range"

    return True, f"spread_ratio={spread_ratio:.2f}"


def _should_use_log_y(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return False
    if np.any(finite <= 0):
        return False

    q05, q95 = np.quantile(finite, [0.05, 0.95])
    if not np.isfinite(q05) or not np.isfinite(q95) or q05 <= 0 or q95 <= 0:
        return False

    spread_ratio = q95 / q05
    return spread_ratio >= 30


def _plot_pareto_hp(
    pareto_df: pd.DataFrame,
    full_df: pd.DataFrame,
    hp_name: str,
    loss_col: str | None,
    loss_label: str,
    hp_bounds: dict[str, tuple[float, float]],
    out_dir: Path,
) -> tuple[bool, str]:
    hp_col = f"hp.{hp_name}"
    if hp_col not in pareto_df.columns:
        return False, f"{hp_col}_not_in_pareto_df"

    flops = _safe_numeric(pareto_df["flops"])
    hp = _safe_numeric(pareto_df[hp_col])
    
    if loss_col is not None and loss_col in pareto_df.columns:
        loss = _safe_numeric(pareto_df[loss_col])
    else:
        # Fallback: try to get from full.csv by matching trial_id
        loss = np.full_like(flops, np.nan)
        if "trial_id" in pareto_df.columns and loss_col is not None:
            trial_ids = pareto_df["trial_id"].values
            if "id" in full_df.columns:
                full_trial_ids = full_df["id"].values
                for i, trial_id in enumerate(trial_ids):
                    try:
                        trial_id_int = int(trial_id)
                        matches = full_df[full_df["id"] == trial_id_int]
                        if not matches.empty and loss_col in matches.columns:
                            loss[i] = pd.to_numeric(matches.iloc[0][loss_col], errors="coerce")
                    except (ValueError, TypeError):
                        pass

    base_mask = np.isfinite(flops) & np.isfinite(hp) & (flops > 0)
    
    if hp_name in {"lr", "weight_decay", "beta1", "beta2", "warmup_steps", "cooldown_steps", "n_tokens"}:
        base_mask &= hp > 0

    if base_mask.sum() < 2:
        return False, "not_enough_valid_points"

    xv = flops[base_mask]
    yv = hp[base_mask]
    lv = loss[base_mask]

    fig, ax = plt.subplots(figsize=(10, 4.8))

    finite_loss = lv[np.isfinite(lv)]
    if finite_loss.size >= 2 and float(np.nanmin(finite_loss)) < float(np.nanmax(finite_loss)):
        norm = mcolors.Normalize(vmin=float(np.nanmin(finite_loss)), vmax=float(np.nanmax(finite_loss)))
        scatter = ax.scatter(xv, yv, c=lv, cmap="viridis", norm=norm, alpha=0.75, s=30)
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label(f"Loss ({loss_label})", rotation=90, labelpad=20)
    else:
        ax.scatter(xv, yv, alpha=0.6, s=30, color="C0")

    ax.set_xscale("log")
    
    use_log_y = hp_name == "lr" or _should_use_log_y(yv)
    if use_log_y:
        ax.set_yscale("log")

    label = HP_LABELS.get(hp_name, hp_name.replace("_", " ").title())
    ax.set_xlabel("Compute (FLOPs)")
    ax.set_ylabel(label)
    ax.set_title(f"Pareto: {label} vs FLOPs")
    ax.grid(True, linestyle="--", alpha=0.35, which="both")

    if hp_name in hp_bounds:
        lower, upper = hp_bounds[hp_name]
        draw_lower = np.isfinite(lower) and (lower > 0 if use_log_y else True)
        draw_upper = np.isfinite(upper) and (upper > 0 if use_log_y else True)
        if draw_lower:
            ax.axhline(lower, linestyle=":", linewidth=1.5, alpha=0.7, label=f"lower bound = {lower:g}")
        if draw_upper:
            ax.axhline(upper, linestyle=":", linewidth=1.5, alpha=0.7, label=f"upper bound = {upper:g}")
        if draw_lower and draw_upper and lower < upper:
            ax.axhspan(lower, upper, alpha=0.10, label="full search space")

    x_min, x_max = np.min(xv), np.max(xv)
    y_min, y_max = np.min(yv), np.max(yv)
    
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="C0", markersize=6, label="Pareto points"),
        Line2D([0], [0], linestyle="", color="none", label=f"Points: {len(xv)}"),
        Line2D([0], [0], linestyle="", color="none", label=f"{label} range: [{y_min:.4g}, {y_max:.4g}]"),
        Line2D([0], [0], linestyle="", color="none", label=f"FLOPs range: [{x_min:.4g}, {x_max:.4g}]"),
    ]
    
    # ax.legend(
    #     handles=legend_handles,
    #     loc="center left",
    #     bbox_to_anchor=(1.30, 0.5),
    #     frameon=True,
    #     fontsize=9,
    #     borderaxespad=0.0,
    # )

    # fig.subplots_adjust(right=0.62)
    fig.tight_layout()
    
    output_path = out_dir / f"hp_vs_loss_{_sanitize_filename(label)}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return True, output_path.name


def generate_pareto_hp_plots(
    summary_dir: Path,
    pareto_csv_path: Path,
    full_csv_path: Path,
    out_dir: Path,
) -> None:
    """Generate HP vs FLOPs plots for pareto front with loss coloring."""
    if not summary_dir.is_dir():
        raise NotADirectoryError(f"Summary directory not found: {summary_dir}")
    if not pareto_csv_path.exists():
        raise FileNotFoundError(f"pareto_front_configs.csv not found: {pareto_csv_path}")
    if not full_csv_path.exists():
        raise FileNotFoundError(f"full.csv not found: {full_csv_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    pareto_df = _parse_pareto_configs(pareto_csv_path)
    full_df = pd.read_csv(full_csv_path)
    
    loss_col, loss_label = _find_loss_column(full_df)
    if loss_col is None:
        print(f"Warning: Could not find test_loss or valid_loss columns. Plotting without color dimension.")
        loss_label = "unknown_loss"
    
    hp_bounds = _hp_bounds_from_full_csv(full_csv_path, HP_NAMES_TO_PLOT)
    
    created = 0
    skipped = 0
    
    for hp_name in HP_NAMES_TO_PLOT:
        ok, details = _plot_pareto_hp(
            pareto_df=pareto_df,
            full_df=full_df,
            hp_name=hp_name,
            loss_col=loss_col,
            loss_label=loss_label,
            hp_bounds=hp_bounds,
            out_dir=out_dir,
        )
        
        if ok:
            created += 1
            print(f"[saved] {details}")
        else:
            skipped += 1
            print(f"[skip] {hp_name}: {details}")

    print(
        f"\nDone. Created {created} plots, skipped {skipped}. "
        f"Output directory: {out_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot pareto front hyperparameters (y-axis) against FLOPs (x-axis) "
            "with loss coloring, from pareto_front_configs.csv."
        )
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=DEFAULT_SUMMARY_DIR,
        help="Directory containing summary files (default: neps/neps_configs/neps_1000_configs/summary)",
    )
    parser.add_argument(
        "--pareto-csv",
        type=Path,
        default=None,
        help="Path to pareto_front_configs.csv (default: <summary-dir>/pareto_front_configs.csv)",
    )
    parser.add_argument(
        "--full-csv",
        type=Path,
        default=None,
        help="Path to full.csv (default: <summary-dir>/full.csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNG plots (default: <summary-dir>/pareto_hp_plots)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_dir = args.summary_dir
    pareto_csv_path = args.pareto_csv if args.pareto_csv is not None else summary_dir / "pareto_front_configs.csv"
    full_csv_path = args.full_csv if args.full_csv is not None else summary_dir / "full.csv"
    out_dir = args.out_dir if args.out_dir is not None else summary_dir / "pareto_hp_plots"

    generate_pareto_hp_plots(
        summary_dir=summary_dir,
        pareto_csv_path=pareto_csv_path,
        full_csv_path=full_csv_path,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
