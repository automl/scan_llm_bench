from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

ROOT_DIRECTORY = Path(__file__).resolve().parents[1] / "neps" / "neps_configs" / "neps_1000_configs"
DEFAULT_SUMMARY_DIR = ROOT_DIRECTORY / "summary"

HP_LABELS = {
    "config.d_model": "Embedding size",
    "config.n_layers": "Number of layers",
    "config.n_heads": "Number of heads",
    "config.n_tokens": "Training tokens",
    "config.micro_batch_size": "Micro batch size",
    "config.grad_accumulation_steps": "Gradient accumulation steps",
    "config.lr": "Learning rate",
    "config.weight_decay": "Weight decay",
    "config.beta1": "Beta 1",
    "config.beta2": "Beta 2",
    "config.warmup_steps": "Warmup steps",
    "config.cooldown_steps": "Cooldown steps",
    "config.steps_budget": "Step budget",
    "config.total_params": "Total parameters",
    "config.total_flops": "Total FLOPs",
    "config.recommended_num_gpus": "Recommended GPUs",
    "config.valid_tokens": "Validation tokens",
    "config.seq_len": "Sequence length",
    "config.vocab_size": "Vocab size",
    "config.seed": "Seed",
}

HP_CATEGORY_COLUMNS = [
    "config.beta1",
    "config.beta2",
    "config.cooldown_steps",
    "config.d_model",
    "config.lr",
    "config.micro_batch_size",
    "config.n_heads",
    "config.n_layers",
    "config.resume_step",
    "config.total_flops",
    "config.total_params",
    "config.n_tokens",
    "config.valid_tokens",
    "config.weight_decay",
]


def _safe_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _find_objective_column(df: pd.DataFrame) -> str | None:
    preferred = [
        "objective_to_minimize",
        "objective",
        "result.objective",
        "extra.test_loss",
        "test_loss",
        "extra.last_test_loss",
        "extra.valid_loss",
        "valid_loss",
        "val_loss",
        "validation_loss",
        "extra.last_valid_loss",
        "loss",
    ]
    for col in preferred:
        if col in df.columns:
            return col
    return None


def _infer_config_id(row: pd.Series) -> str | None:
    for key in ("location", "reported_as", "config.resume_exp_name"):
        value = row.get(key)
        if isinstance(value, str):
            match = re.search(r"config_(\d+)", value)
            if match:
                return match.group(1)

    for key in ("id", "config_id", "trial_id"):
        value = row.get(key)
        if pd.isna(value):
            continue
        try:
            return str(int(float(value)))
        except Exception:
            if isinstance(value, str):
                match = re.search(r"(\d+)", value)
                if match:
                    return match.group(1)
    return None


def _objective_from_report(report_path: Path) -> float | None:
    if not report_path.exists():
        return None
    try:
        with open(report_path, "r", encoding="utf-8") as handle:
            report = yaml.safe_load(handle) or {}
    except Exception:
        return None

    top_level_candidates = ["objective_to_minimize", "objective", "loss", "test_loss", "valid_loss"]
    for key in top_level_candidates:
        if key in report:
            val = pd.to_numeric(pd.Series([report[key]]), errors="coerce").iloc[0]
            if pd.notna(val):
                return float(val)

    extra = report.get("extra", {})
    if isinstance(extra, dict):
        extra_candidates = [
            "test_loss",
            "last_test_loss",
            "valid_loss",
            "last_valid_loss",
            "train_loss",
        ]
        for key in extra_candidates:
            if key in extra:
                val = pd.to_numeric(pd.Series([extra[key]]), errors="coerce").iloc[0]
                if pd.notna(val):
                    return float(val)
    return None


def _resolve_objective_series(df: pd.DataFrame, summary_dir: Path) -> tuple[pd.Series, str]:
    selected_col = _find_objective_column(df)

    if selected_col is None:
        objective_series = pd.Series(np.nan, index=df.index, dtype=float)
        selected_label = "objective_from_report"
    else:
        objective_series = pd.to_numeric(df[selected_col], errors="coerce")
        selected_label = selected_col

    needs_fallback = selected_col is None or objective_series.isna().any()
    if not needs_fallback:
        return objective_series, selected_label

    reports_root = summary_dir.parent / "configs"
    cache: dict[str, float | None] = {}

    missing_idx = objective_series.index if selected_col is None else objective_series[objective_series.isna()].index
    for idx in missing_idx:
        config_id = _infer_config_id(df.loc[idx])
        if config_id is None:
            continue

        if config_id not in cache:
            report_path = reports_root / f"config_{config_id}" / "report.yaml"
            cache[config_id] = _objective_from_report(report_path)

        value = cache[config_id]
        if value is not None:
            objective_series.at[idx] = value

    if objective_series.notna().sum() == 0:
        raise ValueError(
            "Could not find objective/loss values in full.csv or report.yaml files. "
            "Tried objective_to_minimize, objective, and test/valid loss fallbacks."
        )

    if selected_col is None:
        return objective_series, "objective_from_report"

    return objective_series, f"{selected_col} (+report_fallback)"


def _candidate_hp_columns(df: pd.DataFrame, objective_col: str) -> list[str]:
    category_cols = [c for c in HP_CATEGORY_COLUMNS if c in df.columns]
    if category_cols:
        return category_cols

    ignored = {
        objective_col,
        "config_id",
        "id",
        "status",
        "state",
        "trial_id",
        "seed",
    }

    config_cols = [c for c in df.columns if c.startswith("config.") and c not in ignored]
    if config_cols:
        return config_cols

    numeric_cols = []
    for col in df.columns:
        if col in ignored:
            continue
        values = _safe_numeric(df[col])
        if np.isfinite(values).sum() >= 5:
            numeric_cols.append(col)
    return numeric_cols


def _resolve_flops_series(df: pd.DataFrame) -> pd.Series:
    candidates = [
        "config.total_flops",
        "extra.flops",
        "cost",
        "flops",
        "total_flops",
    ]
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _should_use_log_x(values: np.ndarray) -> tuple[bool, str]:
    finite = values[np.isfinite(values)]
    if finite.size < 10:
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


def _plot_single_hp(
    df: pd.DataFrame,
    hp_col: str,
    objective_values: pd.Series,
    flops_values: pd.Series,
    out_dir: Path,
) -> tuple[bool, str]:
    x = _safe_numeric(df[hp_col])
    y = _safe_numeric(objective_values)
    flops = _safe_numeric(flops_values)

    base_mask = np.isfinite(x) & np.isfinite(y)
    if base_mask.sum() < 10:
        return False, "not_enough_valid_points"

    use_log_x, _reason = _should_use_log_x(x[base_mask])
    mask = base_mask & (x > 0) if use_log_x else base_mask
    if mask.sum() < 10:
        return False, "not_enough_points_after_log_filter"

    xv = x[mask]
    yv = y[mask]
    fv = flops[mask]

    fig, ax = plt.subplots(figsize=(10, 4.8))

    finite_pos_flops = fv[np.isfinite(fv) & (fv > 0)]
    if finite_pos_flops.size >= 2 and float(np.nanmin(finite_pos_flops)) < float(np.nanmax(finite_pos_flops)):
        norm = mcolors.LogNorm(vmin=float(np.nanmin(finite_pos_flops)), vmax=float(np.nanmax(finite_pos_flops)))
        scatter = ax.scatter(xv, yv, c=fv, cmap="viridis", norm=norm, alpha=0.75, s=18)
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Total FLOPs")
        cbar.ax.yaxis.labelpad = 12
    else:
        ax.scatter(xv, yv, alpha=0.6, s=18, color="C0")

    if use_log_x:
        ax.set_xscale("log")

    x_min, x_max = np.min(xv), np.max(xv)
    y_min, y_max = np.min(yv), np.max(yv)

    label = HP_LABELS.get(hp_col, hp_col.replace("config.", "").replace("_", " ").title())
    ax.set_xlabel(label)
    ax.set_ylabel("Objective to minimize")
    ax.set_title(f"{label} vs Objective to minimize")
    ax.grid(True, linestyle="--", alpha=0.35, which="both")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="C0", markersize=6, label="Configs"),
        Line2D([0], [0], linestyle="", color="none", label=f"Points: {len(xv)}"),
        Line2D([0], [0], linestyle="", color="none", label=f"{label} range: [{x_min:.4g}, {x_max:.4g}]"),
        Line2D([0], [0], linestyle="", color="none", label=f"Objective range: [{y_min:.4g}, {y_max:.4g}]"),
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


def generate_hp_vs_loss_plots(summary_dir: Path, full_csv_path: Path, out_dir: Path) -> None:
    if not summary_dir.is_dir():
        raise NotADirectoryError(f"Summary directory not found: {summary_dir}")
    if not full_csv_path.exists():
        raise FileNotFoundError(f"full.csv not found: {full_csv_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(full_csv_path)
    objective_values, _objective_label = _resolve_objective_series(df, summary_dir)
    flops_values = _resolve_flops_series(df)
    hp_cols = _candidate_hp_columns(df, "objective_to_minimize")
    if not hp_cols:
        raise ValueError("No numeric hyperparameter columns found in full.csv")

    created = 0
    skipped = 0
    for hp_col in hp_cols:
        ok, details = _plot_single_hp(df, hp_col, objective_values, flops_values, out_dir)
        if ok:
            created += 1
            print(f"[saved] {details}")
        else:
            skipped += 1
            print(f"[skip] {hp_col}: {details}")

    print(
        f"Done. Created {created} plots, skipped {skipped}. "
        f"Output directory: {out_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot full NePS configs hyperparameters (x-axis) against objective/loss (y-axis) "
            "from already-generated summary files."
        )
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=DEFAULT_SUMMARY_DIR,
        help="Directory containing generated summary files (default: neps/neps_configs/neps_1000_configs/summary)",
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
        help="Output directory for PNG plots (default: <summary-dir>/hp_vs_loss_plots)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_dir = args.summary_dir
    full_csv_path = args.full_csv if args.full_csv is not None else summary_dir / "full.csv"
    out_dir = args.out_dir if args.out_dir is not None else summary_dir / "hp_vs_loss_plots"

    generate_hp_vs_loss_plots(summary_dir=summary_dir, full_csv_path=full_csv_path, out_dir=out_dir)


if __name__ == "__main__":
    main()
