"""
Creates train/validation/test CSV files from NePS config/report directories.

For each config, reads config.yaml and report.yaml, then produces one row per
validation step (valid_losses x valid_steps). All rows of a given config stay
together in the same split file.

Usage:
    python scripts/surrogate_data_creator.py \
        --configs_dir neps/neps_configs/neps_full_search_space/configs \
        --output_dir scripts/surrogate_data \
        --train_ratio 0.90 --val_ratio 0.05 --test_ratio 0.05 \
        --seed 42
"""

import argparse
import os
import random
from pathlib import Path

import pandas as pd
import yaml


# From config.yaml (constant across all rows of a config)
CONFIG_COLS = [
    "d_model",
    "n_layers",
    "n_heads",
    "weight_decay",
    "beta1",
    "beta2",
    "grad_clip",
    "warmup_steps",
    "cooldown_steps",
]

# From report.yaml top‑level / extra (constant across all rows of a config)
REPORT_CONST_COLS = [
    "global_batch_size",
    "final_step",
    "total_compute",
    "n_data",
    "n_param",
    "micro_batch_size",
]


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_config(config_dir: Path) -> list[dict] | None:
    """Return a list of row-dicts for one config directory, or None on error."""
    config_path = config_dir / "config.yaml"
    report_path = config_dir / "report.yaml"

    if not config_path.exists() or not report_path.exists():
        print(f"  [WARN] {config_dir.name}: Missing config.yaml or report.yaml, skipping")
        return None

    cfg = load_yaml(config_path)
    rpt = load_yaml(report_path)

    if rpt is None or rpt.get("reported_as") != "success":
        print(f"  [WARN] {config_dir.name}: Report not successful or empty, skipping")
        return None

    extra = rpt.get("extra", {})
    if extra is None:
        print(f"  [WARN] {config_dir.name}: Missing extra section in report, skipping")
        return None

    valid_losses = extra.get("valid_losses") or rpt.get("valid_losses")
    valid_steps = extra.get("valid_steps") or rpt.get("valid_steps")

    # Fall back to top‑level lists if stored there
    if valid_losses is None:
        valid_losses = rpt.get("valid_losses")
    if valid_steps is None:
        valid_steps = rpt.get("valid_steps")

    if not valid_losses or not valid_steps:
        print(f"  [WARN] {config_dir.name}: Missing valid_losses or valid_steps, skipping")
        return None
    if len(valid_losses) != len(valid_steps):
        print(f"  [WARN] {config_dir.name}: valid_losses ({len(valid_losses)}) "
              f"!= valid_steps ({len(valid_steps)}), skipping")
        return None

    config_id = config_dir.name  # "config_1"

    base_row: dict = {"config_id": config_id}

    for col in CONFIG_COLS:
        base_row[col] = cfg.get(col)

    base_row["initial_lr"] = cfg.get("lr")

    base_row["global_batch_size"] = extra.get("global_batch_size")
    base_row["final_step"] = extra.get("final_step")
    base_row["total_compute"] = rpt.get("cost")  # top‑level cost
    base_row["n_data"] = extra.get("n_data")
    base_row["n_param"] = extra.get("n_param")
    base_row["micro_batch_size"] = extra.get("micro_batch_size")

    # Steps budget (max_steps) for flops estimation
    max_steps = extra.get("final_step") or cfg.get("steps_budget")
    total_cost = rpt.get("cost")  # total FLOPs

    # LR and tokens arrays for interpolation
    lr_array = extra.get("lr", [])
    tokens_array = extra.get("tokens", [])

    # The lr/tokens arrays are logged at log_every_steps.
    # Map validation steps to the closest index in these arrays.
    log_every = cfg.get("log_every_steps", 50)

    rows: list[dict] = []
    for i, (v_step, v_loss) in enumerate(zip(valid_steps, valid_losses)):
        row = dict(base_row)

        # Approximate index into lr/tokens arrays
        if lr_array:
            idx = _closest_index(v_step, log_every, len(lr_array))
            row["current_lr"] = lr_array[idx]
        else:
            row["current_lr"] = None

        if tokens_array:
            idx = _closest_index(v_step, log_every, len(tokens_array))
            row["tokens_so_far"] = tokens_array[idx]
        else:
            row["tokens_so_far"] = None

        # Estimated FLOPs up to this step
        if total_cost and max_steps and max_steps > 0:
            row["flops_so_far"] = v_step / max_steps * total_cost
        else:
            row["flops_so_far"] = None

        row["valid_step"] = v_step
        row["valid_loss"] = v_loss

        rows.append(row)

    return rows


def _closest_index(valid_step: int, log_every: int, array_len: int) -> int:
    """Map a validation step to the closest index in a log-every-spaced array."""
    idx = round(valid_step / log_every) - 1
    return max(0, min(idx, array_len - 1))


def build_dataframe(configs_dir: Path) -> pd.DataFrame:
    """Walk all config_* directories and build a unified DataFrame."""
    all_rows: list[dict] = []
    config_dirs = sorted(
        [d for d in configs_dir.iterdir() if d.is_dir() and d.name.startswith("config_")],
        key=lambda p: int(p.name.split("_")[-1]),
    )

    print(f"Found {len(config_dirs)} config directories in {configs_dir}")
    skipped = 0
    for cdir in config_dirs:
        rows = parse_config(cdir)
        if rows is None:
            skipped += 1
            continue
        all_rows.extend(rows)

    print(f"Parsed {len(config_dirs) - skipped} configs ({skipped} skipped), "
          f"total rows: {len(all_rows)}")

    return pd.DataFrame(all_rows)


def split_by_config(
    df: pd.DataFrame,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into train/val/test ensuring all rows of a config remain in the same split.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    config_ids = df["config_id"].unique().tolist()
    random.seed(seed)
    random.shuffle(config_ids)

    n = len(config_ids)
    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_train -= 1
        n_test = 1

    train_ids = set(config_ids[:n_train])
    val_ids = set(config_ids[n_train : n_train + n_val])
    test_ids = set(config_ids[n_train + n_val :])

    df_train = df[df["config_id"].isin(train_ids)].reset_index(drop=True)
    df_val = df[df["config_id"].isin(val_ids)].reset_index(drop=True)
    df_test = df[df["config_id"].isin(test_ids)].reset_index(drop=True)

    return df_train, df_val, df_test


def main():
    parser = argparse.ArgumentParser(
        description="Create surrogate CSV data from NePS config/report directories."
    )
    parser.add_argument(
        "--configs_dir",
        type=str,
        required=True,
        default="neps/neps_configs/neps_full_search_space/configs",
        help="Path to the configs/ directory containing config_* subdirs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="surrogate_data",
        help="Directory where train.csv, val.csv, test.csv will be written.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.90)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir)
    if not configs_dir.is_dir():
        raise FileNotFoundError(f"Configs directory not found: {configs_dir}")

    df = build_dataframe(configs_dir)
    if df.empty:
        print("No data found, nothing to write.")
        return

    df_train, df_val, df_test = split_by_config(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        path = out / f"{name}.csv"
        split_df.to_csv(path, index=False)
        n_configs = split_df["config_id"].nunique()
        print(f"Wrote {path}  ({len(split_df)} rows, {n_configs} configs)")


if __name__ == "__main__":
    main()
