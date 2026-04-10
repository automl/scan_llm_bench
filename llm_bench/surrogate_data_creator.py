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

import numpy as np
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
    # "grad_clip",
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
    test_losses = extra.get("test_losses") or rpt.get("test_losses")
    eval_steps = extra.get("eval_steps") or rpt.get("eval_steps")

    # Fall back to top‑level lists if stored there
    if valid_losses is None:
        valid_losses = rpt.get("valid_losses")
    if test_losses is None:
        test_losses = rpt.get("test_losses")
    if eval_steps is None:
        eval_steps = rpt.get("eval_steps")

    if not valid_losses or not eval_steps or not test_losses:
        print(f"  [WARN] {config_dir.name}: Missing valid_losses, eval_steps, or test_losses, skipping")
        return None
    if len(valid_losses) != len(eval_steps) and len(test_losses) != len(eval_steps):
        print(f"  [WARN] {config_dir.name}: valid_losses ({len(valid_losses)}) "
              f"!= eval_steps ({len(eval_steps)}) != test_losses ({len(test_losses)}), skipping")
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
    train_array = extra.get("train_loss_array", [])

    # The lr/tokens arrays are logged at log_every_steps.
    # Map validation steps to the closest index in these arrays.
    log_every = cfg.get("log_every_steps", 50)

    rows: list[dict] = []
    for i, (eval_step, v_loss, t_loss) in enumerate(zip(eval_steps, valid_losses, test_losses)):
        row = dict(base_row)

        # Approximate index into lr/tokens arrays
        if lr_array:
            idx = _closest_index(eval_step, log_every, len(lr_array))
            row["current_lr"] = lr_array[idx]
        else:
            row["current_lr"] = None

        if tokens_array:
            idx = _closest_index(eval_step, log_every, len(tokens_array))
            row["tokens_so_far"] = tokens_array[idx]
        else:
            row["tokens_so_far"] = None
            
        if train_array:
            idx = _closest_index(eval_step, log_every, len(train_array))
            row["train_loss_so_far"] = train_array[idx]
        else:
            row["train_loss_so_far"] = None

        # Estimated FLOPs up to this step
        if total_cost and max_steps and max_steps > 0:
            row["flops_so_far"] = eval_step / max_steps * total_cost
        else:
            row["flops_so_far"] = None

        row["eval_step"] = eval_step
        row["valid_loss"] = v_loss
        row["test_loss"] = t_loss
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
    test_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Randomly split configs into train/test, keeping all rows of a config together.
    """
    config_ids = df["config_id"].unique().tolist()
    random.seed(seed)
    random.shuffle(config_ids)

    n = len(config_ids)
    n_train = max(1, round(n * train_ratio))
    n_test = n - n_train
    if n_test < 1:
        n_train -= 1
        n_test = 1

    train_ids = set(config_ids[:n_train])
    test_ids = set(config_ids[n_train:])

    df_train = df[df["config_id"].isin(train_ids)].reset_index(drop=True)
    df_test = df[df["config_id"].isin(test_ids)].reset_index(drop=True)

    return df_train, df_test


def split_by_config_log_flops_stratified(
    df: pd.DataFrame,
    train_ratio: float = 0.90,
    test_ratio: float = 0.10,
    seed: int = 42,
    n_bins: int = 10,
    min_test_per_bin: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test split that ensures balanced coverage of the FLOP range.

    Strategy
    --------
    1. Compute each config's total FLOPs (total_compute).  Configs with
       missing / non-positive FLOPs fall into a dedicated "unknown" bin.
    2. Create *n_bins* equal-width bins in log10(FLOPs) space that span the
       observed range.  This keeps configs from different decades (10^16,
       10^17, 10^18, …) in separate bins so each decade is sampled.
    3. Within each bin, sample approx *test_ratio* configs for the test set
       (at least *min_test_per_bin* if the bin is non-empty).  The rest go
       to train.
    4. Prints a bin summary so you can verify every FLOPs decade is covered.
    """
    assert abs(train_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    rng = np.random.default_rng(seed)

    # Per-config representative FLOPs (constant across rows of the same config)
    per_cfg = (
        df.groupby("config_id")["total_compute"]
        .first()
        .reset_index()
        .rename(columns={"total_compute": "flops"})
    )

    valid_mask = per_cfg["flops"].notna() & (per_cfg["flops"] > 0)
    valid_cfg = per_cfg[valid_mask].copy()
    unknown_cfg = per_cfg[~valid_mask].copy()

    train_ids: list[str] = []
    test_ids: list[str] = []

    if not valid_cfg.empty:
        log_flops = np.log10(valid_cfg["flops"].to_numpy(dtype=float))
        bin_edges = np.linspace(log_flops.min(), log_flops.max(), n_bins + 1)
        # Widen the last edge slightly so the maximum value falls inside
        bin_edges[-1] += 1e-9
        bin_labels = np.digitize(log_flops, bin_edges) - 1  # 0-based
        valid_cfg = valid_cfg.copy()
        valid_cfg["_bin"] = bin_labels
        valid_cfg["_log10_flops"] = log_flops

        print(f"\nLog10-FLOPs stratified split  ({n_bins} bins, seed={seed})")
        print(f"  Range: 10^{log_flops.min():.2f} – 10^{log_flops.max():.2f}")
        print(f"  {'Bin':>4}  {'Range':>24}  {'Configs':>8}  {'→ test':>8}")

        for b in range(n_bins):
            bin_cfg_ids = valid_cfg.loc[valid_cfg["_bin"] == b, "config_id"].tolist()
            if not bin_cfg_ids:
                continue
            rng.shuffle(bin_cfg_ids)
            n_test_bin = max(min_test_per_bin, round(len(bin_cfg_ids) * test_ratio))
            n_test_bin = min(n_test_bin, len(bin_cfg_ids))
            test_ids.extend(bin_cfg_ids[:n_test_bin])
            train_ids.extend(bin_cfg_ids[n_test_bin:])
            lo = bin_edges[b]
            hi = bin_edges[b + 1]
            print(
                f"  {b:>4}  [10^{lo:5.2f}, 10^{hi:5.2f})  "
                f"{len(bin_cfg_ids):>8}  {n_test_bin:>8}"
            )
    else:
        print("[WARN] No configs with valid total_compute; falling back to random split.")

    # Configs with unknown FLOPs: distribute proportionally (random)
    if not unknown_cfg.empty:
        unk_ids = unknown_cfg["config_id"].tolist()
        rng.shuffle(unk_ids)
        n_unk_test = max(1, round(len(unk_ids) * test_ratio))
        test_ids.extend(unk_ids[:n_unk_test])
        train_ids.extend(unk_ids[n_unk_test:])
        print(f"  [unknown FLOPs] {len(unk_ids)} configs → {n_unk_test} test")

    # Fallback: if somehow both lists are empty, use all-train
    if not train_ids and not test_ids:
        train_ids = df["config_id"].unique().tolist()

    print(
        f"  Total: {len(train_ids)} train configs, {len(test_ids)} test configs\n"
    )

    train_id_set = set(train_ids)
    test_id_set = set(test_ids)

    df_train = df[df["config_id"].isin(train_id_set)].reset_index(drop=True)
    df_test = df[df["config_id"].isin(test_id_set)].reset_index(drop=True)

    return df_train, df_test


def main():
    parser = argparse.ArgumentParser(
        description="Create surrogate CSV data from NePS config/report directories."
    )
    parser.add_argument(
        "--configs_dir",
        type=str,
        # required=True,
        default="C:/Freiburg/Tema/Scaling_Law/scan_llm_bench/neps/neps_configs/neps_1000_configs/configs",
        help="Path to the configs/ directory containing config_* subdirs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="C:/Freiburg/Tema/Scaling_Law/scan_llm_bench/surrogate/new_surrogate_data_stratified",
        help="Directory where train.csv, test.csv will be written.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.90)
    # parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split_strategy",
        choices=["random", "stratified_log_flops"],
        default="stratified_log_flops",
        help="How to split configs: random shuffle or stratified by log10(FLOPs).",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=4,
        help="Number of log10-FLOPs bins for stratified split.",
    )
    parser.add_argument(
        "--min_test_per_bin",
        type=int,
        default=1,
        help="Minimum configs per bin assigned to test in stratified split.",
    )
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir)
    if not configs_dir.is_dir():
        raise FileNotFoundError(f"Configs directory not found: {configs_dir}")

    df = build_dataframe(configs_dir)
    if df.empty:
        print("No data found, nothing to write.")
        return

    if args.split_strategy == "stratified_log_flops":
        df_train, df_test = split_by_config_log_flops_stratified(
            df,
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            n_bins=args.n_bins,
            min_test_per_bin=args.min_test_per_bin,
        )
    else:
        df_train, df_test = split_by_config(
            df,
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, split_df in [("train", df_train), ("test", df_test)]:
        path = out / f"{name}.csv"
        split_df.to_csv(path, index=False)
        n_configs = split_df["config_id"].nunique()
        print(f"Wrote {path}  ({len(split_df)} rows, {n_configs} configs)")


if __name__ == "__main__":
    main()
