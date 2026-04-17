import argparse

import pandas as pd

from llm_bench.metrics import calculate_correlation_matrix
from llm_bench.utils import plot_correlation_matrix

DOWNSTREAM_TASKS = ["hellaswag", "arc_challenge", "arc_easy", "copa", "piqa", "openbookqa"]

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out", default="llm_bench/plots/corr_matrix.png")

    args = parser.parse_args()

    train_df = pd.read_csv("llm_bench/surrogate_data/train.csv")
    test_df = pd.read_csv("llm_bench/surrogate_data/test.csv")

    df = pd.concat([train_df, test_df], ignore_index=True)

    df["config_id"] = pd.to_numeric(df["config_id"].str.split("_").str[1])
    df["epoch"] = pd.to_numeric(df["eval_step"])

    df = df.sort_values(["config_id", "eval_step"]).reset_index(drop=True)

    downstream_cols = []
    for dataset in DOWNSTREAM_TASKS:
        col_name = f"{dataset}_acc"
        if col_name in df.columns:
            downstream_cols.append(col_name)

    base_cols = ["valid_loss", "test_loss"]
    cols = base_cols + downstream_cols
    df = df[cols]

    for col in downstream_cols:
        df[col] = 1 - df[col]

    df = df.dropna()
    data_dict = {col: df[col].tolist() for col in cols}
    corr_mat, keys = calculate_correlation_matrix(data_dict)
    plot_correlation_matrix(corr_mat, keys, args.out)


if __name__ == "__main__":
    main()