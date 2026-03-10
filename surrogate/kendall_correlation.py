import argparse

import pandas as pd

from surrogate.metrics import calculate_correlation_matrix
from surrogate.utils import plot_correlation_matrix


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv_path",
        type=str,
        default="splits/data.csv",
        help="Path to CSV file."
    )

    parser.add_argument(
        "--cols",
        nargs="+",
        default=["clip_val_loss", "clip_test_loss"],
        help="Columns to compute Kendall tau correlation for."
    )

    parser.add_argument(
        "--out",
        default="corr_matrix.png",
        help="Output heatmap"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df = df[args.cols].dropna()

    data_dict = {}
    for col in args.cols:
        data_dict[col] = df[col].tolist()

    corr_mat, keys = calculate_correlation_matrix(data_dict)
    plot_correlation_matrix(corr_mat, keys, args.out)


if __name__ == "__main__":
    main()