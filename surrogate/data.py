import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from surrogate.utils import convert_width_to_one_hot


class TorchSurrogateDataset(Dataset):
    def __init__(self, csv_path: str, target_col: str):
        self.df = pd.read_csv(csv_path)
        self.target_col = target_col

        self.feature_cols = [
            "lr", "wd", "beta1", "beta2", "eps", "warmup_fraction",
            "vision_width", "text_width", "global_batch_size",
            "total_samples_planned", "training_progress", "lr_ratio",
        ]
        missing = [c for c in self.feature_cols + [target_col] if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        numeric = torch.tensor([
            float(row["lr"]),
            float(row["wd"]),
            float(row["beta1"]),
            float(row["beta2"]),
            float(row["eps"]),
            float(row["warmup_fraction"]),
            float(row["mparams"]),
            float(row["total_samples_planned"]),
            float(row["training_progress"]),
            float(row["lr_ratio"]),
        ], dtype=torch.float32)

        vision_vec = convert_width_to_one_hot(int(row["vision_width"])).to(torch.float32)
        text_vec = convert_width_to_one_hot(int(row["text_width"])).to(torch.float32)

        x = torch.cat([numeric, vision_vec, text_vec])

        y = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        return x, y


class SurrogateDataset:
    """
    Loads train/test CSVs and provides deterministic config based subsets.
    Returns numpy arrays.
    """

    DEFAULT_TARGETS = ["val_loss"]

    DEFAULT_FEATURES = [
        "lr", "wd", "beta1", "beta2", "eps", "warmup_fraction",
        "vision_width", "text_width", "mparams",
        "total_samples_planned",
        "training_progress", "lr_ratio",
    ]

    def __init__(
        self,
        train_csv_path: str,
        test_csv_path: str,
        features: list[str] | None = None,
        targets: list[str] | None = None,
        seed: int = 42,
        config_id_col: str = "config_id",
    ):
        self.features = features if features is not None else self.DEFAULT_FEATURES
        self.targets = targets if targets is not None else self.DEFAULT_TARGETS
        self.seed = int(seed)
        self.config_id_col = config_id_col

        self.train_df = pd.read_csv(train_csv_path)
        self.test_df = pd.read_csv(test_csv_path)

        configs = np.array(sorted(self.train_df[self.config_id_col].unique()))
        rng = np.random.default_rng(self.seed)
        rng.shuffle(configs)
        self._configs = configs

    def get_test_data(self):
        X_test = self.test_df[self.features].to_numpy()
        y_test = self.test_df[self.targets].to_numpy()
        return X_test, y_test

    def get_train_data(self):
        X_train = self.train_df[self.features].to_numpy()
        y_train = self.train_df[self.targets].to_numpy()
        return X_train, y_train

    def get_train_subset_df(self, n_cfg: int) -> pd.DataFrame:
        selected = set(self._configs[: int(n_cfg)])
        return self.train_df[self.train_df[self.config_id_col].isin(selected)]

    def get_train_subset(self, n_cfg: int):
        subset = self.get_train_subset_df(n_cfg)
        X = subset[self.features].to_numpy()
        y = subset[self.targets].to_numpy()
        return X, y

    def get_default_sizes(self):
        total = len(self._configs)
        sizes = [10, 20, 50, 100, 200]
        sizes = [s for s in sizes if s <= total]
        if total not in sizes:
            sizes.append(total)
        return sizes