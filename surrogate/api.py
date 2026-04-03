import math
from dataclasses import dataclass

import numpy as np

from surrogate.data import SurrogateDataset
from surrogate.models.base import MultiLabelSurrogateModel
from surrogate.models.pfn import TabPFNModel
from surrogate.config import LLMConfig


class LLMBenchmark:
    """
    Surrogate benchmark for LLM scaling-law experiments.
    """

    def __init__(
        self,
        train_csv: str = "new_surrogate_data/train_last_rows.csv",
        test_csv: str = "new_surrogate_data/test_last_rows.csv",
        targets: list[str] | None = None,
        seed: int = 1,
        device: str = "cuda",
    ):
        targets = targets or ["valid_loss", "test_loss"]

        self.dataset = SurrogateDataset(
            train_csv_path=train_csv,
            test_csv_path=test_csv,
            targets=targets,
            seed=seed,
        )
        self.targets = self.dataset.targets
        self.features = self.dataset.features

        self.surrogate = MultiLabelSurrogateModel(
            labels=self.targets,
            model_factory=lambda label: TabPFNModel(seed=seed, device=device),
        )

        self.fit()

    def fit(self):
        X, y = self.dataset.get_train_data()
        self.surrogate.fit(X, y)

    def query(self, config: LLMConfig) -> dict:
        """
        Predict loss and return model cost statistics.
        """
        row = config.build_feature_row(self.features)
        model_stats = config.compute_model_stats()

        preds = self.surrogate.predict(row)[0]
        predictions = {
            label: round(float(pred), 4)
            for label, pred in zip(self.targets, preds)
        }

        return {
            "predictions": predictions,
            "model_stats": model_stats,
        }


if __name__ == "__main__":
    bench = LLMBenchmark(targets=["valid_loss", "test_loss"])

    config1 = LLMConfig(
        d_model=512,
        n_layers=12,
        n_heads=8,
        lr=3e-3,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        warmup_steps=0.05,
        cooldown_steps=0.2,
        n_tokens=1_000_000_000,
        training_progress=1.0,
    )

    config2 = LLMConfig(
        d_model=768,
        n_layers=16,
        n_heads=12,
        lr=1e-3,
        weight_decay=0.05,
        beta1=0.9,
        beta2=0.95,
        warmup_steps=0.05,
        cooldown_steps=0.2,
        n_tokens=5_000_000_000,
        training_progress=0.5,
    )

    config3 = LLMConfig(
        d_model=768,
        n_layers=16,
        n_heads=12,
        lr=1e-3,
        weight_decay=0.05,
        beta1=0.9,
        beta2=0.95,
        warmup_steps=0.05,
        cooldown_steps=0.2,
        n_tokens=5_000_000_000,
        training_progress=1,
    )

    result1 = bench.query(config1)
    result2 = bench.query(config2)
    result3 = bench.query(config3)

    print("Config 1 (512d, 12L, 1B tokens, 100% progress):")
    print(f"  Predictions: {result1['predictions']}")
    print(f"  Model Stats: {result3['model_stats']}")
    print(f"  Total params: {result1['model_stats']['total_params']:,}")
    print(f"  Total FLOPs:  {result1['model_stats']['total_training_flops_formatted']}")

    print()
    print("Config 2 (768d, 16L, 5B tokens, 50% progress):")
    print(f"  Predictions: {result2['predictions']}")
    print(f"  Model Stats: {result3['model_stats']}")
    print(f"  Total params: {result2['model_stats']['total_params']:,}")
    print(f"  Total FLOPs:  {result2['model_stats']['total_training_flops_formatted']}")

    print()
    print("Config 3 (768d, 16L, 5B tokens, 100% progress):")
    print(f"  Predictions: {result3['predictions']}")
    print(f"  Model Stats: {result3['model_stats']}")
    print(f"  Total params: {result3['model_stats']['total_params']:,}")
    print(f"  Total FLOPs:  {result3['model_stats']['total_training_flops_formatted']}")


