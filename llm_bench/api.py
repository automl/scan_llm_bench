import math
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
import importlib.resources as pkg_resources
from llm_bench import surrogate_data
from llm_bench.data import SurrogateDataset
from llm_bench.models.pfn import TabPFNModel
from llm_bench.config import LLMConfig


class LLMBenchmark:
    def __init__(
        self,
        train_csv: str | None = None,
        test_csv: str | None = None,
        targets: list[str] | None = None,
        seed: int = 1,
        device: str = "cuda",
    ):
        targets = targets or ["valid_loss", "test_loss", "hellaswag_acc", "arc_challenge_acc", "arc_easy_acc",
                                  "copa_acc", "openbookqa_acc", "piqa_acc"]

        if train_csv is None:
            with pkg_resources.path(surrogate_data, "train.csv") as p:
                train_csv = str(p)

        if test_csv is None:
            with pkg_resources.path(surrogate_data, "test.csv") as p:
                test_csv = str(p)

        self.dataset = SurrogateDataset(
            train_csv_path=train_csv,
            test_csv_path=test_csv,
            targets=targets,
            seed=seed,
        )
        self.targets = self.dataset.targets
        self.features = self.dataset.features

        self.surrogate = TabPFNModel(seed=seed, device=device)

    def query(self, config: LLMConfig) -> dict:
        row = config.build_feature_row(self.features)
        model_stats = config.compute_model_stats()

        predictions = self._predict_performance(row)

        return {
            "predictions": predictions,
            "model_stats": model_stats,
        }

    def query_many(self, configs: list[LLMConfig]) -> list[dict]:
        rows = [config.build_feature_row(self.features) for config in configs]
        X_query = np.vstack(rows)
        stats_list = [config.compute_model_stats() for config in configs]

        predictions_list = self._predict_performance_many(X_query)

        results = []
        for i in range(len(configs)):
            results.append({
                "predictions": predictions_list[i],
                "model_stats": stats_list[i],
            })

        return results

    def _configs_to_surrogate_matrix(self, configs: list[LLMConfig]) -> np.ndarray:
        rows = [config.build_feature_row(self.features) for config in configs]
        return np.vstack(rows)

    def _predict_performance(self, row: np.ndarray) -> dict:
        X, y = self.dataset.get_all_data()

        predictions = {}

        for i, target in enumerate(tqdm(self.targets, desc="Evaluating targets")):
            y_target = y[:, i]

            self.surrogate.fit(X, y_target)
            pred = self.surrogate.predict_with_uncertainty(row)

            predictions[target] = {
                "mean": round(float(pred["mean"][0]), 4),
                "uncertainty_width": round(float(pred["uncertainty_width"][0]), 4)
            }

        return predictions

    def _predict_performance_many(self, X_query: np.ndarray) -> list[dict]:
        X_train, y_train = self.dataset.get_all_data()

        predictions_per_config = [{} for _ in range(X_query.shape[0])]

        for i, target in enumerate(tqdm(self.targets, desc="Evaluating targets")):
            y_target = y_train[:, i]

            self.surrogate.fit(X_train, y_target)
            pred = self.surrogate.predict_with_uncertainty(X_query)

            means = pred["mean"]
            widths = pred.get("uncertainty_width")

            for j in range(X_query.shape[0]):
                predictions_per_config[j][target] = {
                    "mean": round(float(means[j]), 4),
                    "uncertainty_width": round(float(widths[j]), 4) if widths is not None else None,
                }

        return predictions_per_config


if __name__ == "__main__":
    bench = LLMBenchmark(targets=["valid_loss", "test_loss", "hellaswag_acc", "arc_challenge_acc", "arc_easy_acc",
                                  "copa_acc", "openbookqa_acc", "piqa_acc"])

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
    results = bench.query_many([config1, config2, config3])
    print(results)
    # result1 = bench.query(config1)
    # result2 = bench.query(config2)
    # result3 = bench.query(config3)

    # print("Config 1 (512d, 12L, 1B tokens, 100% progress):")
    # print(f"  Predictions: {result1['predictions']}")
    # print(f"  Model Stats: {result1['model_stats']}")
    # print(f"  Total params: {result1['model_stats']['total_params']:,}")
    # print(f"  Total FLOPs:  {result1['model_stats']['total_training_flops_formatted']}")
    #
    # print()
    # print("Config 2 (768d, 16L, 5B tokens, 50% progress):")
    # print(f"  Predictions: {result2['predictions']}")
    # print(f"  Model Stats: {result2['model_stats']}")
    # print(f"  Total params: {result2['model_stats']['total_params']:,}")
    # print(f"  Total FLOPs:  {result2['model_stats']['total_training_flops_formatted']}")
    #
    # print()
    # print("Config 3 (768d, 16L, 5B tokens, 100% progress):")
    # print(f"  Predictions: {result3['predictions']}")
    # print(f"  Model Stats: {result3['model_stats']}")
    # print(f"  Total params: {result3['model_stats']['total_params']:,}")
    # print(f"  Total FLOPs:  {result3['model_stats']['total_training_flops_formatted']}")


