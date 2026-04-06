from typing import Sequence, Callable

import numpy as np

from surrogate.data import SurrogateDataset
from surrogate.metrics import compute_regression_metrics


class SurrogateModel:
    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_with_uncertainty(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {"mean": np.asarray(self.predict(X)).reshape(-1)}

    def validate(self, dataset: SurrogateDataset, sizes):
        X_test, y_test = dataset.get_test_data()
        results = {}

        for n_cfg in sizes:
            X_sub, y_sub = dataset.get_train_subset(n_cfg)
            self.fit(X_sub, y_sub)

            y_pred = self.predict(X_test)

            results[int(n_cfg)] = compute_regression_metrics(y_test, y_pred)

        return results


class MultiLabelSurrogateModel:
    def __init__(
            self,
            labels: Sequence[str],
            model_factory: Callable[[str], SurrogateModel],
    ):
        self.labels = list(labels)
        self.models = {
            label: model_factory(label) for label in self.labels
        }

    def validate(self, dataset: SurrogateDataset, sizes):
        X_test, y_test = dataset.get_test_data()
        results = {}

        for n_cfg in sizes:
            X_sub, y_sub = dataset.get_train_subset(n_cfg)

            per_label = {}
            for i, label in enumerate(self.labels):
                model = self.models[label]
                model.fit(X_sub, y_sub[:, i])
                y_pred = model.predict(X_test)
                per_label[label] = compute_regression_metrics(y_test[:, i], y_pred)

            results[int(n_cfg)] = per_label

        return results

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)

        if y.ndim != 2 or y.shape[1] != len(self.labels):
            raise ValueError(
                f"Expected y shape (N, {len(self.labels)}), got {y.shape}."
            )

        for i, label in enumerate(self.labels):
            self.models[label].fit(X, y[:, i])

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        preds = [np.asarray(self.models[label].predict(X)).reshape(-1) for label in self.labels]
        return np.stack(preds, axis=1)

    def predict_with_uncertainty(self, X: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        X = np.asarray(X)
        outputs: dict[str, dict[str, np.ndarray]] = {}

        for label in self.labels:
            model_outputs = self.models[label].predict_with_uncertainty(X)
            outputs[label] = {
                key: np.asarray(value)
                for key, value in model_outputs.items()
            }

        return outputs

    def get_predictor(self, label: str) -> SurrogateModel:
        return self.models[label]