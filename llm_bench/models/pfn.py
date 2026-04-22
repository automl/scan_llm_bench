import numpy as np
from tabpfn import TabPFNRegressor

from llm_bench.models.base import SurrogateModel


class TabPFNModel(SurrogateModel):
    def __init__(self, seed: int = 42, device: str = "cpu"):
        self.seed = seed
        self.device = device
        self.model = TabPFNRegressor(device=device, random_state=seed, ignore_pretraining_limits=True)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        outputs = self.model.predict(X, output_type="main")
        if isinstance(outputs, dict) and "mean" in outputs:
            return np.asarray(outputs["mean"]).reshape(-1)
        return np.asarray(outputs).reshape(-1)

    def _align_quantiles_by_sample(self, quantiles: np.ndarray, n_samples: int) -> np.ndarray:
        quantiles = np.asarray(quantiles)

        if quantiles.ndim == 1:
            if n_samples != 1:
                raise ValueError(
                    f"Expected quantiles for {n_samples} samples, got 1D shape {quantiles.shape}."
                )
            return quantiles.reshape(1, -1)

        if quantiles.ndim != 2:
            raise ValueError(f"Unexpected quantiles shape: {quantiles.shape}")

        if quantiles.shape[0] == n_samples:
            return quantiles

        if quantiles.shape[1] == n_samples:
            return quantiles.T

        raise ValueError(
            f"Could not align quantiles shape {quantiles.shape} with n_samples={n_samples}."
        )

    def predict_with_uncertainty(self, X: np.ndarray) -> dict[str, np.ndarray]:
        outputs = self.model.predict(X, output_type="main")

        if not isinstance(outputs, dict):
            mean = np.asarray(outputs).reshape(-1)
            return {"mean": mean}

        mean = np.asarray(outputs.get("mean")).reshape(-1)
        n_samples = mean.shape[0]

        result: dict[str, np.ndarray] = {
            "mean": mean,
        }

        if "median" in outputs:
            result["median"] = np.asarray(outputs["median"]).reshape(-1)

        if "mode" in outputs:
            result["mode"] = np.asarray(outputs["mode"]).reshape(-1)

        if "quantiles" in outputs:
            quantiles = self._align_quantiles_by_sample(outputs["quantiles"], n_samples)
            result["quantiles"] = quantiles
            result["uncertainty_width"] = quantiles[:, -1] - quantiles[:, 0]

        return result