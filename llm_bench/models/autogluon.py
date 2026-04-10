import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from llm_bench.models.base import SurrogateModel


class AutoGluonModel(SurrogateModel):

    def __init__(
            self,
            features: list[str],
            label: str = "clip_val_loss",
            time_limit: int = 30*60,
            presets: str = "best_quality",
            base_path: str = "AutogluonModels",
    ):
        self.features = features
        self.label = label
        self.time_limit = time_limit
        self.presets = presets
        self.base_path = base_path

        self.predictor = None
        self.run_id = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.run_id += 1

        X = np.asarray(X)
        y = np.asarray(y)

        train_df = pd.DataFrame(X, columns=self.features)
        train_df[self.label] = y

        path = f"{self.base_path}/run_{self.run_id}"

        self.predictor = TabularPredictor(
            label=self.label,
            problem_type="regression",
            eval_metric="rmse",
            path=path,
        )

        self.predictor.fit(
            train_data=train_df,
            time_limit=self.time_limit,
            presets=self.presets,
            dynamic_stacking=False,
            num_stack_levels=1,
            num_bag_folds=8,
            num_bag_sets=2,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        X_df = pd.DataFrame(X, columns=self.features)

        return self.predictor.predict(X_df).to_numpy()