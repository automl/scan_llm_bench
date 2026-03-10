import numpy as np
from tabpfn import TabPFNRegressor

from surrogate.models.base import SurrogateModel


class TabPFNModel(SurrogateModel):
    def __init__(self, seed: int = 42, device: str = "cpu"):
        self.seed = seed
        self.device = device
        self.model = TabPFNRegressor(device=device, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)