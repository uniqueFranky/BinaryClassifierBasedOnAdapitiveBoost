import numpy as np


class BaseLearner:
    def __init__(self):
        pass

    def fit(self, x: np.ndarray, y: np.ndarray):
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        pass
