import numpy as np


class BaseLearner:
    """
    an abstract class
    base learner for AdaBoost
    """
    def __init__(self, config: dict):
        self.config = config

    def fit(self, x: np.ndarray, y: np.ndarray, distribution: np.ndarray):
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        pass
