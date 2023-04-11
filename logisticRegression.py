import numpy as np
from baseLearner import BaseLearner


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


class LogisticRegressionClassifier(BaseLearner):
    def __init__(self, num_features, lr=0.001, max_iter=100):
        super().__init__()
        self.num_features = num_features
        self.w = np.zeros(num_features, dtype=float)
        self.lr = lr
        self.max_iter = max_iter

    def fit(self, x: np.ndarray, y: np.ndarray):
        for _ in range(self.max_iter):
            grad = np.zeros(self.num_features, dtype=float)
            for i, feature in enumerate(x):
                grad += (sigmoid(y[i] * np.dot(self.w, feature)) - 1) * y[i] * feature
            self.w -= self.lr * grad

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[1] == self.w.shape[0]
        results = []
        for feature in x:
            # dot(w, x) > 0 is equivalent to P(y = 1|w, x) = sigmoid(dot(w, x)) > 0.5
            if np.dot(self.w, feature) > 0:
                results.append(1)
            else:
                results.append(-1)
        return np.array(results, dtype=float)
