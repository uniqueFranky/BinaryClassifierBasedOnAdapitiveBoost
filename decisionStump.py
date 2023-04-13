import math
import numpy as np
from baseLearner import BaseLearner


class DecisionStumpClassifier(BaseLearner):
    def __init__(self, config):
        super(DecisionStumpClassifier, self).__init__(config)
        self.num_features = config['num_features']
        self.decision_index = -1
        self.decision_factor = 0.0
        self.decision_bias = 0.0
        self.best_accuracy = 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.decision_index >= 0
        results = []
        for feature in x:
            if (feature[self.decision_index] - self.decision_bias) * self.decision_factor >= 0:
                results.append(1)
            else:
                results.append(-1)
        return np.array(results)

    def fit(self, x: np.ndarray, y: np.ndarray, distribution: np.ndarray):
        assert x.shape[0] == y.shape[0] and x.shape[1] == self.num_features
        self.best_accuracy = 0
        for decision_index in range(self.num_features):  # enumerate every feature for the decision
            # choose the middle points of the intervals
            points = []
            for feature in x:
                points.append(feature[decision_index])
            points = sorted(set(points))
            middle_points = []
            for i in range(len(points) - 1):
                middle_points.append((points[i] + points[i + 1]) / 2)
            for decision_bias in middle_points:
                for decision_factor in [-1, 1]:
                    accuracy = 0
                    for sample_id in range(x.shape[0]):
                        if (x[sample_id][decision_index] - decision_bias) * decision_factor * y[sample_id] >= 0:
                            accuracy += distribution[sample_id]
                    if accuracy > self.best_accuracy:
                        self.best_accuracy = accuracy
                        self.decision_index = decision_index
                        self.decision_bias = decision_bias
                        self.decision_factor = decision_factor
        print(f'best accuracy = %f, decision_index = %d, decision_bias = %f, decision_factor = %f' % (self.best_accuracy, self.decision_index, self.decision_bias, self.decision_factor))