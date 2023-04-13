import math
import numpy as np
from baseLearner import BaseLearner


class DecisionStump(BaseLearner):
    def __init__(self, config):
        super(DecisionStump, self).__init__(config)
        self.num_features = config['num_features']
        self.decision_index = -1
        self.decision_factor = 0.0
        self.decision_bias = 0.0
        self.best_entropy = 1e9

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
        self.best_entropy = 1e9
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
                negative_distribution_less = 0
                positive_distribution_less = 0
                negative_distribution_greater = 0
                positive_distribution_greater = 0
                for sample_id in range(x.shape[0]):
                    if x[sample_id][decision_index] <= decision_bias:
                        if y[sample_id] == 1:
                            positive_distribution_less += distribution[sample_id]
                        else:
                            negative_distribution_less += distribution[sample_id]
                    else:
                        if y[sample_id] == 1:
                            positive_distribution_greater += distribution[sample_id]
                        else:
                            negative_distribution_greater += distribution[sample_id]
                entropy_less = 0
                entropy_greater = 0
                if negative_distribution_less > 0:
                    entropy_less += - negative_distribution_less * math.log(negative_distribution_less)
                if positive_distribution_less > 0:
                    entropy_less += - positive_distribution_less * math.log(positive_distribution_less)
                if negative_distribution_greater > 0:
                    entropy_less += - negative_distribution_greater * math.log(negative_distribution_greater)
                if positive_distribution_greater > 0:
                    entropy_less += - positive_distribution_greater * math.log(positive_distribution_greater)
                entropy = (negative_distribution_less + positive_distribution_less) * entropy_less + (
                        negative_distribution_greater + positive_distribution_greater) * entropy_greater
                if entropy < self.best_entropy:
                    self.decision_index = decision_index
                    self.decision_bias = decision_bias
                    self.best_entropy = entropy
                    if negative_distribution_less > positive_distribution_less:
                        self.decision_factor = 1
                    else:
                        self.decision_factor = -1
