import math
import numpy as np

import logger
from baseLearner import BaseLearner
from logger import Logger


class DecisionStumpClassifier(BaseLearner):
    def __init__(self, config):
        super(DecisionStumpClassifier, self).__init__(config)
        self.num_features = config['num_features']
        self.decision_index = -1
        self.decision_factor = 0.0
        self.decision_bias = 0.0
        self.best_accuracy = 0
        self.logger = Logger('decisionStump')

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
            decision_manager = DecisionManager(x, y, self.num_features, distribution, decision_index)
            candidate_points = decision_manager.get_candidate_points()
            for decision_bias in candidate_points:
                for decision_factor in [-1, 1]:
                    accuracy = 0
                    if decision_factor == -1:  # the samples with key < bias is predicted to be positive
                        accuracy += decision_manager.calculate_distribution_prefix_sum(positive=True,
                                                                                       bias=decision_bias)
                        accuracy += decision_manager.calculate_distribution_suffix_sum(positive=False,
                                                                                       bias=decision_bias)
                    else:  # the samples with key < bias is predicted to be negative
                        accuracy += decision_manager.calculate_distribution_prefix_sum(positive=False,
                                                                                       bias=decision_bias)
                        accuracy += decision_manager.calculate_distribution_suffix_sum(positive=True,
                                                                                       bias=decision_bias)
                    if accuracy > self.best_accuracy:
                        self.best_accuracy = accuracy
                        self.decision_index = decision_index
                        self.decision_bias = decision_bias
                        self.decision_factor = decision_factor
        self.logger.log(f'best accuracy = %f, decision_index = %d, decision_bias = %f, decision_factor = %f' % (
            self.best_accuracy, self.decision_index, self.decision_bias, self.decision_factor))


class DecisionPoint:
    def __init__(self, key: float, value: int):
        """
        :param key: the value of a specific feature
        :param value: the sample id of the corresponding key
        """
        self.key = key
        self.value = value


class DecisionManager:
    def __init__(self, x: np.ndarray, y: np.ndarray, num_features: int, distribution: np.ndarray, decision_index: int):
        self.x = x
        self.y = y
        self.num_features = num_features
        self.distribution = distribution
        self.decision_index = decision_index

        points = []
        negatives = []
        positives = []
        for (i, feature) in enumerate(x):
            points.append(feature[decision_index])
            if y[i] == 1:
                positives.append(DecisionPoint(key=feature[decision_index], value=i))
            else:
                negatives.append(DecisionPoint(key=feature[decision_index], value=i))

        points = sorted(set(points))
        self.positives = sorted(positives, key=lambda element: element.key)
        self.negatives = sorted(negatives, key=lambda element: element.key)
        len_positives = len(positives)
        len_negatives = len(negatives)

        self.positive_distribution_prefix_sum = [0] * len_positives
        self.negative_distribution_prefix_sum = [0] * len_negatives
        self.positive_distribution_prefix_sum[0] = distribution[positives[0].value]
        self.negative_distribution_prefix_sum[0] = distribution[negatives[0].value]
        for i in range(1, len_positives):
            self.positive_distribution_prefix_sum[i] = self.positive_distribution_prefix_sum[i - 1] + distribution[
                positives[i].value]
        for i in range(1, len_negatives):
            self.negative_distribution_prefix_sum[i] = self.negative_distribution_prefix_sum[i - 1] + distribution[
                negatives[i].value]
        self.candidate_points = []
        len_points = len(points)
        for i in range(len_points - 1):
            self.candidate_points.append((points[i] + points[i + 1]) / 2)

    def get_candidate_points(self) -> list:
        return self.candidate_points

    def calculate_distribution_prefix_sum(self, positive: bool = True, bias: float = 0) -> float:
        if positive:
            idx = DecisionManager.binary_search_less(self.positives, obj=bias)
            return self.positive_distribution_prefix_sum[idx]
        else:
            idx = DecisionManager.binary_search_less(self.negatives, obj=bias)
            return self.negative_distribution_prefix_sum[idx]

    @staticmethod
    def binary_search_less(lst: list, obj: float = 0) -> int:
        len_lst = len(lst)
        l = 0
        r = len_lst - 1
        while l < r:
            mid = int((l + r + 1) / 2)
            if lst[mid].key <= obj:
                l = mid
            else:
                r = mid - 1
        return l

    def calculate_distribution_suffix_sum(self, positive: bool = True, bias: float = 0) -> float:
        if positive:
            idx = DecisionManager.binary_search_greater(self.positives, obj=bias)
            return self.positive_distribution_prefix_sum[-1] - self.positive_distribution_prefix_sum[idx]
        else:
            idx = DecisionManager.binary_search_greater(self.negatives, obj=bias)
            return self.negative_distribution_prefix_sum[-1] - self.negative_distribution_prefix_sum[idx]

    @staticmethod
    def binary_search_greater(lst: list, obj: float = 0) -> int:
        len_lst = len(lst)
        l = 0
        r = len_lst - 1
        while l < r:
            mid = int((l + r + 1) / 2)
            if lst[mid].key < obj:
                l = mid
            else:
                r = mid - 1
        return l
