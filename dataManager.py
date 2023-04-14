import math
import random
from logger import Logger
import numpy as np


class DataManager:
    def __init__(self, data_path: str = 'data.csv', targets_path: str = 'targets.csv', standardize: bool = True):
        self.x = np.genfromtxt(data_path, delimiter=',')
        self.y = np.genfromtxt(targets_path, delimiter=',', dtype=float)
        # standardize
        if standardize:
            for i in range(self.x.shape[1]):
                avg = sum(self.x[:, i]) / self.x.shape[0]
                var = 0
                for j in range(self.x.shape[0]):
                    var += (self.x[j][i] - avg) ** 2
                var /= self.x.shape[0]
                var = math.sqrt(var)
                self.x[:, i] = (self.x[:, i] - avg) / var

        # insert indices of data
        idx = np.array([i for i in range(self.x.shape[0])], dtype=int)
        self.x = np.insert(self.x, 0, idx, axis=1)

        # replace 0 with -1 in labels
        for i in range(self.y.shape[0]):
            if self.y[i] == 0:
                self.y[i] = -1

    def get_folded_data(self, fold_id: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, int):
        """
        load data according to 10-Fold Cross Validation Technique
        :param fold_id:
        :return: train_x, train_y, validation_x, validation_y
        """

        num_tuples_per_fold = int(self.x.shape[0] / 10)

        valid_x = self.x[(fold_id - 1) * num_tuples_per_fold: fold_id * num_tuples_per_fold, :]
        valid_y = self.y[(fold_id - 1) * num_tuples_per_fold: fold_id * num_tuples_per_fold]

        train_x = self.x[0: (fold_id - 1) * num_tuples_per_fold, :]
        train_y = self.y[0: (fold_id - 1) * num_tuples_per_fold]
        train_x = np.append(train_x, self.x[fold_id * num_tuples_per_fold: self.x.shape[0], :], axis=0)
        train_y = np.append(train_y, self.y[fold_id * num_tuples_per_fold: self.y.shape[0]], axis=0)
        mapped_index = []
        for i in range(0, (fold_id - 1) * num_tuples_per_fold):
            mapped_index.append(i)
        for i in range(fold_id * num_tuples_per_fold, self.x.shape[0]):
            mapped_index.append(i)
        return train_x, train_y, valid_x, valid_y, mapped_index, self.x.shape[0]

    @staticmethod
    def generate_distributed_data(x: np.ndarray, y: np.ndarray, distribution: list, multiple: float = 1) -> (
            np.ndarray, np.ndarray):
        """
        generate new dataset based on the provided weight
        :param x:
        :param y:
        :param distribution:
        :param multiple: the size of the generated dataset divided by the size of the original dataset
        :return:
        """
        assert x.shape[0] == y.shape[0]
        prefix_sum = distribution.copy()
        len_prefix_sum = len(prefix_sum)
        for i in range(1, len_prefix_sum):
            prefix_sum[i] += prefix_sum[i - 1]

        generated_x = []
        generated_y = []
        num_samples = int(multiple * x.shape[0])

        # generate new dataset using the weight as distribution
        for _ in range(num_samples):
            rand = random.random()
            selected = DataManager.binary_search(prefix_sum, rand)
            generated_x.append(x[selected].tolist())
            generated_y.append(y[selected])
        return np.array(generated_x), np.array(generated_y)

    @staticmethod
    def binary_search(lst: list, obj: float):
        logger = Logger('dataManager_binarySearch')
        l = 0
        r = len(lst) - 1
        while l < r:
            mid = int((l + r) / 2)
            if lst[mid] < obj:
                l = mid + 1
            else:
                r = mid
        logger.log(f'obj = %f, found = %f' % (obj, lst[l]))
        return l