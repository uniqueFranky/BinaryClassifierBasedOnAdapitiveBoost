import math
import random
import numpy as np


class DataManager:
    def __init__(self, data_path: str = 'data.csv', targets_path: str = 'targets.csv', test_path: str = 'test.csv', standardize: bool = True, use_random: bool = False):
        self.x = np.genfromtxt(data_path, delimiter=',', dtype=float)
        self.y = np.genfromtxt(targets_path, delimiter=',', dtype=float)
        self.test = np.genfromtxt(test_path, delimiter=',', dtype=float)
        # standardize
        if standardize:
            mean = self.x.mean(axis=0)
            std = self.x.std(axis=0)
            self.x = (self.x - mean) / std

            mean = self.test.mean(axis=0)
            std = self.test.std(axis=0)
            self.test = (self.test - mean) / std

        # insert indices of data
        idx = np.array([i for i in range(self.x.shape[0])], dtype=int)
        self.x = np.insert(self.x, 0, idx, axis=1)

        idx = np.array([i for i in range(self.test.shape[0])], dtype=int)
        self.test = np.insert(self.test, 0, idx, axis=1)

        # replace 0 with -1 in labels
        for i in range(self.y.shape[0]):
            if self.y[i] == 0:
                self.y[i] = -1

        if use_random:
            random_idx = [i for i in range(self.x.shape[0])]
            random.Random(998244353).shuffle(random_idx)
            random_x = []
            random_y = []
            for i in range(self.x.shape[0]):
                random_x.append(self.x[random_idx[i]].tolist())
                random_y.append(self.y[random_idx[i]].tolist())
            random_x = np.array(random_x, dtype=int)
            random_y = np.array(random_y, dtype=int)
            assert random_x.shape == self.x.shape and random_y.shape == self.y.shape
            self.x = random_x
            self.y = random_y

    def get_folded_data(self, fold_id: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, int):
        """
        load data according to 10-Fold Cross Validation Technique
        :param fold_id:
        :return: train_x, train_y, validation_x, validation_y
        """
        if fold_id is None:
            return self.x, self.y, self.test, None, self.x[:, 0].astype(int), self.x.shape[0]
        num_tuples_per_fold = int(self.x.shape[0] / 10)

        valid_x = self.x[(fold_id - 1) * num_tuples_per_fold: fold_id * num_tuples_per_fold, :]
        valid_y = self.y[(fold_id - 1) * num_tuples_per_fold: fold_id * num_tuples_per_fold]

        train_x = self.x[0: (fold_id - 1) * num_tuples_per_fold, :]
        train_y = self.y[0: (fold_id - 1) * num_tuples_per_fold]
        train_x = np.append(train_x, self.x[fold_id * num_tuples_per_fold: self.x.shape[0], :], axis=0)
        train_y = np.append(train_y, self.y[fold_id * num_tuples_per_fold: self.y.shape[0]], axis=0)
        return train_x, train_y, valid_x, valid_y, train_x[:, 0].astype(int), self.x.shape[0]

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
        l = 0
        r = len(lst) - 1
        while l < r:
            mid = int((l + r) / 2)
            if lst[mid] < obj:
                l = mid + 1
            else:
                r = mid
        return l