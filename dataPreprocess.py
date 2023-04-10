import math

import numpy as np


def read_from_csv(data_path: str, target_path: str) -> (np.ndarray, np.ndarray):
    """
    :param data_path: the file path of training dataset
    :param target_path: the file path of labels
    :return: a tuple of x and y
    """
    x = np.genfromtxt(data_path, delimiter=',')
    y = np.genfromtxt(target_path, delimiter=',', dtype=int)
    return x, y


def standardize_(x: np.ndarray):
    """
    :param x: the data to be standardized
    :return:
    """
    for i in range(x.shape[1]):
        avg = sum(x[:, i]) / x.shape[0]
        var = 0
        for j in range(x.shape[0]):
            var += (x[j][i] - avg) ** 2
        var /= x.shape[0]
        var = math.sqrt(var)
        x[:, i] = (x[:, i] - avg) / var


def replace_labels_(y: np.ndarray):
    """
    replace 0 in labels with -1
    :param y: labels to be replaced
    :return:
    """
    for i in range(y.shape[0]):
        if y[i] == 0:
            y[i] = -1


def get_data(data_path: str = 'data.csv', target_path: str = 'targets.csv') -> (np.ndarray, np.ndarray):
    """
    :param data_path: the file path of training dataset
    :param target_path: the file path of labels
    :return: x and y after preprocessing
    """
    x, y = read_from_csv(data_path, target_path)
    standardize_(x)
    replace_labels_(y)
    return x, y
