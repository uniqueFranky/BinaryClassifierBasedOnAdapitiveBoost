import numpy as np


def write(x: np.ndarray, y: np.ndarray, base: int, fold: int, path: str = 'experiments'):
    """
    write the predicted
    :param x:
    :param y:
    :param base:
    :param fold:
    :param path:
    :return:
    """
    assert x.shape[0] == y.shape[0]
    with open(path + f'/base%d_fold%d.csv' % (base, fold), 'w') as f:
        for i in range(x.shape[0]):
            f.write(f'%d,%d\n' % (int(x[i][0] + 1), int(y[i])))
