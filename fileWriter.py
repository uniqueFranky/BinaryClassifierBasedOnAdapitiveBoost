import numpy as np


def write(x: np.ndarray, y: np.ndarray, base, fold, path: str = 'experiments'):
    """
    write the predicted results to file
    :param x:
    :param y:
    :param base:
    :param fold:
    :param path:
    :return:
    """
    assert x.shape[0] == y.shape[0]
    if fold is None:
        with open(path, 'w') as f:
            for i in range(x.shape[0]):
                f.write(f'%d\n' % (1 if int(y[i]) == 1 else 0))
    else:
        with open(path + f'/base%d_fold%d.csv' % (base, fold), 'w') as f:
            for i in range(x.shape[0]):
                f.write(f'%d,%d\n' % (int(x[i][0]) + 1, 1 if int(y[i]) == 1 else 0))
