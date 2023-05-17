import time

from logisticRegression import LogisticRegressionClassifier
from decisionStump import DecisionStumpClassifier
from adaBoost import AdaBooster
from logger import Logger
import sys

decision_stump_config = {
    'use_distributed_dataset': False,
    'sample_multiple': None,
    'write_to_file': True,
    'standardize': True,
    'use_random': True
}

logistic_regression_config = {
    'lr': 0.1,
    'max_iter': 5,
    'use_distributed_dataset': False,
    'sample_multiple': None,
    'write_to_file': True,
    'standardize': True,
    'use_random': False
}

def enumerate_super_parameters_logistic_regression():
    logistic_regression_config['write_to_file'] = False
    with open('logistic_super_parameters.csv', 'a') as f:
        logger = Logger('logisticSuperParameters')
        for lr in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]:
            for max_iter in [5, 10, 20, 50, 100]:
                logistic_regression_config['lr'] = lr
                logistic_regression_config['max_iter'] = max_iter
                booster = AdaBooster(LogisticRegressionClassifier, logistic_regression_config, feature_path='data.csv',
                                     label_path='targets.csv')

                for base in [1, 5, 10, 100]:
                    acc = 0
                    for fold in range(1, 11):
                        booster.train(fold, base)
                        acc += booster.valid(fold, base)
                    acc /= 10
                    logger.log('lr = %f, max_iter = %d, base = %d, accuracy rate = %f' % (lr, max_iter, base, acc))
                    f.write(f'%f, %d, %d, %f\n' % (lr, max_iter, base, acc))
                logger.log('\n')


if __name__ == '__main__':
    time_start = time.time()
    booster = None
    if len(sys.argv) == 2 and sys.argv[1] == '0':
        print('running Logistic Regression!!')
        booster = AdaBooster(LogisticRegressionClassifier, logistic_regression_config, feature_path='data.csv',
                             label_path='targets.csv')
    else:
        print('running Decision Stump!!')
        booster = AdaBooster(DecisionStumpClassifier, decision_stump_config, feature_path='data.csv',
                             label_path='targets.csv')

    for base in [1, 5, 10, 100]:
        accuracy = 0
        for fold in range(1, 11):
            booster.train(fold, base)
            accuracy += booster.valid(fold, base)
        print(f'base = %d, total accuracy = %f' % (base, accuracy / 10))
    time_end = time.time()
    print('用时', time_end - time_start, 's')

