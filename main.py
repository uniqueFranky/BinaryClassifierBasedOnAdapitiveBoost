from logisticRegression import LogisticRegressionClassifier
from decisionStump import DecisionStumpClassifier
from adaBoost import AdaBooster
from logger import Logger

decision_stump_config = {
    'use_distributed_dataset': False,
    'sample_multiple': 1.0,
    'write_to_file': True,
    'standardize': True
}

logistic_regression_config = {
    'lr': 0.00005,
    'max_iter': 25,
    'sample_multiple': 2.0,
    'use_distributed_dataset': True,
    'write_to_file': True,
    'standardize': True
}


def enumerate_super_parameters_logistic_regression():
    logistic_regression_config['write_to_file'] = False
    with open('logistic_super_parameters.csv', 'a') as f:
        logger = Logger('logisticSuperParameters')
        for lr in [0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]:
            for max_iter in [5, 10, 20, 50, 100]:
                for sample_multiple in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
                    logistic_regression_config['lr'] = lr
                    logistic_regression_config['max_iter'] = max_iter
                    logistic_regression_config['sample_multiple'] = sample_multiple
                    booster = AdaBooster(LogisticRegressionClassifier, logistic_regression_config, feature_path='data.csv',
                                         label_path='targets.csv')

                    for base in [1, 5, 10, 100]:
                        acc = 0
                        for fold in range(1, 11):
                            booster.train(fold, base)
                            acc += booster.valid(fold, base)
                        acc /= 10
                        logger.log('lr = %f, max_iter = %d, sample_multiple = %f, base = %d, accuracy rate = %f' % (lr, max_iter, sample_multiple, base, acc))
                        f.write(f'%f, %d, %f, %d, %f\n' % (lr, max_iter, sample_multiple, base, acc))
                    logger.log('\n\n\n')
                    f.write('\n\n\n')


if __name__ == '__main__':
    # enumerate_super_parameters_logistic_regression()
    booster = AdaBooster(DecisionStumpClassifier, decision_stump_config, feature_path='data.csv',
                         label_path='targets.csv')
    for base in [1, 5, 10, 100]:
        accuracy = 0
        for fold in range(1, 11):
            booster.train(fold, base)
            accuracy += booster.valid(fold, base)
        print(f'base = %d, total accuracy = %f' % (base, accuracy / 10))


