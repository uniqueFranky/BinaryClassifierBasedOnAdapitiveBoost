import dataManager
from logisticRegression import LogisticRegressionClassifier
from decisionStump import DecisionStumpClassifier
from adaBoost import AdaBooster
from logger import Logger
decision_stump_config = {
    'use_distributed_dataset': False,
    'sample_multiple': 1,
    'write_to_file': True,
    'standardize': True
}

logistic_regression_config = {
    'lr': 0.00005,
    'max_iter': 25,
    'sample_multiple': 1,
    'use_distributed_dataset': True,
    'write_to_file': True,
    'standardize': True
}


def enumerate_super_parameters_logistic_regression():
    with open('logistic_super_parameters.csv', 'a') as f:
        logger = Logger('logisticSuperParameters')
        for lr in [0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]:
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
                logger.log('\n\n\n')
                f.write('\n\n\n')


if __name__ == '__main__':
    enumerate_super_parameters_logistic_regression()
    # booster = AdaBooster(DecisionStumpClassifier, decision_stump_config, feature_path='data.csv',
    #                      label_path='targets.csv')
    # for base in [1, 5, 10, 100]:
    #     print(f'base = %d' % base)
    #     acc = 0
    #     for fold in range(1, 11):
    #         booster.train(fold, base)
    #         acc += booster.valid(fold, base)
    #     print(f'tot acc = %f' % (acc / 10))


