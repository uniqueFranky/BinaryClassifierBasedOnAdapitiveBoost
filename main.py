import dataManager
from logisticRegression import LogisticRegressionClassifier
from decisionStump import DecisionStumpClassifier
from adaBoost import AdaBooster

decisionStumpConfig = {
    'use_distributed_dataset': False,
    'write_to_file': True,
    'standardize': True
}

logisticRegressionConfig = {
    'lr': 0.0001,
    'max_iter': 100,
    'sample_multiple': 1,
    'use_distributed_dataset': True,
    'write_to_file': True,
    'standardize': True
}

if __name__ == '__main__':
    booster = AdaBooster(DecisionStumpClassifier, decisionStumpConfig, feature_path='data.csv', label_path='targets.csv')

    for base in [1, 5, 10, 100]:
        print(f'base = %d' % base)
        acc = 0
        for fold in range(1, 11):
            booster.train(fold, base)
            acc += booster.valid(fold, base)
        print(f'tot acc = %f' % (acc / 10))


