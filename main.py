import dataManager
from logisticRegression import LogisticRegressionClassifier
from decisionStump import DecisionStump
from adaBoost import AdaBooster

if __name__ == '__main__':
    booster = AdaBooster(DecisionStump, learner_config={
        'lr': 0.0005,
        'max_iter': 100,
        'sample_multiple': 1,
        'use_distributed_dataset': False,
        'write_to_file': False
    }, feature_path='data.csv', label_path='targets.csv')

    for base in [1, 5, 10, 100]:
        print(f'base = %d' % base)
        acc = 0
        for fold in range(1, 11):
            booster.train(fold, base)
            acc += booster.valid(fold, base)
        print(f'tot acc = %f' % (acc / 10))


