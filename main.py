import dataManager
from logisticRegression import LogisticRegressionClassifier
from adaBoost import AdaBooster

if __name__ == '__main__':
    booster = AdaBooster(LogisticRegressionClassifier, {
        'lr': 0.001,
        'max_iter': 100
    }, feature_path='data.csv', label_path='targets.csv')

    for base in [5, 10, 100]:
        print(f'base = %d' % base)
        err = 0
        for fold in range(1, 11):
            booster.train(fold, base)
            err += booster.valid(fold)
        print(f'tot err = %f' % (err / 10))


