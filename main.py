import time

from logisticRegression import LogisticRegressionClassifier
from decisionStump import DecisionStumpClassifier
from adaBoost import AdaBooster
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

# python main.py 1    运行决策树桩
# python main.py 0    运行对数几率回归
if __name__ == '__main__':
    time_start = time.time()
    booster = None
    # 对数几率回归
    if len(sys.argv) == 2 and sys.argv[1] == '0':
        print('正在训练对数几率回归……')
        booster = AdaBooster(LogisticRegressionClassifier, logistic_regression_config, feature_path='data.csv',
                             label_path='targets.csv', test_path='test.csv')
    # 决策树桩
    else:
        print('正在训练决策树桩……')
        booster = AdaBooster(DecisionStumpClassifier, decision_stump_config, feature_path='data.csv',
                             label_path='targets.csv', test_path='test.csv')
    # 十折交叉验证
    for base in [1, 5, 10, 100]:
        accuracy = 0
        for fold in range(1, 11):
            booster.train(fold, base)
            accuracy += booster.valid(fold, base)
        print(f'base = %d, total accuracy = %f' % (base, accuracy / 10))
    time_end = time.time()
    print('十折交叉验证完成，用时', time_end - time_start, 's')
    print('十折交叉验证结果已写入experiments目录')

    # 测试
    print('\n开始测试……')
    time_start = time.time()
    booster.train(None, 1 if sys.argv[1] == '0' else 100)  # 决策树桩训练100个基分类器，对数几率回归训练1个基分类器
    booster.predict()  # 对test.csv进行预测并写入pred_y.csv
    time_end = time.time()
    print('测试完成，用时', time_end - time_start, 's')
    print('预测结果已写入pred_y.csv')
