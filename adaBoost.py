import math

import numpy as np

import fileWriter
from baseLearner import BaseLearner
from dataManager import DataManager


class AdaBooster:
    def __init__(self, learner_type: type(BaseLearner), learner_config: dict, feature_path: str, label_path: str):
        """
        :param learner_type: the type of base learner boosted by the AdaBooster
        :param learner_config: the configuration to init the base learners
        :param feature_path: the file path to load features
        :param label_path: the file path to load labels
        """
        self.data_manager = DataManager(feature_path, label_path, standardize=learner_config['standardize'])  # 获取、管理数据
        self.learner_type = learner_type  # 基学习器的类型： LogisticRegressionClassifier 或 DecisionStumpClassifier
        self.learner_config = learner_config  # 用于初始化基学习器
        self.learner_sequence = []  # 用于线性叠加的基学习器们
        self.alpha: [float] = []  # 基学习器线性叠加时的权重
        self.distribution: [float] = []  # 数据中每个样本的权重，即讲义中的D_t (i)，由于其构成一个分布，故起名distribution

    def train(self, fold_id: int, num_base: int):
        """
        对第fold_id折的训练集，训练num_base层基分类器
        :param fold_id:
        :param num_base:
        :return:
        """
        # 从data_manager获取训练数据，验证集在训练时不能出现
        # 由于整体数据被分为10折，各个样本在csv文件中的原始序号与其在train_x和train_y数组中的下标不同，故需要mapped_index实现从数组下标到原始序号的映射
        train_x, train_y, _, _, mapped_index, num_samples = self.data_manager.get_folded_data(fold_id)
        self.learner_config['num_features'] = train_x.shape[1] - 1  # 数据的特征数

        # 初始化样本权重，对于出现在训练集中的样本，权重平均分配，未出现的样本权重为0（实际不会用到未出现的样本）
        self.distribution = [0] * num_samples
        for i in mapped_index:  # i为在训练集中出现的样本的原始序号
            self.distribution[i] = 1 / train_x.shape[0]
        self.learner_sequence = []
        self.alpha = []

        # 若使用样本权重来随机重构训练集，则在重构的训练集中，各个样本应为均匀分布，即multiplied_uniform
        multiplied_uniform = np.ones(train_x.shape[0])

        # 训练模型
        for t in range(num_base):
            print(f'training the %d-th base learner of the %d-th fold...' % (t + 1, fold_id))
            assert sum(self.distribution) <= 1 + 1e-6  # 考虑浮点误差，样本的权重之和应为1

            # 由于基分类器在训练时，只拿到了训练集的数据，而训练集的下标在distribution数组里的分布并不是连续的
            # 为了更为泛化地实现基分类器，基分类器内部不应该获取到样本的编号，因此inner_distribution为基训练器提供一个只包含训练集的样本权重
            inner_distribution = []
            for i in mapped_index:
                inner_distribution.append(self.distribution[i])

            if self.learner_config['use_distributed_dataset']:  # 是否使用样本权重来随机重构训练集
                distributed_x, distributed_y = DataManager.generate_distributed_data(train_x, train_y,
                                                                                     inner_distribution,
                                                                                     multiple=self.learner_config[
                                                                                         'sample_multiple'])
            else:
                distributed_x, distributed_y = train_x, train_y  # 不随机重构，则使用原始训练集

            # 以权重D(t)训练h_t，即以distribution为权重训练learner_t
            # 若已将训练集按照样本权重进行随机重构，则传入的权重应为均匀分布，否则传入distribution
            learner_t = self.learner_type(self.learner_config)
            learner_t.fit(distributed_x[:, 1:], distributed_y,
                          multiplied_uniform if self.learner_config['use_distributed_dataset'] else inner_distribution)
            pred = learner_t.predict(train_x[:, 1:])

            # 计算加权错误率
            err = 0.0
            for i in range(train_x.shape[0]):
                if pred[i] != train_y[i]:
                    err += self.distribution[int(train_x[i][0])]

            self.alpha.append(math.log((1 - err) / err) / 2)  # 根据加权错误率计算alpha
            self.learner_sequence.append(learner_t)  # 将新训练出的基分类器加入分类器序列

            # 更新样本权重
            for i in range(train_x.shape[0]):
                if pred[i] == train_y[i]:  # 预测正确，下一轮的权重变小
                    self.distribution[int(train_x[i][0])] *= math.exp(-self.alpha[t])
                else:  # 预测错误，下一轮的权重更大
                    self.distribution[int(train_x[i][0])] *= math.exp(self.alpha[t])

            # 对样本权重作归一化处理
            tot_distribution = 0.0
            for d in self.distribution:
                tot_distribution += d
            for i in range(num_samples):
                self.distribution[i] /= tot_distribution

    def valid(self, fold_id: int, num_base: int):
        """
        对第fold_id折的验证集进行预测
        :param fold_id:
        :param num_base:
        :return:
        """
        _, _, valid_x, valid_y, _, _ = self.data_manager.get_folded_data(fold_id)  # 仅获取验证集数据

        # 对验证集样本进行预测
        pred = np.array([0] * valid_x.shape[0], dtype=float)
        for i in range(len(self.learner_sequence)):  # 各个基分类器以alpha[i]为权重线性叠加
            pred += self.learner_sequence[i].predict(valid_x[:, 1:]) * self.alpha[i]
        for i in range(len(pred)):  # 计算sign(h_t(x))
            if pred[i] > 0:
                pred[i] = 1
            else:
                pred[i] = -1

        # 计算正确率
        acc = 0.0
        for i in range(len(pred)):
            if pred[i] == valid_y[i]:
                acc += 1
        acc /= valid_y.shape[0]
        print(f'fold = %d, acc rate = %f' % (fold_id, acc))

        if self.learner_config['write_to_file']:  # 将预测结果写入experiments/
            fileWriter.write(valid_x, pred, num_base, fold_id)
        return acc
