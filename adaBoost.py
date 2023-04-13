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
        self.data_manager = DataManager(feature_path, label_path, standardize=learner_config['standardize'])
        self.learner_type = learner_type
        self.learner_config = learner_config
        self.learner_sequence = []
        self.alpha: [float] = []
        self.distribution: [float] = []

    def train(self, fold_id: int, num_base: int):
        train_x, train_y, valid_x, _ = self.data_manager.get_folded_data(fold_id)
        self.learner_config['num_features'] = train_x.shape[1] - 1
        self.distribution = [0] * (train_x.shape[0] + valid_x.shape[0])
        is_in_train = [False] * (train_x.shape[0] + valid_x.shape[0])
        for row in train_x:
            is_in_train[int(row[0])] = True
        for i in range(train_x.shape[0] + valid_x.shape[0]):
            if is_in_train[i]:
                self.distribution[i] = 1 / train_x.shape[0]
        self.learner_sequence = []
        self.alpha = []
        for t in range(num_base):
            print(f'training the %d-th base learner of the %d-th fold...' % (t + 1, fold_id))
            assert sum(self.distribution) <= 1 + 1e-6
            if self.learner_config['use_distributed_dataset']:
                distributed_x, distributed_y = DataManager.generate_distributed_data(train_x, train_y,
                                                                                     self.distribution,
                                                                                     multiple=self.learner_config[
                                                                                         'sample_multiple'])
            else:
                distributed_x, distributed_y = train_x, train_y
            learner_t = self.learner_type(self.learner_config)
            learner_t.fit(distributed_x[:, 1:], distributed_y, self.distribution)
            pred = learner_t.predict(train_x[:, 1:])
            err = 0.0
            for i in range(train_x.shape[0]):
                if pred[i] != train_y[i]:
                    err += self.distribution[int(train_x[i][0])]
            self.alpha.append(math.log((1 - err) / err) / 2)
            self.learner_sequence.append(learner_t)

            for i in range(train_x.shape[0]):
                if pred[i] == train_y[i]:
                    self.distribution[i] *= math.exp(-self.alpha[t])
                else:
                    self.distribution[i] *= math.exp(self.alpha[t])
            tot_distribution = 0.0
            for d in self.distribution:
                tot_distribution += d
            for i in range(train_x.shape[0] + valid_x.shape[0]):
                self.distribution[i] /= tot_distribution

    def valid(self, fold_id: int, num_base: int):
        _, _, valid_x, valid_y = self.data_manager.get_folded_data(fold_id)
        pred = np.array([0] * valid_x.shape[0], dtype=float)
        for i in range(len(self.learner_sequence)):
            pred += self.learner_sequence[i].predict(valid_x[:, 1:]) * self.alpha[i]
        for i in range(len(pred)):
            if pred[i] > 0:
                pred[i] = 1
            else:
                pred[i] = -1
        acc = 0.0
        for i in range(len(pred)):
            if pred[i] == valid_y[i]:
                acc += 1
        acc /= valid_y.shape[0]
        print(f'fold = %d, acc rate = %f' % (fold_id, acc))
        if self.learner_config['write_to_file']:
            fileWriter.write(valid_x, pred, num_base, fold_id)
        return acc
