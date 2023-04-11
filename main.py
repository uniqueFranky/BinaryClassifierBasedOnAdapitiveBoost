import dataManager
from logisticRegression import LogisticRegressionClassifier
if __name__ == '__main__':
    dm = dataManager.DataManager()
    for fold_id in range(1, 11):
        tx, ty, vx, vy = dm.get_folded_data(fold_id)
        clf = LogisticRegressionClassifier(tx.shape[1] - 1)
        clf.fit(tx[:, 1:], ty)
        pred = clf.predict(vx[:, 1:])
        err = 0
        for i in range(len(pred)):
            if pred[i] != vy[i]:
                err += 1
        print(f'fold %d' % fold_id)
        print(f'error rate =  %f' % (err / len(pred)))
