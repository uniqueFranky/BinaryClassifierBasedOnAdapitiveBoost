import dataManager

if __name__ == '__main__':
    dm = dataManager.DataManager()
    tx, ty, vx, vy = dm.get_folded_data(1)
    x, y = dataManager.DataManager.generate_weighted_data(tx, ty, [1 / tx.shape[0]] * tx.shape[0], multiple=2)
    print(x)