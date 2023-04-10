import dataManager

if __name__ == '__main__':
    dm = dataManager.DataManager()
    tx, ty, vx, vy = dm.get_folded_data(2)
    print(tx.shape)
    print(vx.shape)