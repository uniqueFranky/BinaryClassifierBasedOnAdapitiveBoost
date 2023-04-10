from dataPreprocess import read_from_csv
from dataPreprocess import standardize
if __name__ == '__main__':
    x, y = read_from_csv('data.csv', 'targets.csv')
    standardize(x)
    print(x)
