from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

ALPHA = 0.1
N_ITER = 500
TRAINING_SET_DIVISION_PERC = 0.33


boston = load_boston()
bost = pd.DataFrame(boston['data'])
bost.columns = boston['feature_names']
y = boston['target']
bost = bost.to_numpy()

x_train, x_val, y_train, y_val = \
    train_test_split(bost, y, test_size=TRAINING_SET_DIVISION_PERC, random_state=5)

m_train = x_val.shape[0]
part1 = np.linalg.inv(np.dot(x_train.T, x_train))
part2 = np.dot(x_train.T, y_train)

theta = np.dot(part1, part2)
theta = np.array([theta])
x_val = x_val.T
print(f'theta dim {np.ndim(theta)}, theta shape {np.shape(theta)}')
y_predict = np.dot(theta, x_val)

train_err = (1 / m_train) * np.sum(np.abs(x_val - y_predict))
print(train_err)
