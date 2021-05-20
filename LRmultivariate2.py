import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

ALPHA = 0.1
N_ITER = 500
TRAINING_SET_DIVISION_PERC = 0.33


def initialize_parameters(lenw):
    w = np.zeros([1, lenw])  # w --> weight vector
    b = 0  # b --> bias
    return w, b


def forward_prop(x, w, b):  # w --> 1 x n ; x --> n x m | n: no of features| m: no of training ex
    z = np.dot(w, x) + b  # z --> 1 x m | b is scalar but converted to vector by numpy
    return z


def cost_func(z, y):
    m = y.shape[1]  # 1 is used for column
    j = (1/(2*m)) * np.sum(np.square(z - y))
    return j


def back_prop(x, y, z):
    m = y.shape[1]
    dz = (1/m) * (z - y)
    dw = np.dot(dz, x.T)  # dw --> 1 x n
    db = np.sum(dz)
    return dw, db


def grad_descent_update(w, b, dw, db, alpha):
    w = w - alpha * dw
    b = b - alpha * db
    return w, b


def linear_regression_model(x_train, y_train, x_val, y_val, alpha, n_iter):
    lenw = x_train.shape[0]
    w, b = initialize_parameters(lenw)
    costs_train = []
    m_train = y_train.shape[1]
    m_val = y_val.shape[1]

    for i in range(1, n_iter + 1):
        z_train = forward_prop(x_train, w, b)
        cost_train = cost_func(z_train, y_train)
        dw, db = back_prop(x_train, y_train, z_train)
        w, b = grad_descent_update(w, b, dw, db, alpha)

        # finding cost and error of training set
        # store training cost in a list for plotting
        if i % 10 == 0:
            costs_train.append(cost_train)

        train_err = (1/m_train) * np.sum(np.abs(z_train - y_train))

        # finding cost and error of validation set
        z_val = forward_prop(x_val, w, b)
        cost_val = cost_func(z_val, y_val)
        val_err = (1/m_val) * np.sum(np.abs(z_val - y_val))

        # print costs and errors
        print(f'iteration {i}/{str(n_iter)}')
        print(f'training cost {str(cost_train)} | validation cost {str(cost_val)}')
        print(f'training err {str(train_err)} | validation err {str(val_err)}')

    plt.plot(costs_train)
    plt.xlabel('iteration per tens')
    plt.ylabel('training cost')
    plt.title('learning rate')
    # x_val = np.array(x_val)
    # z_val = np.array(z_val)
    # Y_val = np.zeros((13, 167))
    # for i in range(0, 12):
    #     Y_val[i, :] = y_val[0, :]
    # print(f'xval {np.shape(x_val[:, 0])}, z_val {np.shape(z_val)}')
    #
    # plt.scatter(x_val, Y_val, color="red", marker="x")
    # plt.plot(x_val[0:167, 0], z_val)
    plt.show()


def main():
    boston = load_boston()
    bost = pd.DataFrame(boston['data'])
    bost.columns = boston['feature_names']

    # normalize the data so that it lies between -1, 1
    x = (bost - bost.mean()) / (bost.max() - bost.mean())
    y = boston['target']

    x_train, x_val, y_train, y_val = \
        train_test_split(x, y, test_size=TRAINING_SET_DIVISION_PERC, random_state=5)

    x_train = x_train.T
    y_train = np.array([y_train])
    x_val = x_val.T
    y_val = np.array([y_val])
    linear_regression_model(x_train, y_train, x_val, y_val, ALPHA, N_ITER)


if '__name__' == main():
    main()
