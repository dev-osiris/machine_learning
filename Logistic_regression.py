import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

N_ITERS = 1000
ALPHA = 0.00001


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fit(x, y):
    n_samples, n_features = x.shape
    weights = np.zeros(n_features)
    bias = 0
    cost_list = []

    # grad descent
    for _ in range(N_ITERS):
        linear_model = np.dot(x, weights) + bias
        y_predicted = sigmoid(linear_model)

        dw = (1 / n_samples) * np.dot(x.T, (y_predicted - y))
        db = (1 / n_samples) * np.sum(y_predicted - y)

        weights -= ALPHA * dw
        bias -= ALPHA * db

        cost_list.append(cost_func(n_samples, y, y_predicted))

    return weights, bias, cost_list


def cost_func(n_samples, y, y_predicted):
    return (1 / n_samples) * (-y.T).dot(np.log(y_predicted)) - (1 - y).T.dot(np.log(1 - y_predicted))


def predict(x, weights, bias):
    linear_model = np.dot(x, weights) + bias
    y_predicted = sigmoid(linear_model)
    y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
    return y_predicted_cls


def accuracy(y_true, y_pred):
    _accuracy = np.sum(y_true == y_pred) / len(y_true)
    return _accuracy * 100


def main():
    bc = datasets.load_breast_cancer()
    x, y = bc.data, bc.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1234)

    weights, bias, cost_list = fit(x_train, y_train)
    predictions = predict(x_test, weights, bias)

    print(f'accuracy: {accuracy(y_test, predictions): 0.3f} %')
    # cost_list = np.array(cost_list)

    plt.plot(cost_list)
    plt.xlabel('no of iterations')
    plt.ylabel('cost')
    plt.show()


if __name__ == '__main__':
    main()
