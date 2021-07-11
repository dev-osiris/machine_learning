import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def soft_max_mat(mat):
    return np.exp(mat)/(np.sum(np.exp(mat), axis=1).reshape(-1, 1))


def cross_entropy(prediction, target):
    N = prediction.shape[0]
    ce = -np.sum(target * np.log(prediction)) / N
    return ce

def derivative_cross_entropy(y, yhat):
    return np.where(y == 0, 1. / (1 - (yhat + 1e-10)), 1. / (yhat + 1e-10))

def derivative_tanh(x):
    return 1. - np.tanh(x) ** 2

def hard_max(x):
    return np.where(x > 0.5, 1, 0)

def relu(mat):
    return np.where(mat >= 0, mat, 0)

def relu_derivative(mat):
    return np.where(mat >= 0, 1, 0.1)

def relu_der_andrew(dA, z):
    dz = np.array(dA, copy=True)
    # computing relu derivative
    dz[z <= 0] = 0
    return dz

logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
cost = 1. / m * np.sum(logprobs)
loss = 1. / m * np.nansum(logprobs)
# for log loss first step in backprop is
dz3 = 1. / m * (a3 - Y)

# for finding accuracy: p is out output and y is the ground truth
print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
from sklearn.metrics import log_loss
# print(log_loss(targets, predictions))

# print(cross_entropy(predictions, targets))
# print(relu(predictions))
print(np.log(-5))