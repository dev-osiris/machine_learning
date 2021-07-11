import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
import numpy as np

# Hyperparameters
activation_func_name = 'tanh'
epochs = 50_000
num_training_ex = 455
alpha = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

x, y = load_breast_cancer(return_X_y=True)
x = preprocessing.normalize(x, norm='l2')
x_test = x
y_test = y

# test, train = 80, 20 percent
# train = 455 examples
# test = 114 examples
x = x[0: 455].T
y = y[0: 455].reshape((1, 455))
assert(x.shape == (30, 455))
assert(y.shape == (1, 455))

# weights and biases
w1 = np.random.rand(10, x.shape[0])
Vdw1 = np.zeros(w1.shape)
Sdw1 = np.zeros(w1.shape)

b1 = np.zeros((10, 1))
Vdb1 = np.zeros(b1.shape)
Sdb1 = np.zeros(b1.shape)


w2 = np.random.rand(10, 10)
Vdw2 = np.zeros(w2.shape)
Sdw2 = np.zeros(w2.shape)

b2 = np.zeros((10, 1))
Vdb2 = np.zeros(b2.shape)
Sdb2 = np.zeros(b2.shape)


w3 = np.random.rand(1, 10)
Vdw3 = np.zeros(w3.shape)
Sdw3 = np.zeros(w3.shape)

b3 = np.zeros((1, 1))
Vdb3 = np.zeros(b3.shape)
Sdb3 = np.zeros(b3.shape)


def activation_func(mat):
    if activation_func_name == 'relu':
        return np.where(mat >= 0, mat, 0)
    elif activation_func_name == 'leaky_relu':
        return np.where(mat >= 0, mat, 0.1 * mat)
    elif activation_func_name == 'tanh':
        return np.tanh(mat)
    elif activation_func_name == 'sigmoid':
        return 1.0 / (1.0 + np.exp(-mat))


def delta_activation_func(mat):
    if activation_func_name == 'relu':
        return np.where(mat >= 0, 1, 0)
    elif activation_func_name == 'leaky_relu':
        return np.where(mat >= 0, 1, 0.1)
    elif activation_func_name == 'tanh':
        return 1.0 - np.tanh(mat) ** 2
    elif activation_func_name == 'sigmoid':
        sig = 1.0 / (1.0 + np.exp(-mat))
        return np.multiply(sig, (1 - sig))


def hard_max(mat):
    return np.where(mat > 0.5, 1, 0)


def cross_entropy_loss(A, Y):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(A)) + np.multiply(1 - Y, np.log(1 - A)))
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    return cost


cost_list = []
for epoch_num in range(1, epochs):
    z1 = np.dot(w1, x) + b1
    assert(z1.shape == (10, x.shape[1]))

    A1 = activation_func(z1)
    assert(A1.shape == (10, x.shape[1]))

    z2 = np.dot(w2, A1) + b2
    assert(z2.shape == (10, x.shape[1]))

    A2 = activation_func(z2)
    assert(A2.shape == (10, x.shape[1]))

    z3 = np.dot(w3, A2) + b3
    assert(z3.shape == (1, x.shape[1]))

    A3 = activation_func(z3)
    assert(A3.shape == (1, x.shape[1]))
    cost = np.sum((A3 - y) ** 2) / 2

    # cost = cross_entropy_loss(A3, y)
    # backpropagation

    dz3 = (A3 - y)   # or dz2 = A2 - y

    # dz3 = relu_der_andrew(dA3, z3)
    assert(dz3.shape == (1, x.shape[1]))
    dw3 = (1 / num_training_ex) * np.dot(dz3, A2.T)
    assert(dw3.shape == (1, 10))
    db3 = (1 / num_training_ex) * np.sum(dz3, axis=1, keepdims=True)
    assert(db3.shape == (1, 1))
    dA2 = np.dot(w3.T, dz3)

    dz2 = np.dot(w3.T, dz3) * delta_activation_func(z2)
    # dz2 = relu_der_andrew(dA2, z2)
    assert(dz2.shape == (10, x.shape[1]))
    dw2 = (1 / num_training_ex) * np.dot(dz2, A1.T)
    assert(dw2.shape == (10, 10))
    db2 = (1 / num_training_ex) * np.sum(dz2, axis=1, keepdims=True)
    assert(db2.shape == (10, 1))
    dA1 = np.dot(w2.T, dz2)

    dz1 = np.dot(w2.T, dz2) * delta_activation_func(z1)
    # dz1 = relu_der_andrew(dA1, z1)
    assert(dz1.shape == (10, x.shape[1]))
    dw1 = (1 / num_training_ex) * np.dot(dz1, x.T)
    assert (dw1.shape == (10, x.shape[0]))
    db1 = (1 / num_training_ex) * np.sum(dz1, axis=1, keepdims=True)
    assert (db1.shape == (10, 1))

    # ADAM OPTIMIZATION
    # 1 - momentum
    Vdw1 = beta1 * Vdw1 + (1 - beta1) * dw1
    Vdb1 = beta1 * Vdb1 + (1 - beta1) * db1

    Vdw2 = beta1 * Vdw2 + (1 - beta1) * dw2
    Vdb2 = beta1 * Vdb2 + (1 - beta1) * db2

    Vdw3 = beta1 * Vdw3 + (1 - beta1) * dw3
    Vdb3 = beta1 * Vdb3 + (1 - beta1) * db3

    # 2 - rmsprop
    Sdw1 = beta2 * Sdw1 + (1 - beta2) * np.square(dw1)
    Sdb1 = beta2 * Sdb1 + (1 - beta2) * np.square(db1)

    Sdw2 = beta2 * Sdw2 + (1 - beta2) * np.square(dw2)
    Sdb2 = beta2 * Sdb2 + (1 - beta2) * np.square(db2)

    Sdw3 = beta2 * Sdw3 + (1 - beta2) * np.square(dw3)
    Sdb3 = beta2 * Sdb3 + (1 - beta2) * np.square(db3)

    # bias correction for momentum
    denominator_beta1 = np.power(beta1, epoch_num)
    Vdw1_c = Vdw1 / (1 - denominator_beta1)
    Vdb1_c = Vdb1 / (1 - denominator_beta1)

    Vdw2_c = Vdw2 / (1 - denominator_beta1)
    Vdb2_c = Vdb2 / (1 - denominator_beta1)

    Vdw3_c = Vdw3 / (1 - denominator_beta1)
    Vdb3_c = Vdb3 / (1 - denominator_beta1)

    # bias correction for rmsprop
    denominator_beta2 = np.power(beta2, epoch_num)
    Sdw1_c = Sdw1 / (1 - denominator_beta2)
    Sdb1_c = Sdb1 / (1 - denominator_beta2)

    Sdw2_c = Sdw2 / (1 - denominator_beta2)
    Sdb2_c = Sdb2 / (1 - denominator_beta2)

    Sdw3_c = Sdw3 / (1 - denominator_beta2)
    Sdb3_c = Sdb3 / (1 - denominator_beta2)

    # update parameters
    w1 = w1 - (np.divide((alpha * Vdw1_c), np.sqrt(Sdw1_c + epsilon)))
    b1 = b1 - (np.divide((alpha * Vdb1_c), np.sqrt(Sdb1_c + epsilon)))

    w2 = w2 - (np.divide((alpha * Vdw2_c), np.sqrt(Sdw2_c + epsilon)))
    b2 = b2 - (np.divide((alpha * Vdb2_c), np.sqrt(Sdb2_c + epsilon)))

    w3 = w3 - (np.divide((alpha * Vdw3_c), np.sqrt(Sdw3_c + epsilon)))
    b3 = b3 - (np.divide((alpha * Vdb3_c), np.sqrt(Sdb3_c + epsilon)))

    # decreasing learning rate with epochs
    if epoch_num % 10000 == 0:
        alpha = 0.9 * alpha

    if epoch_num % 100 == 0:
        cost_list.append(cost)
        print(f"{epoch_num} | {cost} | {alpha}")


# calculate train error
train_mat = x
train_answer = y
# result = np.tanh(np.dot(w2, np.tanh(np.dot(w1, test_mat) + b1)) + b2)

z1 = np.dot(w1, train_mat) + b1
A1 = activation_func(z1)
z2 = np.dot(w2, A1) + b2
A2 = activation_func(z2)
z3 = np.dot(w3, A2) + b3
A3 = activation_func(z3)

absolute_train_err = (np.sum(np.abs(train_answer - hard_max(A3))))
print(f"train error: {(np.sum(train_answer - A3) ** 2) / train_answer.shape[1]}")
print(f"train error absolute: {absolute_train_err} wrong out of 455 training examples")
print(f"accuracy = {((455 - absolute_train_err) / 455) * 100}%")

# calculate test error
test_mat = x_test[455:].T
assert(test_mat.shape == (30, 114))
test_answer = y_test[455:]
test_answer = test_answer.reshape((1, 114))

z1 = np.dot(w1, test_mat) + b1
A1 = activation_func(z1)
z2 = np.dot(w2, A1) + b2
A2 = activation_func(z2)
z3 = np.dot(w3, A2) + b3
A3 = activation_func(z3)

absolute_test_err = (np.sum(np.abs(test_answer - hard_max(A3))))
print(f"\ntest error: {(np.sum(test_answer - A3) ** 2) / test_answer.shape[1]}")
print(f"test error absolute: {(np.sum(np.abs(test_answer - hard_max(A3))))} wrong out of 114 test examples")
print(f"accuracy = {((114 - absolute_test_err) / 114) * 100}%")

plt.plot(cost_list)
plt.show()
