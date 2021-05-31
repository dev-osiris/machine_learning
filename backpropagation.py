import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import numpy as np

num_training_ex = 1
alpha = 0.0001

data = np.array([[0, 0, 0]])
output = []

for i in range(5000):
    mat = np.random.randint(low=-5, high=5, size=(1, 3))
    if mat[0][0] > 0 > mat[0][2] and mat[0][1] >= 0:
        output.append(1)
    else:
        output.append(0)
    data = np.append(data, mat, axis=0)
output = np.array(output)
data = data[1:, ]

x = data.T
y = output.T
y = y.reshape((1, 5000))
# x, y = load_breast_cancer(return_X_y=True)
# x = x.T
# x = x[:30, :]
# y = y.reshape((1, 569))
# x = np.array([[4, -1, -5], [4, 2, -3], [-2, -1, 5], [3, 2, -1], [1, 5, -5], [3, 3, -4], [1, -2, 1]]).T
# x = x.reshape((3, x.shape[1]))
# y = np.array([0, 1, 0, 1, 1, 1, 0])
# y = y.reshape((1, 7))

# weights and biases
# w1 = np.random.rand(10, x.shape[0])   # 4 nuerons in hidden layer and 30 features
w1 = np.random.rand(5, x.shape[0])
# b1 = np.random.rand(10, 1)
b1 = np.zeros((5, 1))

# w2 = np.random.rand(10, 10)   # 4 nuerons in 2 in hidden layers
# b2 = np.random.rand(10, 1)
w2 = np.random.rand(5, 5)
b2 = np.zeros((5, 1))

# w3 = np.random.rand(1, 10)   # 4 nuerons in hidden layer and one in output
# b3 = np.random.rand(1, 1)
w3 = np.random.rand(1, 5)
b3 = np.array([0]).reshape((1, 1))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivative_tanh(x):
    return 1. - np.tanh(x) ** 2

def hard_max(x):
    return np.where(x > 0.5, 1, 0)

cost_list = []
for i in range(2000):
    z1 = np.dot(w1, x) + b1
    assert(z1.shape == (5, x.shape[1]))

    A1 = np.tanh(z1)
    assert(A1.shape == (5, x.shape[1]))

    z2 = np.dot(w2, A1) + b2
    assert(z2.shape == (5, x.shape[1]))

    A2 = np.tanh(z2)
    assert(A2.shape == (5, x.shape[1]))

    z3 = np.dot(w3, A2) + b3
    assert(z3.shape == (1, x.shape[1]))

    A3 = z3
    # print(f"A3: {A3}")
    assert(A3.shape == (1, x.shape[1]))

    cost = np.sum((A3 - y) ** 2) / 2

    # backpropagation
    dz3 = A3 - y  # or dz3 = z3 - y
    assert(dz3.shape == (1, x.shape[1]))
    dw3 = (1 / num_training_ex) * np.dot(dz3, A2.T)
    assert(dw3.shape == (1, 5))
    db3 = (1 / num_training_ex) * np.sum(dz3, axis=1, keepdims=True)
    assert(db3.shape == (1, 1))
    dA2 = np.dot(w3.T, dz3)

    # dz2_ = np.dot(w3.T, dz3) * derivative_tanh(z2)
    dz2 = np.multiply(dA2, derivative_tanh(z2))
    assert(dz2.shape == (5, x.shape[1]))
    dw2 = (1 / num_training_ex) * np.dot(dz2, A1.T)
    assert(dw2.shape == (5, 5))
    db2 = (1 / num_training_ex) * np.sum(dz2, axis=1, keepdims=True)
    assert(db2.shape == (5, 1))
    dA1 = np.dot(w2.T, dz2)

    # dz1_ = np.dot(w2.T, dz2) * derivative_tanh(z1)
    dz1 = np.multiply(dA1, derivative_tanh(z1))
    assert(dz1.shape == (5, x.shape[1]))
    dw1 = (1 / num_training_ex) * np.dot(dz1, x.T)
    # dw1_ = (1 / num_training_ex) * np.dot(dz1, np.dot(w1.T, dz1).T)
    assert (dw1.shape == (5, x.shape[0]))

    db1 = (1 / num_training_ex) * np.sum(dz1, axis=1, keepdims=True)
    assert (db1.shape == (5, 1))

    # update parameters
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    w3 = w3 - alpha * dw3
    b3 = b3 - alpha * db3

    cost_list.append(cost)
    if i % 10 == 0:
        print(f"{i} | {cost}")

# x2, y2 = load_breast_cancer(return_X_y=True)
#
# test_mat = np.array(x2[10]).T
# test_mat = test_mat[:30]
# actual = y2[10]
# test_mat = test_mat.reshape((30, 1))
# result = np.tanh(np.dot(w3, np.tanh(np.dot(w2, np.tanh(np.dot(w1, test_mat) + b1)) + b2)) + b3)
test_mat = np.array([[4, -1, -5], [4, 2, -3], [-2, -1, 5], [3, 2, -1], [1, 5, -5], [3, 3, -4], [1, -2, 1]]).T
test_mat = test_mat.reshape((3, 7))

z1 = np.dot(w1, test_mat) + b1

A1 = np.tanh(z1)

z2 = np.dot(w2, A1) + b2

A2 = np.tanh(z2)

z3 = np.dot(w3, A2) + b3

A3 = z3


# print(f"result {result} \n")
print(f"result {hard_max(A3)}")
print(f"actual {[0, 1, 0, 1, 1, 1, 0]}")
# print(f"w1: {w1}\n")
# print(f"w2: {w2}\n")
# print(f"w3: {w3}")

plt.plot(cost_list)
plt.show()
