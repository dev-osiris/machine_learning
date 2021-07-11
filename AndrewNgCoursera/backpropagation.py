import matplotlib.pyplot as plt
import numpy as np

num_training_ex = 1
alpha = 0.0001


def create_data(count):
    data = np.array([[0, 0, 0]])
    output = []
    for i in range(count):
        mat = np.random.randint(low=-5, high=5, size=(1, 3))
        if mat[0][0] > 0 > mat[0][2] and mat[0][1] >= 0:
            output.append(1)
        else:
            output.append(0)

        data = np.append(data, mat, axis=0)
    output = np.array(output)
    return data[1:, ].T, output.T


x, y = create_data(5000)
y = y.reshape((1, -1))

# weights and biases
w1 = np.random.rand(5, x.shape[0])
b1 = np.zeros((5, 1))

w2 = np.random.rand(5, 5)
b2 = np.zeros((5, 1))

w3 = np.random.rand(1, 5)
b3 = np.array([0]).reshape((1, 1))


def cross_entropy(prediction, target):
    N = prediction.shape[1]
    ce = -np.sum(target * np.log(prediction + 1e-9)) / N
    return ce


def derivative_cross_entropy(y, yhat):
    return np.where(y == 0, 1. / (1 - (yhat + 1e-10)), 1. / (yhat + 1e-10))


def derivative_tanh(x):
    return 1. - np.tanh(x) ** 2


def hard_max(x):
    return np.where(x > 0.5, 1, 0)


cost_list = []
for i in range(1000):
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
    # cost2 = log_loss(y, A3)

    # backpropagation
    dz3 = A3 - y
    # dz3 = derivative_cross_entropy(y, A3)
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

# test_mat = np.array([[4, -1, -5], [4, 2, -3], [-2, -1, 5], [3, 2, -1], [1, 5, -5], [3, 3, -4],
#                      [1, -2, 1], ]).T

train_mat = x
train_answer = y

z1 = np.dot(w1, train_mat) + b1
A1 = np.tanh(z1)

z2 = np.dot(w2, A1) + b2
A2 = np.tanh(z2)

z3 = np.dot(w3, A2) + b3
A3 = z3

print(f"train error: {(np.sum(train_answer - hard_max(A3)) ** 2) / train_answer.shape[1]}")


test_mat, test_answer = create_data(1000)
test_answer = test_answer.reshape((1, -1))

z1 = np.dot(w1, test_mat) + b1
A1 = np.tanh(z1)

z2 = np.dot(w2, A1) + b2
A2 = np.tanh(z2)

z3 = np.dot(w3, A2) + b3
A3 = z3

print(f"test error: {(np.sum(test_answer - hard_max(A3)) ** 2) / test_answer.shape[1]}")

plt.plot(cost_list)
plt.show()
