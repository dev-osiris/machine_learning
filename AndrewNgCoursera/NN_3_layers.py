# for dimension help refer to www.towardsdatascience.com/using-the-right-dimensions-for-your-neural-network-2d864824d0f
# only forward propagation is performed.
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def soft_max_mat(mat):
    return np.exp(mat)/(np.sum(np.exp(mat), axis=1).reshape(-1, 1))


# a neural network of 3 layers(2 hidden layer of 4 nodes) and one output layer of 3 nodes.

# seven training examples with 3 features each
input_mat = np.array([[.5, .8, .2],
                      [.1, .9, .6],
                      [.2, .2, .3],
                      [.6, .1, .9],
                      [.5, .5, .4],
                      [.9, .1, .9],
                      [.1, .8, .7]]).T

input_mat = input_mat.reshape((3, 7))
assert(input_mat.shape == (3, 7))

W1 = np.random.rand(4, 3)
print(W1)
print("\n\n\n")
bias1 = np.zeros((4, 1))

W2 = np.random.rand(4, 4)
bias2 = np.zeros((4, 1))

W3 = np.random.rand(3, 4)
bias3 = np.zeros((3, 1))
# forward propagation

# first layer
z1 = np.dot(W1, input_mat) + bias1
assert(z1.shape == (4, 7))

Activation1 = sigmoid(z1)
assert(Activation1.shape == (4, 7))

# second layer
z2 = np.dot(W2, Activation1) + bias2
assert(z2.shape == (4, 7))

Activation2 = sigmoid(z2)
assert(Activation2.shape == (4, 7))

# third layer
z3 = np.dot(W3, Activation2) + bias3
assert(z3.shape == (3, 7))

Activation3 = sigmoid(z3)
assert(Activation3.shape == (3, 7))

print(soft_max_mat(Activation3).T)
