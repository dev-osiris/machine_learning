# for dimension help refer to www.towardsdatascience.com/using-the-right-dimensions-for-your-neural-network-2d864824d0f
# only forward propagation is performed.
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# a neural network of 2 layers(1 hidden layer of 4 nodes)

# two training examples with 3 features each
input_mat = np.array([[20, 4, 7], [14, 3, 5]]).T
input_mat = input_mat.reshape((3, 2))
assert(input_mat.shape == (3, 2))

W1 = np.random.rand(4, 3)
bias1 = np.zeros((4, 1))

W2 = np.random.rand(1, 4)
bias2 = np.zeros((1, 1))
# forward propagation

# first layer
z1 = np.dot(W1, input_mat) + bias1
assert(z1.shape == (4, 2))

Activation1 = sigmoid(z1)
assert(Activation1.shape == (4, 2))

# second layer
z2 = np.dot(W2, Activation1) + bias2
assert(z2.shape == (1, 2))

Activation2 = sigmoid(z2)
assert(Activation2.shape == (1, 2))
print(Activation2)