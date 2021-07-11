import numpy as np
# only forward propagation is performed.
"""
    w1, w2, w3: are the wieghts in each layer.
    x_in: is a vector representing single input.
    x_mat_in: is a matrix representing 7 different inputs.
    This will be a N.N. with 2 hidden layers of length 4, input layer is of dimensions 7 x 3,
    output layer is made up of 3 nodes.
"""
w1 = np.array([[2, -1, 1, 4], [-1, 2, -3, 1], [3, -2, -1, 5]])
w2 = np.array([[3, 1, -2, 1], [-2, 4, 1, -4], [-1, -3, 2, -5], [3, 1, 1, 1]])
w3 = np.array([[-1, 3, 2], [1, -1, 3], [3, -2, 2], [1, 2, 1]])
x_in = np.array([0.5, 0.8, 0.2])
x_mat_in = np.array([[.5, .8, .2], [.1, .9, .6], [.2, .2, .3], [.6, .1, .9],
                     [.5, .5, .4], [.9, .1, .9], [.1, .8, .7]])

def soft_max_vec(vec):
    return np.exp(vec)/(np.sum(np.exp(vec)))

def soft_max_mat(mat):
    return np.exp(mat)/(np.sum(np.exp(mat), axis=1).reshape(-1, 1))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

z_2 = np.dot(x_in, w1)
print(z_2)
# take sigmoid of z2 to produce a_2
a_2 = sigmoid(z_2)
# feed this activation value to the next layer i.e. layer 3
z_3 = np.dot(a_2, w2)

a_3 = sigmoid(z_3)

z_4 = np.dot(a_3, w3)
# since we are in the last layer, feed this z_4 to softmax function
y_out = soft_max_vec(z_4)
# y_out is the output for the first iteration
print(y_out)

# A one line function to do the entire N.N. computation
def nn_comp_vec(x):
    return soft_max_vec(sigmoid(sigmoid(np.dot(x, w1)).dot(w2)).dot(w3))

print()
print(nn_comp_vec(x_mat_in))