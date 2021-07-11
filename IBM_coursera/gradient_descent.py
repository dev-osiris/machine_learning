import numpy as np
import pandas as pd
import matplotlib as plt
np.random.seed(1234)

"""
    distribution:
    y = b + theta1 * x1 + theta2 * x2 + eps
    
    x1, x2: vectors of length 100 with values ranging from 0 to 100.
    'const' is a vector of ones of length 100 representing intercept term.
    'eps': is the errror term, it is a vector of length 100.
    we then generate a vector of y_values according to the model and put the predictors together in 
    a feature matrix 'x_mat'
"""
num_obs = 100
x1 = np.random.uniform(0, 10, num_obs)
x2 = np.random.uniform(0, 10, num_obs)
const = np.ones(num_obs)
eps = np.random.normal(0, 0.5, num_obs)

b = 1.5
theta_1 = 2
theta_2 = 5

y = b * const + theta_1 * x1 + theta_2 * x2 + eps
x_mat = np.array([const, x1, x2]).T

# using linear regression
# from sklearn.linear_model import LinearRegression
# lr_model = LinearRegression(fit_intercept=False)
# lr_model.fit(x_mat, y)
# print(lr_model.coef_)
#
# # using matrix manipulation
# theta = inverse(X_transpose * X) * X_transpose * y
# print(np.linalg.inv(np.dot(x_mat.T, x_mat)).dot(x_mat.T).dot(y))


def stochastic_grad_descent(learning_rate, iterations, theta_initial):
    # initialization
    theta = theta_initial
    theta_path = np.zeros(((iterations * num_obs) + 1, 3))
    theta_path[0, :] = theta_initial
    loss_vector = np.zeros(iterations * num_obs)

    # main loop
    count = 0
    for i in range(iterations):
        for j in range(num_obs):
            j = np.random.randint(0, num_obs)
            count += 1
            y_pred = np.dot(theta.T, x_mat.T)
            loss_vector[count - 1] = np.sum((y - y_pred) ** 2)
            grad_vec = (y[j] - y_pred[j] * (x_mat[j, :]))
            theta = theta + learning_rate * grad_vec
            theta_path[count, :] = theta
    return theta_path, loss_vector, theta

learning_rate = 1e-4
num_iter = 500
theta_initial = np.array([3, 3, 3])

theta_path, loss_vec, theta = stochastic_grad_descent(learning_rate, num_iter, theta_initial)
print(list(theta))