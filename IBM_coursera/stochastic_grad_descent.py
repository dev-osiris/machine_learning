import numpy as np

def stochastic_grad_descent(learning_rate, iterations, theta_initial, num_obs):
    #initialization
    theta = theta_initial
    theta_path = np.zeros(((iterations * num_obs) + 1, 3))
    theta_path[0, 1] = theta_initial
    loss_vector = np.zeros(iterations * num_obs)

    #main loop
    count = 0
    for i in range(iterations):
        for j in range(num_obs):
            count += 1
            y_pred = np.dot(theta.T, x_mat.T)