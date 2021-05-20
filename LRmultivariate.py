import matplotlib.pyplot as plt
import numpy as np

n_iters = 1000
alpha = 0.0000002


def predict(x1_, x2_, theta_zero, theta_one, theta_two):
    y_predicted = np.dot(x1_, theta_one) + np.dot(x2_, theta_two) + theta_zero
    return y_predicted


def fit(x0_, x1_, x2_, data_y, theta_zero, theta_one, theta_two):

    for i in range(n_iters):
        # y_predicted = np.dot(x1_.T, theta_one) + np.dot(x2_.T, theta_two) + theta_zero
        y_predicted = np.multiply(x1_, theta_one) + np.multiply(x2_, theta_two) + theta_zero
        print(f'x2 shape = {np.shape(x2_)}')
        print(f'ypredict shape = {np.shape(y_predicted)}')
        print(f'thetaone shape = {np.shape(theta_one)}')
        dtheta_zero = 1/np.size(x0_) * np.dot(x0_, (y_predicted - data_y))
        dtheta_one = 1/np.size(x1_) * np.dot(x1_, (y_predicted - data_y))
        dtheta_two = 1/np.size(x1_) * np.dot(x2_, (y_predicted - data_y))
        # dtheta_one = 1/np.size(data_x) * np.sum(y_predicted - data_y)

        theta_zero -= alpha * dtheta_zero
        theta_one -= alpha * dtheta_one
        theta_two -= alpha * dtheta_two

        # err_list.append(mse(data_y, predict(data_x, theta_zero, theta_one)))

    return theta_zero, theta_one, theta_two


def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)


def main():
    data_x = np.array([[1500, 28],
                       [1600, 30],
                       [1450, 24],
                       [1350, 20],
                       [1864, 38],
                       [2000, 46],
                       [1476, 25],
                       [1001, 16],
                       [961, 12],
                       [1894, 40]])

    data_y = np.array([[60000, 65000, 54000, 51000, 71000, 78000, 53420, 32000, 29540, 73520]])
    x0 = np.ones(10)
    x1 = np.array([data_x[:, 0]])
    x2 = np.array([data_x[:, 1]])
    print(f'x1 = {x1}')

    initial_theta_zero, initial_theta_one, initial_theta_two = 0, 0, 0
    Theta_zero, Theta_one, Theta_two = fit(x0, x1, x2, data_y, initial_theta_zero,
                                           initial_theta_one, initial_theta_two)

    y_predicted = predict(x1, x2, Theta_zero, Theta_one, Theta_two)

    print('Error = ' + str(mse(data_y, y_predicted)))
    print(f'theta zero = {Theta_zero}')
    print(f'theta one = {Theta_one}')
    print(f'theta two = {Theta_two}')

    plt.scatter(x1, data_y, color="red", marker="x")
    plt.scatter(x2, data_y, color="red", marker="x")
    # plt.scatter(Data_x, Data_y, color="blue", marker="x")
    plt.plot(x1, predict(x1, x2, Theta_zero, Theta_one, Theta_two))
    plt.xlabel('x')
    plt.ylabel('prediction')
    plt.show()


if __name__ == '__main__':
    main()