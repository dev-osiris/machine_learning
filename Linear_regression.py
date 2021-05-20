import numpy as np
import matplotlib.pyplot as plt
import random


alpha = 0.0000001
n_iters = 1000
err_list = []


def func(m, x, c):
    return m * x + c


def ret_rand():
    return random.randint(-20, 20)


def gen_random_initial_data():
    random_x = []
    random_y = []
    test_x = []
    test_y = []
    for i in range(100):
        random_x.append(random.randint(900, 1050))
        test_x.append(random.randint(900, 1050))

    for j in range(100):
        random_y.append(func(0.65, random_x[j] + ret_rand(), 25))
        test_y.append(func(0.65, test_x[j] + ret_rand(), 25))

    rand_data = np.column_stack((random_x, random_y))
    return random_x, random_y, test_y, test_x


def fit(data_x, data_y, theta_zero, theta_one):

    for i in range(n_iters):
        y_predicted = np.dot(data_x, theta_zero) + theta_one

        dtheta_zero = 1/np.size(data_x) * np.dot(data_x.T, (y_predicted - data_y))
        # print(f'dtheta0 = {dtheta_zero}')
        dtheta_one = 1/np.size(data_x) * np.sum(y_predicted - data_y)

        theta_zero -= alpha * dtheta_zero
        theta_one -= alpha * dtheta_one

        err_list.append(mse(data_y, predict(data_x, theta_zero, theta_one)))

    return theta_zero, theta_one


def predict(data_x, theta_zero, theta_one):
    y_predicted = np.dot(data_x, theta_zero) + theta_one
    return y_predicted


def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)


def main():
    Data_x, Data_y, Test_y, Test_x = gen_random_initial_data()
    Data_x = np.array(Data_x)
    Data_y = np.array(Data_y)

    Theta_zero, Theta_one = fit(Data_x, Data_y, 0, 0)

    y_predicted = predict(Test_x, Theta_zero, Theta_one)

    print('Error = ' + str(mse(Data_y, y_predicted)))
    print(f'theta one = {Theta_one}')
    print(f'theta zero = {Theta_zero}')

    plt.scatter(Test_x, Test_y, color="red", marker="x")
    # plt.scatter(Data_x, Data_y, color="blue", marker="x")
    plt.plot(Data_x, predict(Data_x, Theta_zero, Theta_one))
    plt.xlabel('x')
    plt.ylabel('prediction')
    plt.show()

    # iter_list = []
    # for ele in range(1000):
    #     iter_list.append(ele)
    #
    # plt.plot(iter_list, err_list)
    # plt.show()


if __name__ == '__main__':
    main()
