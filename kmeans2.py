import random
import numpy as np
import matplotlib.pyplot as plt

cost = 0


def initialize(num_of_clusters, _dataset):
    # NUM_OF_CLUSTERS points are picked randomly from dataset and labeled as centroids
    random_centroids = []
    for _ in range(num_of_clusters):
        random_centroids.append(random.choice(_dataset))

    return random_centroids


def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# centroids is a list of all random centroids, _example is one elment from dataset
def assign(_example, centroids):
    distances = []
    for i in centroids:
        # returns the index of the cluster from (1 - k) which is closest to the ith example
        distances.append(distance(_example, i))

    return np.argmin(distances)  # returning index from 1 to k


def fit(_dataset, _random_centroids):
    old_centroids = []
    for epoch in range(1, 50):
        closest_centroids = []
        # cluster assignment step
        for example in _dataset:
            # ith element(example) of closest_centroids is the corresponding
            # index of _random_centroids for which the distance is minimum.
            closest_centroids.append(assign(example, _random_centroids))

        # move centroid step
        _random_centroids = move_centroid(closest_centroids, _random_centroids, _dataset)
        # print(list(_random_centroids))
        if np.array_equal(old_centroids, _random_centroids):
            # previous centroids are same as new ones, means that optimum solution has been reached
            cost_function(_random_centroids, closest_centroids, _dataset)
            return _random_centroids
        old_centroids = _random_centroids

    print("No result found.")
    return _random_centroids


def move_centroid(_closest_centroids, _random_centroids, _dataset):
    average_list = []  # [[1, 2], [4, 5] ,......]
    updated_centroids = []
    for centroid_index in range(len(_random_centroids)):
        # find centroid_index in _closest_centroids
        for index, closest_centroid in enumerate(_closest_centroids):
            if centroid_index == closest_centroid:
                average_list.append(_dataset[index])
                # cost_function(_random_centroids[centroid_index], _dataset[index], _dataset)

        updated_centroids.append(find_average(average_list))
        average_list = []

    return updated_centroids


def find_average(_average_list):
    x_values = []
    y_values = []
    for point in _average_list:
        x_values.append(point[0])
        y_values.append(point[1])
    if len(x_values) == 0:
        x_values.append(0)
    if len(y_values) == 0:
        y_values.append(0)

    return [sum(x_values) / len(x_values), sum(y_values) / len(y_values)]


def cost_function(_random_centroids, _closest_centroids, _dataset):
    global cost
    for centroid_index in range(len(_random_centroids)):
        # find centroid_index in _closest_centroids
        for index, closest_centroid in enumerate(_closest_centroids):
            if centroid_index == closest_centroid:
                cost += distance(_random_centroids[centroid_index], _dataset[index]) / len(_dataset)


def plot(dataset, centroid_list, min_cost_index):
    x_values = []
    y_values = []
    for i in dataset:
        x_values.append(i[0])
        y_values.append(i[1])
    plt.scatter(x_values, y_values)
    x_values = []
    y_values = []
    for i in centroid_list[min_cost_index]:
        x_values.append(i[0])
        y_values.append(i[1])
    plt.scatter(x_values, y_values, c="green", s=80, marker='x')
    plt.show()


def main():
    from sklearn.datasets import make_blobs
    global cost
    # noinspection PyPep8Naming
    NUM_OF_CLUSTERS = 4
    cost_list = []
    centroid_list = []
    x, y = make_blobs(centers=4, n_samples=100, cluster_std=1.5, n_features=2, random_state=10)
    for i in range(30):
        random_centroids = initialize(NUM_OF_CLUSTERS, x)
        updated_centroids = fit(x, random_centroids)
        centroid_list.append(updated_centroids)
        cost_list.append(cost)
        cost = 0
    min_cost_index = np.argmin(cost_list)
    print(f"min cost = {cost_list[min_cost_index]}")
    print(f"final centroids: \n{centroid_list[min_cost_index]}")

    plot(x, centroid_list, min_cost_index)


if __name__ == '__main__':
    main()
