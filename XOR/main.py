from data.generate import generate_dataset
from model.model import Network
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset = generate_dataset(10000)
    test_dataset = generate_dataset(1000)
    net = Network()
    error = []
    for (x, y) in dataset:
        y_hat = net.forward(x)
        if y_hat > 1:
            y_hat = 0
        error.append((y - y_hat)**2)
        adj = error[-1] * y
        net.update_weights(adj)
    error = np.sum(error) / len(dataset)
    print("Train error: {}".format(error))
    error = []
    plots = []
    for (x, y) in test_dataset:
        y_hat = net.forward(x)
        if y_hat > 1:
            y_hat = 0
        plots.append([y - y_hat])
        error.append((y - y_hat)**2)
    error = np.sum(error) / len(dataset)
    plt.plot(plots, linewidth=0.2)
    plt.show()
    print("Test error: {}".format(error))



