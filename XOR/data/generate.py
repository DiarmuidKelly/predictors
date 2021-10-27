import random
import numpy as np


def generate_dataset(num_samples=10000):
    dataset = []
    for i in range(num_samples):
        d = [np.array([random.choice([0, 1]), random.choice([0, 1])])]
        if d[0][0] == 1 and d[0][1] == 1:
            d.append(0)
        elif d[0][0] == 0 and d[0][1] == 0:
            d.append(0)
        elif d[0][0] == 1 or d[0][1] == 1:
            d.append(1)
        dataset.append(d)
    dataset = np.array(dataset, dtype=object)
    return dataset


if __name__ == "__main__":
    generate_dataset()