import csv
import random
from pathlib import Path
import numpy as np
import time as t
from data.model import Record, calculate_ranges
from data.utils import bin_array
from visualise import graph_power
from matplotlib import pyplot as plt
from model.model import LSTM, ClassicRNN
from tqdm import tqdm
parent = Path(__file__).parent

split = .7
dataset_size = 250000
batch_size = 10
# original data is sampled per minute; so 60 is an hour 60 * 24 a day... etc
bin_width = 60
look_back = 10


def normalise(arr):
    min = np.min(arr, axis=0)
    max = np.max(arr, axis=0)
    for idx, val in enumerate(arr):
        arr[idx] = (val - min) / (max - min)
    return arr


def create_dataset(power_vals, look_back=1):
    X, Y = [], []
    for i in range(len(power_vals) - look_back - 1):
        a = power_vals[i:(i + look_back)]
        X.append(a)
        Y.append(power_vals[i + look_back])
    X = np.array(X)
    X = np.expand_dims(X, axis=2)
    Y = np.array(Y)
    Y = np.expand_dims(Y, axis=1)
    return X, Y


if __name__ == "__main__":
    dataset = []
    if not Path.is_file(Path("./data/processed.npy")):
        with open(str(parent) + '/data/household_power_consumption.txt', 'r', newline='') as csvfile:
            linereader = csv.reader(csvfile, delimiter=';')
            for row in linereader:
                dataset.append(row)
            dataset = np.array(dataset)
            np.save("./data/processed", dataset)
    else:
        dataset = np.load("./data/processed.npy", allow_pickle=True)
    labels = dataset[0]
    labels = np.delete(labels, 1)
    labels[0] = "DateTime"
    labels = np.append(labels, "residual active energy")
    labels = np.append(labels, "error active vs amp *volt")
    labels = np.append(labels, "power")
    processed = []
    process_error = []
    start = t.time()
    s_val = random.randrange(1, len(dataset) - dataset_size, 1)
    print("Dataset range {}:{}".format(s_val, s_val + dataset_size))
    for d in dataset[s_val:s_val + dataset_size]:
        rec = Record().process_entry(d)
        if rec is not False:
            processed.append(rec)
        else:
            process_error.append(d)
    print("Dataset processing time : {}".format(t.time() - start))
    del d
    del start

    # TODO: break into own function and explain why it's here
    ranges = calculate_ranges(processed)
    print("{0:>25}: \t{1:>10}\t{2:>10}\t{3:>10}".format("label", "min", "mean", "max"))
    for idx, l in enumerate(labels):
        print("{0:>25}: \t{1:>10}\t{2:>10}\t{3:>10}".format(l, ranges[idx][0], ranges[idx][1], ranges[idx][2]))

    dataset = []
    for rec in processed:
        r = Record()
        r.process_record(rec, ranges)
        dataset.append(r)

    # graph_power(dataset)
    power_vals = []
    for rec in dataset:
        power_vals.append(rec.power)

    power_vals = bin_array(power_vals, bin_width=bin_width)

    train = power_vals[:int(len(power_vals) * split)]
    test = power_vals[int(len(power_vals) * split):]

    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)

    train_loss = 0.0
    running_loss = 0.0
    test_loss = 0.0
    preds = []
    LX = len(X_train)
    rem = LX % batch_size
    batches = (LX - rem)
    new_train_X = []
    new_train_Y = []
    for i in range(0, batches, batch_size):
        new_train_X.append(X_train[i:i+batch_size])
        new_train_Y.append(Y_train[i:i + batch_size])
    new_train_X.append(X_train[-rem:])
    new_train_Y.append(Y_train[-rem:])
    rnn = ClassicRNN(new_train_X[0].shape[1], 1)
    with tqdm(total=len(new_train_X)) as pbar:
        for idx, b in enumerate(new_train_X):
            for (sample, target) in zip(b, new_train_Y[idx]):
                running_loss += rnn.forward_backward(sample, target)
            train_loss += running_loss / float(b.shape[0])
            pbar.update(1)
    for (sample, target) in zip(X_test, Y_test):
        sample_loss, pred = rnn.forward(sample, target)
        test_loss += sample_loss
        preds.append(pred)
    train_loss = train_loss / float(len(new_train_X))
    test_loss = test_loss / float(Y_train.shape[0])
    print("Train set loss: {}".format(train_loss))
    print("Test set loss: {}".format(test_loss))
    preds = np.array(preds)
    preds = normalise(preds)
    vals = preds[:, 0, 0]
    axis = np.arange(0, len(X_train) + len(X_test), 1)
    mean = np.mean(X_train[:, -1])
    plt.hlines(y=mean, xmin=0, xmax=len(axis[:len(X_train)]))
    mean = np.mean(X_test[:, -1])
    plt.hlines(y=mean, xmin=0 + len(X_train), xmax=len(axis))
    plt.plot(axis[:len(X_train)], X_train[:, -1], 'b', label="Train Actual", alpha=0.5, linewidth=0.5)
    plt.plot(axis[len(X_train):], X_test[:, -1], 'r', label="Actual", alpha=0.5, linewidth=0.5)
    plt.plot(axis[len(X_train):], preds[:, 0, 0], 'g', label="Predicted", alpha=0.5, linewidth=0.5)
    plt.legend()
    plt.show()
    print()


