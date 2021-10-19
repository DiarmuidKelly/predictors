from matplotlib import pyplot as plt
import numpy as np
import datetime

def graph_power(dataset):
    x = []
    y = [[], [], [], []]
    labels = ["power", "sub_1", "sub_2", "sub_3"]
    error = []
    bin_width = 60 * 24
    for rec in dataset:
        x.append(rec.time_date)
        y[0].append(rec.power)
        y[1].append(rec.sub_meters[0])
        y[2].append(rec.sub_meters[1])
        y[3].append(rec.sub_meters[2])
        # y[1].append(rec.error_active)
        # y[2].append(rec.voltage)
        # y[3].append(rec.current)
        # y[4].append(rec.global_active_Ah_min)
    x = np.array(x)
    x = x[:(x.size // bin_width) * bin_width].reshape(-1, bin_width).mean(axis=1)
    processed = []
    for idx, val in enumerate(x):
        processed.append(datetime.datetime.fromtimestamp(round(val, 0)))
    del x
    x = processed
    del processed

    for idx, series in enumerate(y):
        series = np.array(series)
        result = series[:(series.size // bin_width) * bin_width].reshape(-1, bin_width).mean(axis=1)
        plt.plot(x, result, alpha=0.4, linewidth=0.5, label=labels[idx])
    focus = 0
    # plt.plot(x, y[focus], alpha=0.8, label=labels[focus])
    plt.legend()
    plt.show()
