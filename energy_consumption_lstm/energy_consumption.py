import csv
import random
from pathlib import Path
import numpy as np
import time as t
from data.model import Record, calculate_ranges
from visualise import graph_power

parent = Path(__file__).parent

dataset = []

if __name__ == "__main__":
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

    for d in dataset[1:100000]:
        rec = Record().process_entry(d)
        if rec is not False:
            processed.append(rec)
        else:
            process_error.append(d)
    print(t.time() - start)
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

    graph_power(dataset)
    print()




