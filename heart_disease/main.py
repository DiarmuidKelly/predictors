import csv
import random
from pathlib import Path
from data.model import Record
import configparser

from model.model import NeuralNetwork

import torch

parent = Path(__file__).parent

config = configparser.ConfigParser()
config.read(str(parent) + '/hyperparameters.ino')
config = config["DEFAULT"]

# https://www.kaggle.com/fedesoriano/heart-failure-prediction
if __name__ == '__main__':
    dataset = []
    for key in config:
        print(key)
    t = config['TEST_PERCENTAGE_SPLIT']

    with open( str(parent) + '/data/heart.csv', 'r', newline='') as csvfile:
        linereader = csv.reader(csvfile, delimiter=' ')
        for row in linereader:
            row = row[0].split(',')
            dataset.append(row)
        processed = []
        for rec in dataset[1:]:
            r = Record()
            r.process_record(rec)
            processed.append(r)
        random.shuffle(processed)
        test_end = round(len(processed) / (1 / float(config['TEST_PERCENTAGE_SPLIT'])))
        test = processed[:test_end]
        train = processed[test_end:]
        print("Data Split --- Test: {} \t Train: {}".format(len(test), len(train)))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(device))
        net = NeuralNetwork(len(processed[0][0:-1]), len(processed[0][-1])) # TODO: fix input and output shapes
        net()




    print()