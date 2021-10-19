import csv
import random
from pathlib import Path

from data.model import Record
from data.dataset import Dataset
import configparser

from model.model import NeuralNetwork
import torch
from torch.utils.data import DataLoader

parent = Path(__file__).parent

config = configparser.ConfigParser()
config.read(str(parent) + '/hyperparameters.ino')
config = config["DEFAULT"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


# https://www.kaggle.com/fedesoriano/heart-failure-prediction
if __name__ == '__main__':
    dataset = []
    for key in config:
        print(key)
    t = config['TEST_PERCENTAGE_SPLIT']

    with open(str(parent) + '/data/heart.csv', 'r', newline='') as csvfile:
        linereader = csv.reader(csvfile, delimiter=' ')
        for row in linereader:
            row = row[0].split(',')
            dataset.append(row)
        processed = []
        for rec in dataset[1:]:
            r = Record()
            r.process_record(rec)
            processed.append(r)
        feature_vector = []
        targets = []
        for rec in processed:
            feature_vector.append(rec.get_feature_vector())
            targets.append(rec.get_target())

        random.shuffle(processed)
        test_end = round(len(processed) / (1 / float(config['TEST_PERCENTAGE_SPLIT'])))
        test_x = feature_vector[:test_end]
        test_y = targets[:test_end]
        train_x = feature_vector[test_end:]
        train_y = targets[test_end:]
        print("Data Split --- Test: {} \t Train: {}".format(len(test_x), len(train_x)))

        train_set = Dataset(train_x, train_y)
        test_set = Dataset(test_x, test_y)

        net = NeuralNetwork(len(test_x[0]), 1)
        train_loader = DataLoader(train_set, batch_size=int(config['BATCH_SIZE']))
        test_loader = DataLoader(test_set, batch_size=len(test_set))
        # train_features, train_labels = iter(train_loader)
        opt = torch.optim.Adam(net.parameters(), lr=float(config['LEARNING_RATE']))
        loss_fn = torch.nn.BCEWithLogitsLoss()
        accuracy = []
        net.train()
        for epoch in range(int(config['MAX_EPOCHS'])):
            train_acc = []
            test_acc = []
            for batch, (local_batch, local_labels) in enumerate(train_loader):
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                preds = net(local_batch)
                local_labels = local_labels.unsqueeze(1)
                loss = loss_fn(preds, local_labels.float())

                t_acc = binary_acc(preds, local_labels.float())

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_acc.append(t_acc)
                print("Epoch: {}, Batch: {}, Train accuracy: {}%".format(epoch, batch, train_acc[-1]))
                print("Epoch: {}, Batch: {}, Train loss: {}".format(epoch, batch, loss.item()))

            with torch.no_grad():
                test_batch = []
                test_labels = []
                for b, l in test_loader:
                    test_batch = b
                    test_labels = l
                test_batch, test_labels = test_batch.to(device), test_labels.to(device)
                test_labels = test_labels.unsqueeze(1)
                preds = net(test_batch)
                test_acc.append(binary_acc(preds, test_labels))
                print("Epoch: {}, Test accuracy: {}%".format(epoch, test_acc[-1]))
                test_loss = loss_fn(preds, test_labels.float())
                print("Epoch: {}, Test loss: {}".format(epoch, test_loss.item()))
            print("Epoch {}".format(epoch, batch))

    print("EXIT")
