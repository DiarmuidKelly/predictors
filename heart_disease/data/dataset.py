import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.labels = torch.Tensor(labels)
        self.labels = self.labels.type(torch.LongTensor)
        self.features = torch.Tensor(features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
