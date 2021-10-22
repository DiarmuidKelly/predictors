import numpy as np


def MSE_loss(X, Y, w):
    val = 0
    for x, y in zip(X, Y):
        val += y - np.dot(x, w)
    return val / 4


class Network:
    def __init__(self):
        self.L1 = Layer(2)
        self.L2 = Layer(2)

    def forward(self, x):
        x = self.L1.forward(x)
        out = self.L2.forward(x)
        return out


class Layer:
    def __init__(self, input_width=2):
        self.units = []
        for u in range(input_width):
            self.units.append(Unit())

    def forward(self, x):
        ret = []
        for u in self.units:
            ret.append(u.activation(x))
        return ret


class Unit:
    def __init__(self, input_shape=2):
        self.weight = np.ones(input_shape)
        self.bias = [0, -1]

    def activation(self, x):
        return self.ReLu((x * self.weight) + self.bias)

    def ReLu(self, x):
        ret = []
        if len(x.shape) > 1:
            for x_i in x:
                ret.append(self.ReLu(x_i))
        else:
            for x_i in x:
                ret.append(max(0, x_i))
        return ret
