import numpy as np


class Network:
    def __init__(self):
        self.L1 = Layer(layer_width=2, input_width=1, bias=[0, -1])

    def forward(self, x):
        x = self.L1.forward(x)
        return np.sum(x)

    def update_weights(self, adj):
        self.L1.update_weights(adj)


class Layer:
    def __init__(self, layer_width=1, input_width=1, bias=None):
        if bias is None:
            bias = np.zeros(layer_width)
        else:
            bias = np.array(bias)
        if bias.shape[0] != layer_width:
            raise Exception("Bias must be same shape as layer")
        self.units = []
        for u in range(layer_width):
            self.units.append(Unit(input_width, bias[u]))

    def update_weights(self, adj):
        for u in self.units:
            u.weight = u.weight + adj

    def forward(self, x):
        ret = []
        for u in self.units:
            ret.append(u.activation(x))
        return ret


class Unit:
    def __init__(self, input_shape=2, bias=0):
        self.weight = np.random.uniform(0, 1, input_shape)
        self.bias = bias

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
