import math

import numpy as np
from numpy import exp


def sigmoid(x):
    return 1 / (1 + exp(-x))


def tanh(x):
    return np.tanh(x)