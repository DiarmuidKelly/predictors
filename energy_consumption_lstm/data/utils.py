import numpy as np


def bin_array(arr, bin_width=60*24):
    """

    :param arr:
    :param bin_width:
    :return:
    """
    arr = np.array(arr)
    return arr[:(arr.size // bin_width) * bin_width].reshape(-1, bin_width).mean(axis=1)