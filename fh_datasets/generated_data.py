import numpy as np

from fedot_helper.data import Data


def random_data(shape=(10, 3, 2, 2), target=None):
    return Data(index=np.arange(shape[0]),
                features=np.random.rand(*shape),
                target=np.random.rand(shape[0]) if target is None else target)


def linear_data(shape=(50, 3)):
    time = np.arange(shape[0] + 1)
    signal = time * np.random.rand(1) + np.random.rand(1)
    signal, target = signal[:-1], signal[1:]
    window_size = shape[1]
    _temp = [signal[i:-(window_size - i - 1)] for i in range(window_size - 1)] + [signal[window_size - 1:]]
    features = np.array(_temp).T
    return Data(index=time[:features.shape[0] - 1],
                features=features[:-1],
                target=target[window_size:])
