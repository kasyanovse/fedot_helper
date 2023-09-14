from typing import Optional

import numpy as np

from fedot_helper.data import Data
from fedot_helper.estimator.estimator import Estimator


class NaiveEstimator(Estimator):
    def predict(self, x: Data):
        return x.add_predict(x.features)


class AverageEstimator(Estimator):
    parameters_tuning_range = {'window': {'type': int, 'min': 1, 'max': 100, 'step': 1}}

    def __init__(self, window: int = 2):
        super().__init__()
        self.window = window

    def predict(self, x: Data):
        if x.ordered:
            f = x.features / self.window
            w = np.sum(f[-self.window:], axis=0)
            predict = np.zeros((len(x.target), ) + w.shape)
            predict[-1] = w
            for i in range(-1, -predict.shape[0] + self.window, -1):
                w = w - f[i] + f[i - self.window]
                predict[i - 1] = w
            for i in range(-predict.shape[0] + self.window, -predict.shape[0], -1):
                w = w - f[i]
                predict[i - 1] = w
        else:
            raise NotImplementedError()
        return x.add_predict(predict)
