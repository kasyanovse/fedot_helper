from typing import Optional

from statsmodels.tsa.arima.model import ARIMA

from fedot_helper.data import Data
from fedot_helper.estimator.estimator import Estimator


class ARIMAEstimator(Estimator):
    parameters_tuning_range = {'p': {'type': int, 'min': 2, 'max': 30, 'step': 1},
                               'd': {'type': int, 'min': 0, 'max': 2, 'step': 1},
                               'q': {'type': int, 'min': 0, 'max': 30, 'step': 1},
                               }

    def __init__(self, p: int = 2, d: int = 1, q: int = 2, forecast_length: int = 1):
        super().__init__()
        self.p = p
        self.d = d
        self.q = q
        self.forecast_length = forecast_length

    def fit(self, x: Data, y: Optional[Data] = None):
        self.estimator = (ARIMA(x.features, order=(self.p, self.d, self.q)).fit(), )
        return self

    def predict(self, x: Data):
        print(1)
        pass


class SARIMAEstimator(Estimator):
    parameters_tuning_range = {'p': {'type': int, 'min': 0, 'max': 20, 'step': 1},
                               'd': {'type': int, 'min': 0, 'max': 2, 'step': 1},
                               'q': {'type': int, 'min': 0, 'max': 20, 'step': 1},
                               'P': {'type': int, 'min': 0, 'max': 20, 'step': 1},
                               'D': {'type': int, 'min': 0, 'max': 2, 'step': 1},
                               'Q': {'type': int, 'min': 0, 'max': 20, 'step': 1},
                               's': {'type': int, 'min': 0, 'max': 200, 'step': 10},
                               'trend': {'type': 'category', 'values': (None, 'ct', 'c', 't')},
                               }

    def __init__(self, p: int = 2, d: int = 1, q: int = 2, P: int = 0,
                 D: int = 0, Q: int = 0, s: int = 0, trend: Optional[str] = None,
                 forecast_length: int = 1):
        super().__init__()
        self.p = p
        self.d = d
        self.q = q
        self.Q = Q
        self.D = D
        self.P = P
        self.s = s
        self.trend = trend
        self.forecast_length = forecast_length

    def fit(self, x: Data, y: Optional[Data] = None):
        raise NotImplementedError()
        params = {'order': (self.p, self.d, self.q),
                  'seasonal_order': (self.P, self.D, self.Q, self.s),
                  'trend': self.trend}

        estimator = ARIMA(x.features, **params).fit()
        self.estimator = (estimator, )

    def predict(self, x: Data):
        raise NotImplementedError()
        print(1)
        pass
