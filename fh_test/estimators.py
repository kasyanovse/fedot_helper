from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression

from fedot_helper.data import Data
from fedot_helper.estimator.baseline_estimator import NaiveEstimator
from fedot_helper.estimator.estimator import Estimator


class EstimatorForTest(Estimator):
    def __init__(self, method: str = 'naive'):
        methods = {'linear': LinearRegression,
                   'naive': NaiveEstimator}
        self.model = methods.get(method)()
        if self.model is None:
            raise ValueError(f"Unknown method {method}. Allowed methods: {', '.join(methods)}")

    def fit(self, x: Data, y: Optional[Data] = None):
        self.model.fit(x.features.reshape((x.features.shape[0], -1)), x.target)
        return self

    def predict(self, x: Data):
        new = x.copy()
        new.predict = self.model.predict(x.features.reshape((x.features.shape[0], -1)))
        return new
