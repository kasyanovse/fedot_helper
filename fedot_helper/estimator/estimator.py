from typing import Optional, Tuple

from sklearn.base import BaseEstimator

from fedot_helper.data import Data
from fedot_helper.utils import get_seed


class Estimator(BaseEstimator):
    parameters_tuning_range = dict()

    def __init__(self, estimator: Optional[Tuple[BaseEstimator]] = None,
                 n_jobs: int = -1,
                 seed: Optional[int] = None):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.seed = get_seed()

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"{', '.join(str(key) + ': ' + str(val) for key, val in self.get_params(False).items())})")

    def fit(self, x: Data, y: Optional[Data] = None):
        return self

    def predict(self, x: Data):
        raise NotImplementedError()

    def fit_predict(self, x: Data, y: Optional[Data] = None):
        self.fit(x, y)
        return self.predict(x)
