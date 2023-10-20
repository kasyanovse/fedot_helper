from typing import Optional, List

from sklearn.base import BaseEstimator

from fedot_helper.task import Task
from fedot_helper.utils import get_seed


class Estimator(BaseEstimator):
    task_length = 1

    def __init__(self,
                 subestimator: Optional[List[BaseEstimator]] = None,
                 seed: Optional[int] = get_seed()):
        self.subestimator = subestimator
        self.seed = seed

        self.fitted = False
        self.estimator_parameters = dict()
        self.default_parameters = dict()
        self.parameters_tuning_range = dict()

    def fit(self, task: Task):
        self.estimators = [self.estimator_class(**self.estimator_parameters).fit(*train.extract())
                           for train, _ in task]
        self.fitted = True
        return self

    def predict(self, task: Task):
        if not self.fitted:
            raise ValueError(f"estimator is not fitted")
        return [estimator.predict(test.extract()[0]) for estimator, (_, test) in zip(self.estimators, task)]

    def fit_predict(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.predict(*args, **kwargs)
