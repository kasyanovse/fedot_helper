import catboost

from fedot_helper.estimator._estimator import Estimator
from fedot_helper.task import Task, TaskTypesEnum


class CatBoostRegressor(Estimator):
    def __init__(self, verbose: int = 0):
        super().__init__(subestimator=catboost.CatBoostRegressor)
        self.verbose = verbose


class CatBoostClassifier(Estimator):
    def __init__(self):
        super().__init__(estimator_class=catboost.CatBoostClassifier)


# TODO add tests
