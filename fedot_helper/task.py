from enum import Enum
from typing import Optional

from fedot_helper.data import Data
from fedot_helper.utils import get_seed

from sklearn.metrics import mean_absolute_percentage_error


class TaskTypesEnum(Enum):
    classification = 'classification'
    regression = 'regression'
    ts_forecasting = 'ts_forecasting'


class Task:
    """ Task is data with some information about what to do """

    def __init__(self,
                 type_: TaskTypesEnum,
                 data: Data,
                 cv_folds: int = 1,
                 seed: Optional[int] = None,
                 **params):
        # TODO make more explicit
        self.type = type_
        self._data = data
        self.seed = seed or get_seed()
        self.cv_folds = cv_folds
        self.split_ratio = 0.8
        self.stratify = False
        self.shuffle = True
        self.metric = mean_absolute_percentage_error

        for name, val in params.items():
            setattr(self, name, val)

    def __iter__(self):
        if self.type not in {TaskTypesEnum.regression, TaskTypesEnum.classification}:
            raise NotImplementedError()
        return self._data.cv_folds(cv_folds=self.cv_folds,
                                   split_ratio=self.split_ratio,
                                   shuffle=self.shuffle,
                                   stratify=self.stratify,
                                   seed=self.seed)
        return self

# TODO add tests for Task

