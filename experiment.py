from typing import Optional, Union, List

import numpy as np
from sklearn.model_selection import TimeSeriesSplit, KFold, train_test_split

from fedot_helper.data import Data
from fedot_helper.estimator.estimator import Estimator
from fedot_helper.metrics import make_scorer
from fedot_helper.utils import get_seed


class Experiment(Data):
    # TODO: tests
    def __init__(self,
                 data: Data,
                 forecast_length: int,
                 holdout_split_ratio: float,
                 cv_folds: int,
                 score_fun: Optional[Union[callable, List[callable]]] = None,
                 seed: Optional[int] = None):
        """ Wrap for data with some parameters
        :type seed: random seed
        :type score_fun: callable or list of callables for metric calculation
        :type cv_folds: cross validation folds for validation should be in range 1:inf
        :type holdout_split_ratio: split ratio for holdout sample used only for testing
        :type forecast_length: forecast length for time series prediction
        :type data: data for fit and predict
        """

        # TODO: speed up
        # TODO: add preprocessors
        # TODO: add way to change split parameters as stratification, shuffling and so on
        # TODO: add postprocessors
        # TODO: does experiment should be nested from Data?

        super().__init__(time=data.time, features=data.features, target=data.target, ordered=data.ordered)
        self.forecast_length = forecast_length
        self.holdout_split_ratio = holdout_split_ratio
        self.cv_folds = cv_folds
        self.scorer = make_scorer(score_fun)
        self.seed = seed or get_seed()

    def get_data(self, validation: bool = True):
        # prepare data for test
        holdout_splitter = DataSplitter(data=self.copy(), split_ratio=self.holdout_split_ratio, seed=self.seed,
                                        cv_folds=None, forecast_length=self.forecast_length)
        main_data, holdout = next(holdout_splitter.get())
        if not validation:
            return main_data, holdout

        # prepare data for validation
        data_splitter = DataSplitter(data=main_data, split_ratio=None, seed=self.seed,
                                     cv_folds=self.cv_folds, forecast_length=self.forecast_length)
        main_data, holdout = data_splitter.get()
        return data_splitter.get(), holdout

    def add_preprocess(self, transformer):
        # TODO: add preprocessing
        pass

    def _score(self, train, test, model):
        model = model.__sklearn_clone__()
        model.fit(train)
        return self.scorer(model, test)

    def score(self, model: Estimator, validation: bool = True):
        if validation:
            result = [self._score(*data, model) for data in self.get_data(validation)[0]]
            return np.mean(result)
        else:
            return self._score(*self.get_data(validation), model)


class DataSplitter:
    # TODO: tests
    def __init__(self, data: Data, split_ratio: Optional[float], seed: int,
                 cv_folds: Optional[int], forecast_length: int = 1):

        # TODO: it can be defined in Data or Experiment class

        self.data = data
        self.split_ratio = split_ratio
        self.seed = seed
        self.cv_folds = cv_folds
        self.forecast_length = forecast_length

        data_shape = self.data.target_len
        if self.cv_folds is not None:
            if self.forecast_length > data_shape / (self.cv_folds + 1):
                proposed_cv_folds_count = int((data_shape - self.forecast_length) // self.forecast_length)
                self.cv_folds = proposed_cv_folds_count if proposed_cv_folds_count >= 2 else None

        if self.cv_folds is None:
            test_shape = int(data_shape * (1 - self.split_ratio))
            if self.forecast_length > test_shape:
                self.split_ratio = 1 - self.forecast_length / data_shape
            test_share = 1 - self.split_ratio
        else:
            test_share = 1 / (self.cv_folds + 1)
        self.horizon = int(self.forecast_length * data_shape * test_share / self.forecast_length)

    def get(self):
        indexes = np.arange(self.data.shape[0])
        if self.cv_folds is not None:
            if self.data.ordered:
                kf = TimeSeriesSplit(n_splits=self.cv_folds, test_size=self.horizon).split(indexes, indexes)
            else:
                kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed).split(indexes, indexes)
        else:
            params = dict() if self.data.ordered else {'shuffle': True, 'random_state': self.seed}
            kf = [train_test_split(indexes, train_size=self.split_ratio, **params)]

        for train, test in kf:
            yield self.data[train], self.data[test]
