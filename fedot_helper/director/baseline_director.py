from typing import Optional, List, Union

from sklearn.metrics import mean_absolute_percentage_error

from fedot_helper.data import Data
from fedot_helper.estimator.ar_estimator import ARIMAEstimator
from fedot_helper.estimator.baseline_estimator import NaiveEstimator, AverageEstimator
from fedot_helper.estimator.estimator import Estimator
from fedot_helper.metrics import make_scorer
from fedot_helper.tuner import Tuner


class BaselineTSDirector(Estimator):
    """ consecutive evaluation of baseline and choose the best """

    def __init__(self,
                 score_fun: Union[callable, List[callable]] = mean_absolute_percentage_error,
                 method: str = 'TPE',
                 tuner_timeout: int = 2,
                 ):
        super().__init__()
        self.score_fun = score_fun
        self.method = method
        self.tuner_timeout = tuner_timeout

    def fit(self, x: Data, y: Optional[Data] = None):
        tuner_params = {'scorer': make_scorer(self.score_fun),
                        'method': self.method}

        # basic
        basic = self.stage(x, y,
                           [NaiveEstimator, AverageEstimator],
                           tuner_params,
                           {'n_trials': 100, 'timeout': self.tuner_timeout})
        # ar
        # ar = self.stage(x, y,
        #                 [ARIMAEstimator, ],
        #                 tuner_params,
        #                 {'n_trials': 20, 'timeout': self.tuner_timeout})

    def predict(self, x: Data):
        pass

    def stage(self, x, y, models, tuner_params, tune_params):
        tuned_models = []
        for model in models:
            tuner = Tuner(model(), **tuner_params)
            tuned_models.append(tuner.tune(x, y, **tune_params))
        return min(tuned_models, key=lambda x: x[1])

