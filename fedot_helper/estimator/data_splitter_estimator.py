from typing import Optional, Tuple, List

from fedot_helper.estimator.estimator import Estimator
from fedot_helper.data import Data


class HorizontalDataSpitterEstimator(Estimator):
    """ Break features by axis=1, run inner estimators and concatenate results """
    def __init__(self, estimator: Optional[Tuple[Estimator]] = None):
        super().__init__()
        self.estimator = estimator

    def slicer(self, x):
        count = len(self.estimator)
        if x.feature_shape[0] % count != 0:
            raise ValueError(f"Feature size {x.feature_shape[0]} cannot be divided for {count} estimators")
        return [x[:, i::count] for i in range(count)]

    def fit(self, x: Data, y: Optional[Data] = None):
        for estimator, ix in zip(self.estimator, self.slicer(x)):
            estimator.fit(ix, y)
        return self.estimator

    def predict(self, x: Data):
        prediction = Data()
        for estimator, ix in zip(self.estimator, self.slicer(x)):
            prediction = prediction.hstack(estimator.predict(ix).predict_to_features())
        prediction.predict = prediction.features
        return prediction
