from abc import abstractmethod

from sklearn.base import BaseEstimator


class Estimator(BaseEstimator):
    def fit(self, x, y):
        return self

    def predict(self, x, y):
        return x
