import numpy as np

from fh_datasets.generated_data import random_data, linear_data
from fedot_helper.estimator.data_splitter_estimator import HorizontalDataSpitterEstimator
from fh_test.estimators import EstimatorForTest


def test():
    data = linear_data(shape=(10, 3))

    estimators = [EstimatorForTest('naive') for _ in range(3)]
    hds_estimator = HorizontalDataSpitterEstimator(estimators)
    hds_estimator.fit(data)
    prediction = hds_estimator.predict(data)
    assert prediction.feature_shape[0] == len(estimators)
    assert len(prediction) == len(data)
    assert all(np.array_equal(data.target, prediction.features[:, i]) for i in range(prediction.feature_shape[0]))

    estimators = [EstimatorForTest('linear') for _ in range(3)]
    hds_estimator = HorizontalDataSpitterEstimator(estimators)
    hds_estimator.fit(data)
    prediction = hds_estimator.predict(data)
    assert prediction.feature_shape[0] == len(estimators)
    assert len(prediction) == len(data)
    assert all(np.array_equal(data.target, prediction.features[:, i]) for i in range(prediction.feature_shape[0]))
