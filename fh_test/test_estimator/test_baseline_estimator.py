import pytest
import numpy as np

from fh_datasets.generated_data import random_data
from fedot_helper.estimator.baseline_estimator import NaiveEstimator, AverageEstimator


def test_naive():
    data = random_data()
    predict = NaiveEstimator().fit_predict(data)
    assert np.array_equal(np.array(predict.predict), np.array(data.features))


@pytest.mark.parametrize(['window', ],
                         [(1, ),
                          (2, ),
                          (3, ),
                          (5, )])
def test_average_ordered(window):
    data = random_data()
    data.features = np.array([[0] * 3, [1, 1, 1], [2,2,2], [3] * 3, [4] * 3, [5] * 3, [6] * 3, [7] * 3, [8] * 3, [9] * 3])
    predict = AverageEstimator(window=window).fit_predict(data)

    ideal_predict = np.zeros((len(data.target), ) + data.feature_shape)
    ideal_predict[0] = np.array(data.features[0])
    for i in range(1, ideal_predict.shape[0]):
        if i - window + 1 < 0:
            ideal_predict[i] = np.mean(np.array(data.features[0:i + 1]), axis=0) * (i + 1) / window
        else:
            ideal_predict[i] = np.mean(np.array(data.features[i - window + 1:i + 1]), axis=0)
    assert np.allclose(np.array(predict.predict), np.array(ideal_predict))


def test_average_non_ordered():
    pass
