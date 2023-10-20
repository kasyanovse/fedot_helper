from sklearn.metrics import mean_absolute_percentage_error

from fedot_helper.estimator.catboost_estimator import CatBoostRegressor
from fedot_helper.task import Task, TaskTypesEnum
from fh_datasets.generated_data import linear_data


def test_catboost_regressor():
    estimator = CatBoostRegressor()
    task = Task(type_=TaskTypesEnum.regression, data=linear_data(shape=(1000, 20)))
    estimator.fit(task)
    predict = estimator.predict(task)
    assert mean_absolute_percentage_error(next(iter(task))[1].target, predict[0]) < 0.01
