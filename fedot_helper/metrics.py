from typing import Optional, Union, List

from fedot_helper.data import Data


def make_scorer(score_fun: Union[callable, List[callable]]):
    if not isinstance(score_fun, list):
        score_fun = [score_fun]

    if len(score_fun) == 0:
        raise ValueError(f"There are no score functions")

    def scorer(model, x: Data, y: Optional[Data] = None, __score_fun=score_fun):
        if model is None:
            return int(2 ** 32 - 1)
        predict = model.predict(x)
        target =x.target if y is None else y.target
        scores = [score_fun(target, predict.predict) for score_fun in __score_fun]
        return sum(scores) / len(scores)

    return scorer