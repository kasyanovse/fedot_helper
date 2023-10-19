from collections.abc import Collection
from typing import Optional, Any, Tuple, Union, List

import numpy as np


class Data:
    _type: Any = np.ndarray

    def __init__(self,
                 time: Optional[np.ndarray] = None,
                 features: Optional[np.ndarray] = None,
                 target: Optional[np.ndarray] = None,
                 predict: Optional[np.ndarray] = None,
                 ordered: bool = True):
        self.features = features
        self.target = target if target is not None else self.features
        self.time = time
        self.predict = predict
        self.ordered = ordered

        if self.time is None and self.target is not None:
            self.time = np.arange(len(self.target))

        if self.time is not None:
            if not isinstance(self.time, self._type):
                self.time = np.array(self.time)
            if self.time.ndim > 1:
                if set(self.time.shape[1:]) != {1}:
                    raise ValueError(f"Time should be 1-dimensional not {self.time.shape}")
                self.time = np.ravel(self.time)

        if self.features is not None:
            if not isinstance(self.features, self._type):
                self.features = np.array(self.features)
            if self.features.ndim == 1:
                self.features = np.reshape(self.features, (-1, 1))

        if self.target is not None:
            if not isinstance(self.target, self._type):
                self.target = np.array(self.target)
            if self.target.ndim > 1:
                if set(self.target.shape[1:]) != {1}:
                    raise ValueError(f"Target should be 1-dimensional not {self.target.shape}")
                self.target = np.ravel(self.target)

    def __repr__(self):
        return (f"time: {self.time.shape if self.time is not None else None} "
                f"features: {self.features.shape if self.features is not None else None} "
                f"target: {self.target.shape if self.target is not None else None} "
                f"predict: {self.predict.shape if self.predict is not None else None} "
                )

    # size
    def __len__(self):
        return None if self.time is None else len(self.time)

    @property
    def empty(self):
        return self.time is None and self.features is None and self.target is None

    @property
    def len(self):
        return None if self.time is None else len(self)

    @property
    def target_len(self):
        return self.target.shape[0]

    @property
    def feature_shape(self):
        return None if self.features is None else self.features.shape[1:]

    @property
    def shape(self):
        return None if self.features is None else self.features.shape

    # concat and slice
    def __getitem__(self, item: Union[slice, Tuple[slice], Collection]):
        data = self.copy()

        if isinstance(item, tuple):
            data.features = data.features[(slice(None), ) + item[1:]]
            item = item[0]

        if not isinstance(item, slice) or item != slice(None):
            data.time = data.time[item]
            data.features = data.features[item]
            data.target = data.target[item]
        return data

    def hstack(self, data: Union['Data', List['Data']]):
        if isinstance(data, Data):
            data = [data]

        if self is None or self.empty:
            if any(not np.array_equal(data[0].time, x.time) for x in data[1:]):
                raise ValueError(f"Cannot horizontal concatenate ``Data``'s with different ``time``")
            if any(data[0].feature_shape[1:] != x.feature_shape[1:] for x in data[1:]):
                raise ValueError(f"Cannot horizontal concatenate ``Data``'s with different ``feature`` shapes")
            return Data(time=data[0].time,
                        features=np.concatenate([x.features for x in data], axis=1),
                        target=data[0].target)
        else:
            return Data.hstack(None, [self] + data)

    def vstack(self, data: Union['Data', List['Data']]):
        if isinstance(data, Data):
            data = [data]

        if self is None or self.empty:
            if any(data[0].feature_shape[1:] != x.feature_shape[1:] for x in data[1:]):
                raise ValueError(f"Cannot vertical concatenate ``Data``'s with different ``features`` shapes")
            return Data(time=np.concatenate([x.time for x in data], axis=0),
                        features=np.concatenate([x.features for x in data], axis=0),
                        target=np.concatenate([x.target for x in data], axis=0))
        else:
            return Data.vstack(None, [self] + data)

    # predict
    def add_predict(self, predict: np.array):
        new = self.copy()
        new.predict = predict
        return new

    def predict_to_features(self):
        return Data(time=self.time,
                    features=self.predict,
                    target=self.target)

    # compare
    def __eq__(self, other):
        if self is other:
            return True
        if not self.shape == other.shape:
            return False
        if not np.array_equal(self.time, other.time):
            return False
        if not np.array_equal(self.features, other.features):
            return False
        if not np.array_equal(self.target, other.target):
            return False
        return True

    # copy
    def copy(self):
        return Data(time=self.time,
                    features=self.features,
                    target=self.target,
                    ordered=self.ordered)

    def deepcopy(self):
        return Data(time=self.time.copy(),
                    features=self.features.copy(),
                    target=self.target.copy(),
                    ordered=self.ordered)

    #checks
    def has_predict(self):
        return self.predict is not None

    def is_evenly_spaced(self):
        raise NotImplementedError()
