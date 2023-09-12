import pytest
import numpy as np

from fedot_helper.data import Data
from fh_datasets.generated_data import random_data


def get_data(shape=9):
    return Data(time=np.arange(shape) + 1,
                features=np.reshape((np.arange(shape * shape) + 1) * 10, (shape, shape)),
                target=(np.arange(shape) + 1) * 100)


@pytest.mark.parametrize(('time', 'features', 'target'),
                         [(None, [0, 1, 2, 3, 4, 5], None),
                          ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], None),
                          (None, [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]),
                          ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])])
def test_data_creation(time, features, target):
    data = Data(time=time, features=features, target=target)
    assert isinstance(data.time, data._type)
    assert isinstance(data.features, data._type)
    assert isinstance(data.target, data._type)
    assert np.array_equal(np.ravel(data.time), np.ravel(data.features))
    assert np.array_equal(np.ravel(data.target), np.ravel(data.features))
    assert len(data) == len(data.time)
    assert data.len == len(data)
    assert data.shape == data.features.shape
    assert data.feature_shape == data.features.shape[1:]


@pytest.mark.parametrize(('start', 'stop', 'step'),
                         [(None, 2, None),
                          (0, 3, None),
                          (2, 5, 1),
                          (2, -5, 1),
                          (2, 8, 2),
                          (-6, -2, 1),
                          (None, -2, 2),
                          ])
def test_data_slice(start, stop, step):
    data = get_data()
    sliced = data[start:stop:step]
    indexes = np.arange(len(data))[start:stop:step]

    assert np.array_equal(sliced.time, np.take(np.array(data.time), indexes, 0))
    assert np.array_equal(sliced.features, np.take(np.array(data.features), indexes, 0))
    assert np.array_equal(sliced.target, np.take(np.array(data.target), indexes, 0))


def test_data_copy():
    data = get_data()
    copied_data = data.copy()

    assert data is not copied_data
    assert copied_data.time is data.time
    assert copied_data.features is data.features
    assert copied_data.target is data.target


def test_data_deepcopy():
    data = get_data()
    copied_data = data.deepcopy()

    assert data is not copied_data
    assert copied_data.time is not data.time
    assert copied_data.features is not data.features
    assert copied_data.target is not data.target
    assert np.array_equal(copied_data.time, data.time)
    assert np.array_equal(copied_data.features, data.features)
    assert np.array_equal(copied_data.target, data.target)


def test_eq():
    datas = [get_data() for _ in range(2)]

    assert datas[0] == datas[1]
    assert datas[0] != datas[0][::2]

    data = datas[0].copy()
    data.time = data.time.copy()
    assert datas[0] == data

    data = datas[0].deepcopy()
    assert datas[0] == data

    data = datas[0].deepcopy()
    data.time[0] = data.time[0] + 1
    assert datas[0] != data

    data = datas[0].deepcopy()
    data.features[0] = data.features[0] + 1
    assert datas[0] != data

    data = datas[0].deepcopy()
    data.target[0] = data.target[0] + 1
    assert datas[0] != data


def test_hstack():
    def test(datas, data_stacked):
        n = len(datas)
        slice_shape = int(data_stacked.feature_shape[0] // n)
        data_sliced = [data_stacked[:, i * slice_shape:(i + 1) * slice_shape] for i in range(n)]
        assert all(d == ds for d, ds in zip(datas, data_sliced))

    target = random_data().target
    datas = [random_data(target=target) for _ in range(3)]

    # case 1
    data = datas[0]
    data_fs = data.feature_shape
    data_stacked = data.hstack(datas[1:])
    assert data.feature_shape == data_fs
    test(datas, data_stacked)

    # case 2
    test(datas, Data().hstack(datas))

    # case 3
    with pytest.raises(ValueError):
        new_datas = [data.deepcopy() for data in datas]
        new_datas[0].time = new_datas[0].time[1:]
        test(datas, Data().hstack(new_datas))

    # case 4
    with pytest.raises(ValueError):
        new_datas = [data.deepcopy() for data in datas]
        new_datas[0].features = new_datas[0].features[:, :, 1:]
        test(datas, Data().hstack(new_datas))



def test_vstack():
    def test(datas, data_stacked):
        n = len(datas)
        slice_shape = int(data_stacked.shape[0] // n)
        data_sliced = [data_stacked[i * slice_shape:(i + 1) * slice_shape] for i in range(n)]
        assert all(d == ds for d, ds in zip(datas, data_sliced))

    target = random_data().target
    datas = [random_data(target=target) for _ in range(3)]

    # case 1
    data = datas[0]
    data_fs = data.feature_shape
    data_stacked = data.vstack(datas[1:])
    assert data.feature_shape == data_fs
    test(datas, data_stacked)

    # case 2
    test(datas, Data().vstack(datas))

    # case 3
    with pytest.raises(ValueError):
        new_datas = [data.deepcopy() for data in datas]
        new_datas[0].features = new_datas[0].features[:, :, 1:]
        test(datas, Data().vstack(new_datas))

    # case 4
    with pytest.raises(ValueError):
        new_datas = [data.deepcopy() for data in datas]
        new_datas[0].features = new_datas[0].features[:, 1:]
        test(datas, Data().vstack(new_datas))
