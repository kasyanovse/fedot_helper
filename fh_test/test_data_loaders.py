from fh_datasets.data_loader import M4Loader
from fedot_helper.data import Data

def test_m4_loader():
    loader = M4Loader()
    data = loader['Y1']
    assert isinstance(data, Data)
