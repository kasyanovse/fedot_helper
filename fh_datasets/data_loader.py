import pickle
from datetime import datetime, timedelta
from pathlib import Path
from itertools import accumulate

from fedot_helper.data import Data

M4_NAME = Path(__file__).parent / 'data' / 'm4.pickle'


class Loader:
    def __init__(self):
        self.data = None
        self.info = None

    def __get_item__(self, item):
        if self.data is None:
            with open(self.path, 'rb') as f:
                self.data, self.info = pickle.load(f)

        if item not in self.data:
            raise KeyError(f"Unknown key {item}")

        return self.data[item], self.info[item]


class M4Loader(Loader):
    path = M4_NAME

    def __getitem__(self, item):
        data, info = super().__get_item__(item)

        time = [datetime.strptime(info['StartingDate'], '%d-%m-%y %H:%M')]
        if info['SP'] == 'Yearly':
            time = range(len(data))
        elif info['SP'] == 'Quarterly':
            time = range(len(data))
        elif info['SP'] == 'Monthly':
            time = range(len(data))
        else:
            mapper = {'Daily': 'days', 'Hourly': 'hours', 'Weekly': 'weeks'}
            time = time + [timedelta(**{mapper[info['SP']]: 1})] * (len(data) - 1)
            time = accumulate(time)

        return Data(time=time, features=data, target=data)
