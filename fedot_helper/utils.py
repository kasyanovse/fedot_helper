import random


def get_seed():
    return random.randint(0, int(2**32 - 1))
