import numpy as np


class SteelStockyard(object):
    def __init__(self):
        self.yard = np.full([4, 4], 0)


if __name__ == '__main__':
    s = SteelStockyard()
    print(s.yard)
