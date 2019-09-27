import numpy as np


class Locating(object):
    def __init__(self, num_pile=4, max_stack=4, initial_state=None):
        self.action_space = num_pile
        self.max_stack = max_stack
        self.empty = 0
        self.yard = np.full([max_stack, num_pile], self.empty)

    def action(self, action):
        for i in range(self.max_stack):
            if self.yard[self.max_stack - i - 1, action] == self.empty:
                self.yard[self.max_stack - i - 1, action] = 1
                break

    def reset(self):
        return self.yard


if __name__ == '__main__':
    s = Locating()
    for j in range(2):
        for i in range(4):
            s.action(i)
    print(s.yard)
