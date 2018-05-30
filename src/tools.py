"""This module provides additional utilities, such as common functions

those utilities do not have any logical dependencies on other parts
of the project.

"""

import numpy as np


def truncate(arg, lower=0, upper=1):
    """This function returns value truncated within range <lower, upper>

    Args:
        arg (number) - value to be truncated
        lower (number) - lower truncating bound, default to 0
        upper (number) - upper truncating bound, default to 1

    Returns:
        arg (number) - truncated function argument
    """

    if arg > upper:
        return upper
    if arg < lower:
        return lower
    return arg


class SumTree:
    """Structure used in agent for storing priorities of the transitions.

    Used for PER (prioritized experience replay) implementation in Memory
    """
    pointer = 0

    def __init__(self, max_size):
        # number of leaves
        self.max_size = max_size
        # like a binary heap in an array
        # number-of-leaves + number-of-nodes = max_size + (max_size - 1)
        self.nodes = np.zeros(2 * max_size - 1)
        self.data = np.zeros(max_size, dtype=object)
        self.counter = 0

    def update(self, index, priority):
        change = priority - self.nodes[index]
        self.nodes[index] = priority

        while index != 0:
            index = (index - 1) // 2
            self.nodes[index] += change

    def add(self, priority, data):
        index = self.pointer + self.max_size - 1
        self.data[self.pointer] = data
        self.update(index, priority)

        if self.counter < self.max_size:
            self.counter += 1

    def get_leaves(self):
        return self.nodes[-self.max_size:]

    def get(self, value):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = 2 * parent_index + 2

            if left_child_index >= len(self.nodes):
                index = parent_index
                break
            else:
                if value <= self.nodes[left_child_index]:
                    parent_index = left_child_index
                else:
                    value -= self.nodes[left_child_index]
                    parent_index = right_child_index

        data_index = index - self.max_size + 1

        return (index, self.nodes[index], self.data[data_index])

    def get_priority_sum(self):
        return self.nodes[0]
