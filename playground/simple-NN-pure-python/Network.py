from random import randrange
from math import tanh


class Network:
    """Prosta siec, czysty python"""

    def __init__(self, l_sizes=(2, 4, 2), layers=3):
        """tu bedzie docstring metody init

        args:
        """

        self.weights = [[randrange(-5, 5) for w in range(l_sizes[l])] for l in range(1, layers)]
        self.biases = [[randrange(-5, 5) for b in range(l_sizes[l])] for l in range(1, layers)]

    @staticmethod
    def calc_loss(self, y_predicted, y_true):
        """tu bedzie docstring metody calc_loss"""

        return (1 - y_predicted[y_true]) ** 2

    def forward(self, x):
        """tu bedzie docstring metody forward"""
