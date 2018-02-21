from random import randrange
from math import tanh
import random
random.seed(1)

class Network:
    """Prosta siec, czysty python"""

    def __init__(self, l_sizes=(2, 4, 2), num_layers=3):
        """tu bedzie docstring metody init
        args:
        """

        self.l_sizes = l_sizes
        self.num_layers = num_layers
        self.weights = None
        self.biases = None

        self.init_weights_biases()

    def init_weights_biases(self):
        self.weights = [[[random.random() - 0.5 for x in range(self.l_sizes[l-1])] for w in range(self.l_sizes[l])] for l in range(1, self.num_layers)]
        self.biases = [[random.random() - 0.5 for b in range(self.l_sizes[l])] for l in range(1, self.num_layers)]

    @staticmethod
    def calc_loss(self, y_predicted, y_true):
        """tu bedzie docstring metody calc_loss"""

        loss = []

        return (1/2) * (1 - y_predicted[y_true]) ** 2

    def forward(self, x):
        """tu bedzie docstring metody forward"""

        activations = [x]

        for layer in range(1, self.num_layers):
            a_layer = []
            for neuron in range(self.l_sizes[layer]):
                weighted_sum = 0
                for i, weight in enumerate(self.weights[layer-1][neuron]):
                    weighted_sum += weight * activations[layer-1][i]
                a_layer.append(tanh(weighted_sum + self.biases[layer-1][neuron]))
            activations.append(a_layer)

        return activations

    def backprop(self, loss, activations):

        """TODO: backprop for any num of layers. now for 3 layers"""



        pass
