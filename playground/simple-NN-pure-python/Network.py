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
    def calc_loss(y_predicted, y_true):
        """zwraca listę - każdy element to wynik funkcji loss dla poszczególnego output neurona"""

        return [1/2 * (y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_predicted)]

    def tanh_derivative(self, x):
        return 1 - tanh(x) ** 2

    def forward(self, x):
        """tu bedzie docstring metody forward"""

        activations = [x]
        zs = []

        for layer in range(1, self.num_layers):
            a_layer = []
            z_layer = []
            for neuron in range(self.l_sizes[layer]):
                weighted_sum = 0
                for i, weight in enumerate(self.weights[layer-1][neuron]):
                    weighted_sum += weight * activations[layer-1][i]
                z_layer.append(weighted_sum)
                a_layer.append(tanh(weighted_sum + self.biases[layer-1][neuron]))
            zs.append(z_layer)
            activations.append(a_layer)

        return zs, activations

    def backprop(self, zs, activations, y_true, activ_fn_derivative, l_rate):

        """TODO: backprop for any num of layers. now for 3 layers"""

        # output layer gradients
        # NOTE: this part works for any number of hidden layers because of 'minus indexing'
        # TODO: maybe we can do 1 loop for output and hidden layers? or maybe not
        ob_gradients = []
        ow_gradients = []
        # for every neuron in output layer
        for neuron in range(self.l_sizes[-1]):
            # calculate delta - equation in the blog post
            # first term is derivative of our cost funtion
            # TODO: use external function cost_derivative() or smth...
            # Note: zs[-1] == zs[1] for 3 layers, because {len(zs) = len(activations)-1}, because we don't have zs for input layer
            delta = (activations[-1][neuron] - y_true[neuron]) * activ_fn_derivative(zs[-1][neuron])
            # grad for output layer biases is just the delta
            ob_gradients.append(delta)

            # grad for a certain weight is the delta times activation that 'this weight weights'
            # Note: this fragment works for any num of hidden layers because of '-2' indexing
            neuron_w_gradients = []
            for before_node in range(self.l_sizes[-2]):
                neuron_w_gradients.append(delta * activations[-2][before_node])
            ow_gradients.append(neuron_w_gradients)

        # hidden layer
        # TODO: try to make a loop to make it work for any number of hidden layers

        hb_gradients = []
        hw_gradients = []

        # TODO: self.l_sizes[1] works only if we have 3 layers
        layer = 1
        for neuron_k in range(self.l_sizes[layer]):
            # calculate delta. It is more complicated this time (see blog post). its:
            # delta = tanh_derivative * sum[for each neuron in layer+1](delta_j * weight_jk)
            # where weight_jk is the one between current neuron k and neuron j from layer+1
            # delta_j is delta for the neuron j from layer+1
            sum_of_weighted_output_error = 0
            for neuron_j in range(self.l_sizes[layer+1]):
                # ob_gradients[0] is the same as delta_j so we use it instead of remembering deltas
                # NOTE: weights[layer] are weights of next layer, not current! coz we don't have weights for input layer
                sum_of_weighted_output_error += ob_gradients[0] * self.weights[layer][neuron_j][neuron_k]
            delta = sum_of_weighted_output_error * activ_fn_derivative(zs[0][neuron_k])
            hb_gradients.append(delta)

            # gradients of hidden weights - see blog post. We use the current delta and activations from last layer
            neuron_hw_gradients = []
            for before_node in range(self.l_sizes[layer-1]):
                neuron_hw_gradients.append(delta * activations[layer-1][before_node])
            hw_gradients.append(neuron_hw_gradients)

        # update parameters
        # TODO: make it work for any number of layers, and try to merge loops into one.
        # output layer
        layer = 2
        for neuron in range(self.l_sizes[layer]):
            self.biases[layer-1][neuron] -= l_rate * ob_gradients[neuron]
            for w in range(self.l_sizes[layer - 1]):
                self.weights[layer-1][neuron][w] -= l_rate * ow_gradients[neuron][w]
        # hidden layer
        layer = 1
        for neuron in range(self.l_sizes[layer]):
            self.biases[layer-1][neuron] -= l_rate * hb_gradients[neuron]
            for w in range(self.l_sizes[layer - 1]):
                self.weights[layer-1][neuron][w] -= l_rate * hw_gradients[neuron][w]
