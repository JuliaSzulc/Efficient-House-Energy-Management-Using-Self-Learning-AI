from math import tanh
from collections import defaultdict
import random


class Network:
    """(nie taka) prosta siec, czysty python

    Ta siec jest napisana w sposob sekwencyjny (petle), wiec jest na maxa wolna
    w obliczeniach. Nie bez powodu normalnie wykorzystuje sie macierze. Ale tak
     jest poniekad latwiej zrozumiec feed-forward i backpropagation.
    """

    # TODO: split file into clear code file and jupyter notebook with comments
    def __init__(self, layer_sizes=(2, 4, 2)):
        """Inicjalizacja parametrow
        :param layer_sizes: ilosc neuronow w kazdej warstwie
        """
        self.neurons_on_layer = layer_sizes
        self.num_layers = len(layer_sizes)
        self.output_layer = self.num_layers - 1
        self.weights = [[]]  # empty list for input layer's neurons!
        self.biases = [[]]  # empty list for input layer's neurons!
        self.init_weights_biases()

    def init_weights_biases(self):
        """inicjalizacja wag i biasów

        zauwaz, ze wagi dotycza polaczen miedzy warstwami, wiec w naszym zapisie
        nie wystepuja dla warstwy zerowej (weights[1] bedzie oznaczac wagi na
        polaczeniach pomiedzy 0 a 1 warstwa)
        """

        for layer in range(1, self.num_layers):
            layer_biases = []
            layer_weights = []
            for node in range(self.neurons_on_layer[layer]):
                layer_biases.append(random.uniform(0, 0.5))
                # weight is for each "incoming" connection c...
                connections = range(self.neurons_on_layer[layer - 1])
                node_weights = [random.uniform(0, 0.5) for c in connections]
                layer_weights.append(node_weights)
            self.biases.append(layer_biases)
            self.weights.append(layer_weights)

    @staticmethod
    def calc_loss(y_predicted, y_true):
        """Obliczenie wartości "pomyłki" w predykcji

        :param y_predicted: wartosc otrzymana przez siec
        :param y_true: prawidłowa wartosc, 1 lub 0
        :return: lista - każdy element to wynik funkcji loss dla poszczególnego
        output neurona
        """
        return [1/2 * (y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_predicted)]

    @staticmethod
    def activ_fn_derivative(x):
        """Obliczenie pochodnej funkcji aktywacji"""
        return 1 - tanh(x) ** 2

    def forward(self, x):
        """Obliczenie wyjść (aktywacji) z neuronow

        dla zadanego wejscia, przy ustalonych wagach i biasach
        :param x: wektor danych wejsciowych
        :return all_z: # wektor Zetów; z to wazona suma + bias w kazdym neuronie
        :return all_activations: # "wyjścia" z neuronow.
        """

        all_activations = [x]   # x jest "wyjsciem" z warstwy 0
        all_z = [[]]

        for layer in range(1, self.num_layers):
            a_layer = []
            z_layer = []
            for neuron in range(self.neurons_on_layer[layer]):
                weighted_sum = 0
                for i, weight in enumerate(self.weights[layer][neuron]):
                    weighted_sum += weight * all_activations[layer-1][i]
                
                z = weighted_sum + self.biases[layer][neuron]
                a = tanh(z)
                z_layer.append(z)
                a_layer.append(a)
            all_z.append(z_layer)
            all_activations.append(a_layer)

        return all_z, all_activations

    def backprop(self, all_z, all_activations, y_true, l_rate):
        """Propagacja wsteczna

        Poprawianie wartosci wag i biasow w kazdym nodzie, w zaleznosci od tego
        jaki wplyw maja na wynik koncowy i w zaleznosci od samego wyniku koncowego
        :param all_z: wektor Zetów
        :param all_activations: wektor wszystkich "wyjść" (aktywacji) z neuronów
        :param y_true: wektor prawdziwych wartości dla wyjść
        :param l_rate: "predkosc uczenia"
        """

        gradient = {'weights': defaultdict(list),
                    'biases': defaultdict(list)}

        for layer in range(self.num_layers - 1, 0, -1):
            for neuron in range(self.neurons_on_layer[layer]):
                if layer == self.output_layer:
                    # first term is derivative of our cost function
                    delta = (all_activations[layer][neuron] - y_true[neuron]) *\
                            self.activ_fn_derivative(all_z[layer][neuron])

                else:   # hidden layers
                    # delta = tanh_derivative * sum[for each neuron in layer+1](delta_j * weight_jk)
                    # where weight_jk is the one between current neuron k and neuron j from layer+1
                    # delta_j is delta for the neuron j from layer+1
                    # gradient['biases'] is the same as delta_j so we use it instead of remembering deltas
                    sum_of_weighted_output_error = 0
                    for next_neuron in range(self.neurons_on_layer[layer + 1]):
                        sum_of_weighted_output_error += gradient['biases'][layer + 1][0] *\
                                                        self.weights[layer + 1][next_neuron][neuron]

                    delta = sum_of_weighted_output_error * self.activ_fn_derivative(all_z[layer][neuron])

                # grad for output layer biases is just the delta
                gradient['biases'][layer].append(delta)

                weights_partial_derivatives = []
                # grad for a certain weight is the delta times activation that 'this weight weights'
                for prev_neuron in range(self.neurons_on_layer[layer - 1]):
                    weights_partial_derivatives.append(delta * all_activations[layer - 1][prev_neuron])
                gradient['weights'][layer].append(weights_partial_derivatives)

        # update parameters
        for layer in range(1, self.num_layers):
            for neuron in range(self.neurons_on_layer[layer]):
                self.biases[layer][neuron] -= l_rate * gradient['biases'][layer][neuron]
                for w in range(self.neurons_on_layer[layer - 1]):
                    self.weights[layer][neuron][w] -= l_rate * gradient['weights'][layer][neuron][w]
