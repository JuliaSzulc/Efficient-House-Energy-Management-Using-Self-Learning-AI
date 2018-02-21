from random import randrange
from math import tanh
import random
random.seed(1)


class Network:
    """(nie taka) prosta siec, czysty python

    Ta siec jest napisana w sposob sekwencyjny (petle), wiec jest na maxa wolna
    w obliczeniach. Nie bez powodu normalnie wykorzystuje sie macierze. Ale tak
     jest poniekad latwiej zrozumiec
    """

    def __init__(self, layer_sizes=(2, 4, 2)):
        """Inicjalizacja parametrow
        :param layer_sizes: ilosc neuronow w kazdej warstwie
        """
        self.neurons_on_layer = layer_sizes
        self.num_layers = len(layer_sizes)
        self.num_hidden = self.num_layers - 2
        self.weights = []
        self.biases = []
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
        x : wektor danych wejsciowych
        """

        all_activations = [x]   # "wyjścia" z neuronow. x jest "wyjsciem" z warstwy 0
        all_z = []             # wektor Zetów; z to wazona suma + bias w kazdym neuronie

        for layer in range(1, self.num_layers):
            a_layer = []
            z_layer = []
            for neuron in range(self.neurons_on_layer[layer]):
                weighted_sum = 0
                for i, weight in enumerate(self.weights[layer-1][neuron]):
                    weighted_sum += weight * all_activations[layer-1][i]
                
                z = weighted_sum + self.biases[layer-1][neuron]
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

        # TODO: backprop for any num of layers. now for 3 layers

        # output layer gradients
        # NOTE: this part works for any number of hidden layers because of 'minus indexing'
        # TODO: maybe we can do 1 loop for output and hidden layers? or maybe not
        ob_gradients = []
        ow_gradients = []
        # for every neuron in output layer
        for neuron in range(self.neurons_on_layer[-1]):
            # calculate delta - equation in the blog post
            # first term is derivative of our cost funtion
            # Note: all_z[-1] == all_z[1] for 3 layers, because {len(all_z) = len(all_activations)-1},
            # because we don't have all_z for input layer
            delta = (all_activations[-1][neuron] - y_true[neuron]) * self.activ_fn_derivative(all_z[-1][neuron])
            # grad for output layer biases is just the delta
            ob_gradients.append(delta)

            # grad for a certain weight is the delta times activation that 'this weight weights'
            # Note: this fragment works for any num of hidden layers because of '-2' indexing
            neuron_w_gradients = []
            for prev_node in range(self.neurons_on_layer[-2]):
                neuron_w_gradients.append(delta * all_activations[-2][prev_node])
            ow_gradients.append(neuron_w_gradients)

        # hidden layer
        # TODO: try to make a loop to make it work for any number of hidden layers

        hb_gradients = []
        hw_gradients = []

        # TODO: self.l_sizes[1] works only if we have 3 layers
        layer = 1
        for neuron_k in range(self.neurons_on_layer[layer]):
            # calculate delta. It is more complicated this time (see blog post). its:
            # delta = tanh_derivative * sum[for each neuron in layer+1](delta_j * weight_jk)
            # where weight_jk is the one between current neuron k and neuron j from layer+1
            # delta_j is delta for the neuron j from layer+1
            sum_of_weighted_output_error = 0
            for neuron_j in range(self.neurons_on_layer[layer + 1]):
                # ob_gradients[0] is the same as delta_j so we use it instead of remembering deltas
                # NOTE: weights[layer] are weights of next layer, not current! coz we don't have weights for input layer
                sum_of_weighted_output_error += ob_gradients[0] * self.weights[layer][neuron_j][neuron_k]
            delta = sum_of_weighted_output_error * self.activ_fn_derivative(all_z[0][neuron_k])
            hb_gradients.append(delta)

            # gradients of hidden weights - see blog post. We use the current delta and activations from last layer
            neuron_hw_gradients = []
            for prev_node in range(self.neurons_on_layer[layer - 1]):
                neuron_hw_gradients.append(delta * all_activations[layer - 1][prev_node])
            hw_gradients.append(neuron_hw_gradients)

        # update parameters
        # TODO: make it work for any number of layers, and try to merge loops into one.
        # output layer
        # this will do the trick: range(self.num_layers - 1, 0, -1)
        layer = 2
        for neuron in range(self.neurons_on_layer[layer]):
            self.biases[layer-1][neuron] -= l_rate * ob_gradients[neuron]
            for w in range(self.neurons_on_layer[layer - 1]):
                self.weights[layer-1][neuron][w] -= l_rate * ow_gradients[neuron][w]
        # hidden layer
        layer = 1
        for neuron in range(self.neurons_on_layer[layer]):
            self.biases[layer-1][neuron] -= l_rate * hb_gradients[neuron]
            for w in range(self.neurons_on_layer[layer - 1]):
                self.weights[layer-1][neuron][w] -= l_rate * hw_gradients[neuron][w]
