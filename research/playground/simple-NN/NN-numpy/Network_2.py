import numpy as np


# Suppress Pycharm false dict warnings
# Suppress PyPep8Naming as we want to use uppercase for matrices
# noinspection PyTypeChecker,PyPep8Naming
class Network2:
    """ Prosta sieć z użyciem Numpy do obliczeń macierzowych

    Ta siec jest napisana przy pomocy zapisu macierzowego, który pozwala na uniknięcie wielu pętl z pierwszej
    implementacji i zwiększa czytelność kodu (dla osób, która rozumieją iterację pierwszą i znają rachunek macierzowy)
    """

    def __init__(self, layer_sizes=(2, 4, 2)):
        """Inicjalizacja parametrow
        :param layer_sizes: ilosc neuronow w kazdej warstwie
        """
        self.neurons_on_layer = layer_sizes
        self.num_layers = len(layer_sizes)
        self.output_layer = self.num_layers - 1
        self.Ws = dict()
        self.Bs = dict()
        self.init_weights_biases()

    def init_weights_biases(self):
        """ Inicjalizacja wag i biasów
        """
        for layer in range(1, self.num_layers):
            self.Ws[layer] = np.random.random((self.neurons_on_layer[layer - 1], self.neurons_on_layer[layer]))
            self.Bs[layer] = np.random.random(self.neurons_on_layer[layer])

    @staticmethod
    def calc_loss(y_predicted, y_true):
        """Obliczenie wartości "pomyłki" w predykcji

        :param y_predicted: np array - aktywacje z output layer
        :param y_true: prawidłowe wartości dla output layer
        :return: np array - każdy element to wynik funkcji loss dla poszczególnego
        output neurona
        """
        return np.square((y_true - y_predicted) / 2)

    @staticmethod
    def activ_fn_derivative(z):
        """Obliczenie pochodnej funkcji aktywacji"""
        return 1 - np.square(np.tanh(z))

    def forward(self, x):
        """Obliczenie wyjść (aktywacji) z neuronow
        dla zadanego wejscia, przy ustalonych wagach i biasach

        :param x: wektor danych wejsciowych
        :return Zs: # wektor Zetów; z to wazona suma + bias w kazdym neuronie
        :return As: # "wyjścia" z neuronow.
        """
        As, Zs = dict(), dict()
        As[0] = np.array(x)

        for layer in range(1, self.num_layers):
            Z = As[layer-1].dot(self.Ws[layer]) + self.Bs[layer]
            Zs[layer] = Z
            As[layer] = np.tanh(Z)

        return Zs, As

    def backprop(self, Zs, As, y_true, l_rate):
        """Propagacja wsteczna

        Poprawianie wartosci wag i biasow w kazdym nodzie, w zaleznosci od tego
        jaki wplyw maja na wynik koncowy i w zaleznosci od samego wyniku koncowego
        :param Zs: wektor Zetów
        :param As: wektor wszystkich "wyjść" (aktywacji) z neuronów
        :param y_true: wektor prawdziwych wartości dla wyjść
        :param l_rate: "predkosc uczenia"
        """

        gradient = {'weights': dict(),
                    'biases': dict()}

        # output layer
        layer = self.num_layers - 1
        delta = np.multiply((np.subtract(As[layer], y_true)), self.activ_fn_derivative(Zs[layer]))

        gradient['biases'][layer] = delta
        gradient['weights'][layer] = np.multiply(As[layer - 1].reshape((-1, 1)), delta)

        # hidden layers
        for layer in range(self.num_layers - 2, 0, -1):
            weighted_deltas = (self.Ws[layer + 1]).dot(gradient["biases"][layer + 1])  # deltas from next layer
            delta = np.multiply(weighted_deltas, self.activ_fn_derivative(Zs[layer]))
            gradient['biases'][layer] = delta
            gradient['weights'][layer] = np.multiply(As[layer - 1].reshape((-1, 1)), delta)

        # update parameters
        for layer in range(1, self.num_layers):
            self.Bs[layer] -= l_rate * gradient['biases'][layer]  # note: no learning rate
            self.Ws[layer] -= l_rate * gradient['weights'][layer]

