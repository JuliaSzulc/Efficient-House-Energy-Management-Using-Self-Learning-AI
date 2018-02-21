"""ten skrypt odpalamy"""
from sklearn import datasets
from Network import Network
import random

random.seed(1)


# inicjalizacja wszystkiego
set_size = 100
X, Y = datasets.make_moons(set_size, noise=0.1)
trainX, trainY = X[:int(0.8 * set_size)], Y[:int(0.8 * set_size)]
testX, testY = X[int(0.8 * set_size):], Y[int(0.8 * set_size):]

network = Network(l_sizes=(2, 4, 2))

print(network.forward(trainX[0]))

exit(1)

# uczenie sieci
for index, x in enumerate(trainX):
    activations = network.forward(x)
    loss = network.calc_loss(activations[-1], trainY[index])
    network.backprop(loss, activations)

# ewaluacja na zbiorze testowym
accuracy_sum = 0
for index, x in enumerate(testX):
    y_predicted = network.forward(x)
    predicted_class = y_predicted.index(max(y_predicted))
    accuracy_sum += not (predicted_class ^ testY[index]) # ^ oznacza XOR

print(accuracy_sum / len(testX))
