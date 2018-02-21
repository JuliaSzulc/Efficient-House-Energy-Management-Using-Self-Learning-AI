"""ten skrypt odpalamy"""
from sklearn import datasets
from Network import Network
import random

random.seed(1)


# inicjalizacja wszystkiego
l_rate = 0.01
set_size = 10000
X, Y = datasets.make_moons(set_size, noise=0)
split_ratio = 0.90
train_set_size = int(split_ratio * set_size)
trainX, trainY = X[:train_set_size], Y[:train_set_size]
testX, testY = X[train_set_size:], Y[train_set_size:]

l_sizes = (2, 4, 2)
network = Network(l_sizes)

# uczenie sieci
print("Learning the net...")
for index, x in enumerate(trainX):
    if index % 1000 == 0:
        print("{0:.2f} %".format(round(index/train_set_size * 100, 2)))
    zs, activations = network.forward(x)
    y_true = [0] * l_sizes[-1]
    # zamiast y które wskazuje klasę, tworzę wektor mówiący o oczekiwanej wartości dla danego output neorun
    y_true[trainY[index]] = 1
    network.backprop(zs, activations, y_true, network.tanh_derivative, l_rate)

print("Learning done.")
# ewaluacja na zbiorze testowym

print("Testing the net...")
accuracy_sum = 0
for index, x in enumerate(testX):
    zs, activations = network.forward(x)
    predicted_class = activations[-1].index(max(activations[-1]))
    accuracy_sum += not (predicted_class ^ testY[index]) # ^ oznacza XOR

print("Testing done. Score:")
print(accuracy_sum / len(testX))
