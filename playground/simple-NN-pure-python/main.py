"""ten skrypt odpalamy"""
from sklearn import datasets
from .Network import Network

# inicjalizacja wszystkiego
set_size = 100
X, Y = datasets.make_moons(set_size, noise=0.1)
trainX, trainY = X[:0.8 * set_size], Y[:0.8 * set_size]
testX, testY = X[0.8 * set_size:], Y[0.8 * set_size:]

network = Network(sizes=(2, 4, 2))

# uczenie sieci
for index, x in enumerate(trainX):
    y_predicted = network.forward(x)
    loss = network.calc_loss(y_predicted, trainY[index])
    network.backprop(loss)

# ewaluacja na zbiorze testowym
accuracy_sum = 0
for index, x in enumerate(testX):
    y_predicted = network.forward(x)
    predicted_class = y_predicted.index(max(y_predicted))
    accuracy_sum += not (predicted_class ^ testY[index]) # ^ oznacza XOR

print(accuracy_sum / len(testX))
