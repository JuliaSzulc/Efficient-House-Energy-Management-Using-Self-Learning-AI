import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
    """
        Simple Neural Network.

    """
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_neurons, hidden_neurons)
        self.fc2 = torch.nn.Linear(hidden_neurons, output_neurons)

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)   # Hyperbolic tangent activation function
        x = self.fc2(x)
        return F.softmax(x, dim=1)  # normalized exponential function
