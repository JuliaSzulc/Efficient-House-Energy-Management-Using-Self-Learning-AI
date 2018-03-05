from torch import autograd, nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn import datasets
import torch.nn.functional as F
import torch

class MoonsDataset(Dataset):
    """Random Moons dataset."""

    def __init__(self, num_samples, noise, random_state):
        X, Y = datasets.make_moons(num_samples, noise, random_state)
        y_true = list()
        for i in range(num_samples):
            y_true.append([0, 1] if Y[i] == 1 else [1, 0])
        self.X = X
        self.Y = y_true

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.Y[idx])


class Net(nn.Module):
    """ Prosta sieć w PyTorch (3-layer)
    """
    def __init__(self, i_size, h_size, o_size):
        super().__init__()
        self.fc1 = nn.Linear(i_size, h_size)
        self.fc2 = nn.Linear(h_size, o_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


# prepare moons dataset with train and test dataloaders
train_set_size = 1000
test_set_size = 100
input_size = 2
num_classes = 2
batch_size = 5

train_dataset = MoonsDataset(num_samples=train_set_size, noise=0.2, random_state=4)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

test_dataset = MoonsDataset(num_samples=test_set_size, noise=0.2, random_state=5)
test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=False, num_workers=1)

# prepare network
l_rate = 0.005
hidden_size = 10
network = Net(input_size, hidden_size, num_classes)
optimizer = optim.SGD(network.parameters(), lr=l_rate, momentum=0.9)

# trenujemy i testujemy
num_epochs = 10
for epoch in range(num_epochs):
    print("Epoch {}:".format(epoch))

    print("Learning...")
    for i, (x_batch, y_batch) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = network(autograd.Variable(x_batch))
        loss = F.mse_loss(output, autograd.Variable(y_batch))
        loss.backward()
        optimizer.step()

    correct = 0
    total = 0
    for data in test_dataloader:
        x, y = data
        outputs = network(autograd.Variable(x))
        predicted = torch.max(outputs.data, 1)[1]
        total += y.size(0)
        correct += (predicted == y.long()).sum()

    print('Accuracy: %d %%' % (100 * correct / total))

