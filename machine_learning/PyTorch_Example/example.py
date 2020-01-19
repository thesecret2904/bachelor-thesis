import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(20 * 20, 25)
        self.layer2 = nn.Linear(25, 10)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x


# Create Neural-Network
net = Net()
# Cast network to double
net = net.double()

# Load trainings data
data = sio.loadmat('../ex4data1.mat')
inputs = torch.from_numpy(data['X'])

# Convert target labels 1 to 10 to corresponding vectors
targets = data['y']
template = [i + 1 for i in range(10)]
targets = torch.tensor([t == template for t in targets], dtype=torch.double)

shuffle = [i for i in range(len(inputs))]

# Shuffle up data
np.random.shuffle(shuffle)
targets = targets[shuffle]
inputs = inputs[shuffle]

# Separate training data
end = len(inputs) * 3 // 4
training_inputs = inputs[:end]
print(training_inputs.size())
testing_inputs = inputs[end:]
print(testing_inputs.size())
training_targets = targets[:end]
testing_targets = targets[end:]

# choose loss function
criterion = nn.BCELoss()
lamb = 1


# implement regularization
def reg_criterion(x, y):
    params = list(net.parameters())
    return criterion(x, y) * 10 + lamb / (2 * x.size()[0]) * ((params[0] ** 2).sum() + (params[2] ** 2).sum())


'''# load pre calculated weights
weights = sio.loadmat('../ex4weights.mat')
theta1 = torch.tensor(weights['Theta1'])
theta2 = torch.tensor(weights['Theta2'])

# set weights
net.state_dict()['layer1.weight'][:] = theta1[:, 1:]
net.state_dict()['layer1.bias'][:] = theta1[:, 0]
net.state_dict()['layer2.weight'][:] = theta2[:, 1:]
net.state_dict()['layer2.bias'][:] = theta2[:, 0]'''


def decision(tensor):
    out = torch.zeros_like(tensor)
    if len(out.size()) > 1:
        indices = tensor.argmax(dim=1)
        for i in range(len(indices)):
            out[i, indices[i]] = 1
    else:
        out[tensor.argmax().item()] = 1
    return out


# training loop
N = 1000
learning_rate = 0.01
losses = []
precisions = []
shuffle = [i for i in range(len(training_inputs))]
optimizer = torch.optim.SGD(net.parameters(), learning_rate)
for i in range(N):
    np.random.shuffle(shuffle)
    training_inputs = training_inputs[shuffle]
    training_targets = training_targets[shuffle]

    optimizer.zero_grad()
    loss = reg_criterion(net(training_inputs), training_targets)
    losses.append(loss)
    loss.backward()
    optimizer.step()
    precisions.append(1 - ((testing_targets - decision(net(testing_inputs))).abs()).mean())

print(f'precision = {1 - ((testing_targets - decision(net(testing_inputs))).abs()).mean():.2%}')
plt.plot(losses)
plt.show()
plt.plot(precisions)
plt.show()

# calculate loss
'''loss = reg_criterion(net(inputs), targets)

# zero gradients
net.zero_grad()

# calculate gradients
print('layer1.bias.grad before backward')
print(net.layer1.bias.grad)

loss.backward()

print('layer1.bias.grad after backward')
print(net.layer1.bias.grad)'''
