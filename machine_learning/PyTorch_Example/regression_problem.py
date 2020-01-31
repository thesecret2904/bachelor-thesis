import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, architecture, regression_problem=True):
        super(Net, self).__init__()
        self.regression = regression_problem
        self.layers = []
        for i in range(len(architecture) - 1):
            self.layers.append(nn.Linear(architecture[i], architecture[i + 1]))
        self.layers = nn.ModuleList(self.layers)
        if regression_problem:
            self.loss_func = nn.MSELoss()
        else:
            self.loss_func = nn.BCELoss()
        self.parameter_regression = 0.

        def loss(x, y):
            if self.parameter_regression > 0:
                extra = 0
                params = list(self.parameters())
                for i in range(0, len(params), 2):
                    extra += (params[i] ** 2).sum()
                return self.loss_func(x, y) * 10 + self.parameter_regression / (2 * x.size()[0]) * extra
            else:
                return self.loss_func(x, y) * 10

        self.loss = loss
        self.original_parameters = self.state_dict().copy()
        self.optimizer = None

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = torch.sigmoid(self.layers[i](x))
        if self.regression:
            x = self.layers[-1](x)
        else:
            x = torch.sigmoid(self.layers[-1](x))
        return x

    def train_on_set(self, x, y, learning_rate=0.01, MAX_ITER=5000, verbose=False, momentum=0):
        shuffle = [i for i in range(x.size()[0])]
        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(self.parameters(), learning_rate, momentum=momentum)
        for i in range(MAX_ITER):
            np.random.shuffle(shuffle)
            x = x[shuffle]
            y = y[shuffle]

            self.optimizer.zero_grad()
            loss = self.loss(self(x), y)
            loss.backward()
            if verbose:
                print(loss.item())
            self.optimizer.step()

    def mini_batch_training(self, x, y, learning_rate=0.01, MAX_ITER=5000, verbose=False, momentum=0, batch_size=1):
        shuffle = [i for i in range(x.size()[0])]
        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(self.parameters(), learning_rate, momentum=momentum)
        for j in range(MAX_ITER):
            np.random.shuffle(shuffle)
            x = x[shuffle]
            y = y[shuffle]
            for i in range(0, x.size()[0], batch_size):
                self.optimizer.zero_grad()
                loss = self.loss(self(x[i:min(i + batch_size, x.size()[0])]), y[i:min(i + batch_size, x.size()[0])])
                loss.backward()
                if verbose:
                    print(loss.item())
                self.optimizer.step()

    def reset(self):
        self.load_state_dict(self.original_parameters.copy())


if __name__ == '__main__':
    N = 500
    x = np.ones((1, N)) * np.linspace(0, 1, N)
    t = np.sin(2 * np.pi * x) + np.cos(4 * np.pi * x) + np.random.randn(N) * 0.2
    model = (np.sin(2 * np.pi * x) + np.cos(4 * np.pi * x)).T

    x = x.T
    t = t.T

    x = (x - x.mean(axis=0)) / x.var(axis=0)
    t = (t - t.mean(axis=0)) / t.var(axis=0)

    x = torch.tensor(x)
    t = torch.tensor(t)

    plt.plot(x, t, '.')
    plt.show()

    training_inputs = x[0::2, :]
    testing_inputs = x[1::4, :]
    validation_inputs = x[3::4, :]

    training_targets = t[0::2, :]
    testing_targets = t[1::4, :]
    validation_targets = t[3::4, :]

    net = Net((1, 8, 1)).double()
    net.parameter_regression = 0.

    '''training_loss = []
    validation_loss = []
    for i in np.linspace(0, 1, 50):
        net.reset()
        net.parameter_regression = i
        net.train_on_set(training_inputs, training_targets, MAX_ITER=5000, learning_rate=0.01)
        training_loss.append(net.loss_func(net(training_inputs), training_targets))
        validation_loss.append(net.loss_func(net(validation_inputs), validation_targets))
        print(f'{i} / {10}')

    plt.plot(np.linspace(0, 10, 50), training_loss, label='Training set')
    plt.plot(np.linspace(0, 10, 50), validation_loss, label='Validation set')
    plt.legend()
    plt.show()'''

    net.train_on_set(training_inputs, training_targets, learning_rate=0.01, MAX_ITER=5000)
    plt.plot(testing_inputs, testing_targets, '.', label='testing set')
    plt.plot(testing_inputs, net(testing_inputs).detach(), label='neural network')
    plt.plot(x, model, label='model function')
    plt.legend()
    plt.show()
