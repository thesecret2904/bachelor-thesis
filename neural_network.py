from machine_learning.PyTorch_Example.regression_problem import Net
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime

__PATH__ = 'neural_network_parameters.pt'
inputs = np.load('shuffled_E_inputs.npy')
targets = np.load('shuffled_targets.npy')

means = inputs.mean(axis=0)
vars = np.sqrt(inputs.var(axis=0))
inputs = np.nan_to_num((inputs - means) / vars)

# targets = (targets - targets.mean(axis=0)) / np.sqrt(targets.var(axis=0))

inputs = torch.tensor(inputs)
targets = torch.tensor(targets)

training_in = inputs[:3 * len(inputs) // 4]
training_out = targets[:3 * len(targets) // 4]

testing_in = inputs[3 * len(inputs) // 4::2]
testing_out = targets[3 * len(targets) // 4::2]

cross_in = inputs[3 * len(inputs) // 4 + 1::2]
cross_out = targets[3 * len(inputs) // 4 + 1::2]

N = 5
learning_rate = 0.01
momentum = 0.8
number_layers = 3
number_nodes = 20
architecture = (inputs.shape[1], *[number_nodes] * number_layers, N)
net = Net(architecture).double()
net.regression = False
predictions = []
training_predictions = []
train = True
save = False
mae_loss = torch.nn.L1Loss()

if train:
    number_of_examples = range(5000, 80000, 10000)
    train_loss = []
    cv_loss = []
    writer = SummaryWriter(f'runs/harmonic_oscillator/states/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    for N in range(1, 11):
        architecture = (inputs.shape[1], *[number_nodes] * number_layers, N)
        net = Net(architecture).double()
        # net.reset()
        # writer = SummaryWriter(f'runs/harmonic_oscillator/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
        try:
            for i in range(25):
                net.mini_batch_training(training_in, training_out[:, :N], MAX_ITER=1, verbose=False,
                                        learning_rate=learning_rate, momentum=momentum, batch_size=5)
                print(i)
            # train_loss.append(net.loss(net(training_in), training_out[:, :N]).item())
            writer.add_scalar('training loss (mse)', net.loss(net(training_in), training_out[:, :N]).item(), N)
            writer.add_scalar('training loss (mae)', mae_loss(net(training_in), training_out[:, :N]).item(), N)
            # cv_loss.append(net.loss(net(cross_in), cross_out[:, :N]).item())
            writer.add_scalar('cross validation loss (mse)', net.loss(net(cross_in), cross_out[:, :N]).item(), N)
            writer.add_scalar('cross validation loss (mae)', mae_loss(net(cross_in), cross_out[:, :N]).item(), N)
            print('N = ', N)
        except KeyboardInterrupt:
            pass

    if save:
        torch.save(net.state_dict(), __PATH__)

    # plt.plot(train_loss, label='Trainings Set')
    # plt.plot(cv_loss, label='Cross Validation Set')
    # plt.legend()
    # plt.show()
else:
    net.load_state_dict(torch.load(__PATH__))

bins = [i for i in range(N + 1)]
for i in range(10):
    plt.figure()
    plt.subplot(211)
    plt.hist(bins[:-1], bins, weights=testing_out[i, :N])
    plt.title('Testing Data')

    plt.subplot(212)
    plt.hist(bins[:-1], bins, weights=net(testing_in[i]).detach())
    plt.title('Neural Network Predictions')

    plt.gcf().tight_layout()
    plt.show()
