from machine_learning.PyTorch_Example.regression_problem import Net
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime

# __PATH__ = 'neural_network_parameters.pt'
# __PATH__ = 'less_examplex_neural_network_parameters.pt'
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

example = len(training_in)
examples = range(1000, example + 1000, 1000)
__PATH__ = f'{example}_examples_neural_network_parameters.pt'

testing_in = inputs[3 * len(inputs) // 4::2]
testing_out = targets[3 * len(targets) // 4::2]

cross_in = inputs[3 * len(inputs) // 4 + 1::2]
cross_out = targets[3 * len(inputs) // 4 + 1::2]

N = 5
learning_rate = 0.01
momentum = 0.8
# number_layers = 3
number_layers = 5
# number_nodes = 20
number_nodes = 50
architecture = (inputs.shape[1], *[number_nodes] * number_layers, N)
net = Net(architecture).double()
net.regression = False
predictions = []
training_predictions = []
train = False
save = False
load = True
mae_loss = torch.nn.L1Loss()

if load:
    errors = []
    for example in examples:
        __PATH__ = f'{example}_examples_neural_network_parameters.pt'
        net.load_state_dict(torch.load(__PATH__))
        errors.append(net.loss(net(testing_in), testing_out[:, :N]).item())
    plt.plot(examples, errors)
    plt.xlabel('Training set size')
    plt.ylabel('Mean square error on testing set')
    plt.savefig('harmonic_oscillator/set_size.pdf')
    plt.show()
    exit()

for example in examples:
    __PATH__ = f'{example}_examples_neural_network_parameters.pt'
    if train:
        train_loss = []
        cv_loss = []
        writer = SummaryWriter(
            f'runs/training_examples/architectur_5_20-Exmpales_{example}')
        # net.reset()
        # writer = SummaryWriter(f'runs/harmonic_oscillator/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
        try:
            threshold = 1e-6
            net.mini_batch_training(training_in[:example], training_out[:example, :N], MAX_ITER=1, verbose=False,
                                    learning_rate=learning_rate, momentum=momentum, batch_size=5)
            error = net.loss(net(training_in[:example]), training_out[:example, :N]).item()
            error_diff = error
            i = 0
            while error_diff > threshold:
                net.mini_batch_training(training_in[:example], training_out[:example, :N], MAX_ITER=1, verbose=False,
                                        learning_rate=learning_rate, momentum=momentum, batch_size=5)
                new_error = net.loss(net(training_in[:example]), training_out[:example, :N]).item()
                error_diff = abs(error - new_error)
                error = new_error
                print(f'examples = {example}, i = {i}, diff = {error_diff}')
                # train_loss.append(net.loss(net(training_in), training_out[:, :N]).item())
                writer.add_scalar('training loss (mse)',
                                  net.loss(net(training_in[:example]), training_out[:example, :N]).item(), i)
                writer.add_scalar('training loss (mae)',
                                  mae_loss(net(training_in[:example]), training_out[:example, :N]).item(), i)
                # cv_loss.append(net.loss(net(cross_in), cross_out[:, :N]).item())
                writer.add_scalar('cross validation loss (mse)', net.loss(net(cross_in), cross_out[:, :N]).item(), i)
                writer.add_scalar('cross validation loss (mae)', mae_loss(net(cross_in), cross_out[:, :N]).item(), i)
                i += 1
        except KeyboardInterrupt:
            pass

    if save:
        torch.save(net.state_dict(), __PATH__)

# plt.plot(train_loss, label='Trainings Set')
# plt.plot(cv_loss, label='Cross Validation Set')
# plt.legend()
# plt.show()

bins = [i for i in range(N + 1)]
print(f'mae on testing set: {mae_loss(net(training_in), training_out[:, :N]).item()}')
print(f'mse on testing set: {net.loss(net(training_in), training_out[:, :N]).item()}')
for i in range(10):
    plt.figure()
    plt.subplot(211)
    plt.hist(bins[:-1], bins, weights=testing_out[i, :N], align='left')
    plt.title('Testing Data')

    plt.subplot(212)
    plt.hist(bins[:-1], bins, weights=net(testing_in[i]).detach(), align='left')
    plt.title('Neural Network Predictions')

    plt.gcf().tight_layout()
    plt.savefig(f'harmonic_oscillator/neural_network/prediction{i + 1}.pdf')
    plt.show()
