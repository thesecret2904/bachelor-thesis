from machine_learning.PyTorch_Example.regression_problem import Net
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from crank_nicolson import Stepper
import datetime
import tikzplotlib

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
# __PATH__ = f'{example}_examples_neural_network_parameters.pt'
__PATH__ = f'network_3_20.pt'

testing_in = inputs[3 * len(inputs) // 4::2]
testing_out = targets[3 * len(targets) // 4::2]

cross_in = inputs[3 * len(inputs) // 4 + 1::2]
cross_out = targets[3 * len(inputs) // 4 + 1::2]

N = 5
learning_rate = 0.01
momentum = 0.8
number_layers = 3
# number_layers = 5
number_nodes = 20
# number_nodes = 50
architecture = (inputs.shape[1], *[number_nodes] * number_layers, N)
net = Net(architecture).double()
net.regression = False
predictions = []
training_predictions = []
train = True
save = False
load = False
mae_loss = torch.nn.L1Loss()

if load:
    # errors = []
    # for example in examples:
    #     __PATH__ = f'{example}_examples_neural_network_parameters.pt'
    #     net.load_state_dict(torch.load(__PATH__))
    #     errors.append(net.loss(net(testing_in), testing_out[:, :N]).item())
    # plt.plot(examples, errors)
    # plt.xlabel('Training set size')
    # plt.ylabel('Mean square error on testing set')
    # plt.savefig('harmonic_oscillator/set_size.pdf')
    # plt.show()
    # __PATH__ = 'new_neural_network_parameters.pt'
    # times = np.linspace(0, 6, 61)
    # # number_layers = 3
    # number_layers = 5
    # # number_nodes = 20
    # number_nodes = 50
    # architecture = (times.shape[0], *[number_nodes] * number_layers, N)
    # net = Net(architecture).double()
    # net.regression = False
    net.load_state_dict(torch.load(__PATH__))
    #
    #
    # def V(x, t):
    #     return 0.5 * np.square(x)
    #
    #
    # # number of sample points
    # N = 1000
    # # x domain
    # x_min = -50
    # x_max = 50
    # # starting point
    # t = 0
    # t_max = 6
    # # create list of all x points
    # x = np.linspace(x_min, x_max, N, endpoint=True)
    # # time step
    # dt = 0.1
    # # calculate initial state
    # # current_state = psi(x)
    #
    # # create the Stepper which propagates times
    # stepper = Stepper(None, t, x, V)
    # stepper.t_max = t_max
    # time_shift = t_max / 2
    # energys, eigenstates = stepper.get_stationary_solutions(k=250)
    #
    # for i in range(10):
    #     parameters_max = np.array([10., 5., 10., 10., 10.])
    #     parameters = np.random.rand(len(parameters_max)) * parameters_max
    #
    #     frequencies = [(1, parameters[2]), (parameters[3], parameters[4])]
    #     stepper.set_electric_field(parameters[0], time_shift, parameters[1], frequencies)
    #
    #     stepper.set_time(0)
    #     stepper.set_state(eigenstates[0], normalzie=False)
    #     stepper.step_to(t_max, dt)
    #     projections = stepper.projection(eigenstates)
    #     projections = np.real(np.conj(projections) * projections)
    #     N = 5
    #     print(projections[:N])
    #
    #     input = np.zeros_like(times)
    #     for j in range(len(times)):
    #         input[j] = stepper.E([0], times[j])[0]
    #
    #     prediction = net(torch.tensor(np.nan_to_num((input - means) / vars))).detach()
    #     plt.figure()
    #     bins = [i for i in range(N + 1)]
    #     plt.subplot(211)
    #     plt.hist(bins[:-1], bins, weights=projections[:N], align='left')
    #     plt.title('Crank-Nicolson')
    #
    #     plt.subplot(212)
    #     plt.hist(bins[:-1], bins, weights=prediction, align='left')
    #     plt.title('Neural Network Predictions')
    #
    #     plt.gcf().tight_layout()
    #     plt.show()
    # exit()

# for example in examples:
#     __PATH__ = f'{example}_examples_neural_network_parameters.pt'
testing_loss = []
if train:
    train_loss = []
    cv_loss = []
    # writer = SummaryWriter(
    #     f'runs/paramter_vs_field_training/architectur_5_20-parameters')
    # net.reset()
    sizes = range(1000, len(training_in), 1000)
    for size in sizes:
        writer = SummaryWriter(f'runs/set_size/{size}')
        net = Net(architecture).double()
        net.regression = False
        try:
            threshold = 1e-6
            MAX_ITER = 1000
            error = net.loss(net(training_in[:size]), training_out[:size, :N]).item()
            error_diff = error
            i = 0
            while (error_diff > threshold or i < 100) and i < MAX_ITER:
                net.mini_batch_training(training_in[:size], training_out[:size, :N], MAX_ITER=1, verbose=False,
                                        learning_rate=learning_rate, momentum=momentum, batch_size=5)
                new_error = net.loss(net(training_in[:size]), training_out[:size, :N]).item()
                error_diff = abs(error - new_error)
                error = new_error
                print(f'i = {i}, diff = {error_diff}')
                # train_loss.append(net.loss(net(training_in), training_out[:, :N]).item())
                writer.add_scalar('training loss (mse)', net.loss(net(training_in[:size]), training_out[:size, :N]).item(), i)
                writer.add_scalar('training loss (mae)', mae_loss(net(training_in[:size]), training_out[:size, :N]).item(), i)
                # cv_loss.append(net.loss(net(cross_in), cross_out[:, :N]).item())
                writer.add_scalar('cross validation loss (mse)', net.loss(net(cross_in), cross_out[:, :N]).item(), i)
                writer.add_scalar('cross validation loss (mae)', mae_loss(net(cross_in), cross_out[:, :N]).item(), i)
                i += 1
        except KeyboardInterrupt:
            pass
        testing_loss.append(net.loss(net(testing_in), testing_out[:, :N]).item())
    plt.plot(sizes, testing_loss)
    plt.xlabel('Training set size')
    plt.ylabel('MSE on testing set')
    plt.savefig('harmonic_oscillator/set_size.pdf')
    tikzplotlib.save('harmonic_oscillator/set_size.tex')
    plt.show()

    # for number_nodes in range(10, 110, 10):
    #     writer = SummaryWriter(
    #              f'runs/architecture_test/architectur_{number_layers}_{number_nodes}')
    #     architecture = (inputs.shape[1], *[number_nodes] * number_layers, N)
    #     net = Net(architecture).double()
    #     net.regression = False
    #     net.mini_batch_training(training_in[:example], training_out[:example, :N], MAX_ITER=1, verbose=False,
    #                             learning_rate=learning_rate, momentum=momentum, batch_size=5)
    #     error = net.loss(net(training_in[:example]), training_out[:example, :N]).item()
    #     error_diff = error
    #     i = 0
    #     while error_diff > threshold and i < MAX_ITER:
    #         net.mini_batch_training(training_in[:example], training_out[:example, :N], MAX_ITER=1, verbose=False,
    #                                 learning_rate=learning_rate, momentum=momentum, batch_size=5)
    #         new_error = net.loss(net(training_in[:example]), training_out[:example, :N]).item()
    #         error_diff = abs(error - new_error)
    #         error = new_error
    #         print(f'nodes = {number_nodes}, i = {i}, diff = {error_diff}')
    #         # train_loss.append(net.loss(net(training_in), training_out[:, :N]).item())
    #         writer.add_scalar('training loss (mse)',
    #                           net.loss(net(training_in[:example]), training_out[:example, :N]).item(), i)
    #         writer.add_scalar('training loss (mae)',
    #                           mae_loss(net(training_in[:example]), training_out[:example, :N]).item(), i)
    #         # cv_loss.append(net.loss(net(cross_in), cross_out[:, :N]).item())
    #         writer.add_scalar('cross validation loss (mse)', net.loss(net(cross_in), cross_out[:, :N]).item(), i)
    #         writer.add_scalar('cross validation loss (mae)', mae_loss(net(cross_in), cross_out[:, :N]).item(), i)
    #         i += 1
    #     testing_loss.append(net.loss(net(testing_in), testing_out[:, :N]).item())

if save:
    torch.save(net.state_dict(), __PATH__)

# plt.plot(train_loss, label='Trainings Set')
# plt.plot(cv_loss, label='Cross Validation Set')
# plt.legend()
# plt.show()

bins = [i for i in range(N + 1)]
print(f'mae on testing set: {mae_loss(net(testing_in), testing_out[:, :N]).item()}')
print(f'mse on testing set: {net.loss(net(testing_in), testing_out[:, :N]).item()}')
for i in range(10):
    plt.figure()
    plt.subplot(211)
    plt.xlabel('State')
    plt.ylabel('Occupation')
    plt.hist(bins[:-1], bins, weights=testing_out[i, :N], align='left')
    plt.title('Testing Data')

    plt.subplot(212)
    plt.xlabel('State')
    plt.ylabel('Occupation')
    plt.hist(bins[:-1], bins, weights=net(testing_in[i]).detach(), align='left')
    plt.title('Neural Network Predictions')

    plt.gcf().tight_layout()
    tikzplotlib.save(f'harmonic_oscillator/neural_network/prediction{i + 1}.tex')
    plt.savefig(f'harmonic_oscillator/neural_network/prediction{i + 1}.pdf')
    # plt.show()
