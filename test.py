from machine_learning.PyTorch_Example.regression_problem import Net
import numpy as np
import matplotlib.pyplot as plt
from crank_nicolson import Stepper
import torch

# path for saved parameters of neural network
__PATH__ = 'network_3_20.pt'
# __PATH__ = 'less_examplex_neural_network_parameters.pt'
# load trainings data for calculating feature scaling (for inputs)
inputs = np.load('shuffled_E_inputs.npy')
# need means and vars for correctly scaling new input for neural network
means = inputs.mean(axis=0)
vars = np.sqrt(inputs.var(axis=0))

del inputs

max_parameters = np.array([10, 2, 5, 5, 5])
max_parameters = np.array([1., 5., 1., 5.])

times = np.linspace(0, 6, 61)

number_layers = 3
number_nodes = 20
architecture = (times.shape[0], *[number_nodes] * number_layers, 5)
net = Net(architecture).double()
net.regression = False
# net.load_state_dict(torch.load(__PATH__))


def V(x, t):
    return 0.5 * np.square(x)


# number of sample points
N = 1000
# x domain
x_min = -50
x_max = 50
# starting point
t = 0
t_max = 6
# create list of all x points
x = np.linspace(x_min, x_max, N, endpoint=True)
# time step
dt = 0.1
# calculate initial state
# current_state = psi(x)

# create the Stepper which propagates times
stepper = Stepper(None, t, x, V)
stepper.t_max = t_max
time_shift = t_max / 2
energys, eigenstates = stepper.get_stationary_solutions(k=250)
iterations = 1000
diff = np.zeros((iterations, 5))

for i in range(iterations):
    parameters = np.random.rand(len(max_parameters)) * max_parameters
    if i == 0:
        parameters = np.array([0.6591, 0.6819, 0.7859, 4.4431])

    # frequencies = [(1, parameters[2]), (parameters[3], parameters[4])]
    # stepper.set_electric_field(parameters[0], time_shift, parameters[1], frequencies)
    stepper.set_electric_field2(parameters[0], [(1, parameters[1]), (parameters[2], parameters[3])])

    stepper.set_time(0)
    stepper.set_state(eigenstates[0], normalzie=False)
    stepper.step_to(t_max, dt)
    projections = stepper.projection(eigenstates)
    projections = np.real(np.conj(projections) * projections)
    N = 5
    print(parameters)

    input = np.zeros_like(times)
    for j in range(len(times)):
        input[j] = stepper.E([0], times[j])[0]

    prediction = net(torch.tensor(np.nan_to_num((input - means) / vars))).detach()
    diff[i] = prediction - projections[:N]

    # plt.figure()
    # bins = [i for i in range(N + 1)]
    # plt.subplot(211)
    # plt.hist(bins[:-1], bins, weights=projections[:N], align='left')
    # plt.title('Crank-Nicolson')
    #
    # plt.subplot(212)
    # plt.hist(bins[:-1], bins, weights=prediction, align='left')
    # plt.title('Neural Network Predictions')
    #
    # plt.gcf().tight_layout()
    # plt.show()

print(np.square(diff).mean())
