from machine_learning.PyTorch_Example.regression_problem import Net
import numpy as np
import matplotlib.pyplot as plt
from crank_nicolson import Stepper
import torch
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random

# path for saved parameters of neural network
__PATH__ = 'neural_network_parameters.pt'
# load trainings data for calculating feature scaling (for inputs)
inputs = np.load('shuffled_E_inputs.npy')
# need means and vars for correctly scaling new input for neural network
means = inputs.mean(axis=0)
vars = np.sqrt(inputs.var(axis=0))

del inputs

# Init a stepper for electric field calculation and later conformation
w = 1.


def V(x: np.ndarray, t: float):
    '''if 1 < t < 4:
        return 100 - x
    else:
        return 0'''
    # return 0 * x
    # oscillator:
    return (w ** 2) / 2 * np.square(x)


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

energys, eigenstates = stepper.get_stationary_solutions(k=250)

# times at which electric field needs to be calculated
t = np.linspace(0, 6, 61)

# Init neural network
N = 5
# occupation state to maximize
n_to_max = 1
number_layers = 3
number_nodes = 20
architecture = (t.shape[0], *[number_nodes] * number_layers, N)
net = Net(architecture).double()
net.regression = False
net.load_state_dict(torch.load(__PATH__))

# Init genetic algorithm
# Fitness class for maximizing an objective
creator.create('FitnessMax', base.Fitness, weights=(1.0,))
# Individual class containing electric field arguments
creator.create('Individual', list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()


# method for creating and Individual
def attributes():
    ind = creator.Individual()
    # main amplitude range = 0 to 10
    ind.append(10 * random.random())
    # time width range = 0 to 2
    ind.append(2 * random.random())
    # frequency range = 0 to 5
    ind.append(5 * random.random())
    # sub amplitude range = 0 to 5
    ind.append(5 * random.random())
    # frequency range = 0 to 5
    ind.append(5 * random.random())
    return ind


toolbox.register('attributes', attributes)
# create an individual
toolbox.register('individual', toolbox.attributes)
# create a population of individuals
toolbox.register('population', tools.initRepeat, list, toolbox.individual)


# calculate electric field from arguments
def get_input(args):
    # init electric field
    e_field = np.zeros_like(t)
    # set electric field according to current arguments
    stepper.set_electric_field(args[0], t[-1] / 2, args[1], [(1, args[2]), (args[3], args[4])])
    # calculate electric field at every time t
    for i in range(len(t)):
        e_field[i] = (stepper.E(stepper.x, t[i])[0])
    # correctly feature scale inputs
    input = np.nan_to_num((e_field - means) / vars)
    # convert it to a tensor for future processing
    return torch.tensor(input)


# fitness function
def fitness(ind):
    input = get_input(ind)
    return net(input)[n_to_max].item(),


toolbox.register('evaluate', fitness)
# mating function
toolbox.register('mate', tools.cxTwoPoint)


# mutation function
def mutate(ind, indpb=0.05):
    for i in range(len(ind)):
        if i == 0 and random.random() < indpb:
            shift = 0.1 * (2 * random.random() - 1)
            ind[i] = max(0, min(50, ind[i] + shift))
        elif i == 1 and random.random() < indpb:
            shift = 0.1 * (2 * random.random() - 1)
            ind[i] = max(0, min(5, ind[i] + shift))
        else:
            shift = 0.1 * (2 * random.random() - 1)
            ind[i] = max(0, min(5, ind[i] + shift))
    return ind,


toolbox.register('mutate', mutate)
toolbox.register('select', tools.selTournament, tournsize=3)

pop = toolbox.population(n=300)
hof = tools.HallOfFame(1)
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, halloffame=hof, verbose=False)

best = hof[0]
print(best)
bins = [i for i in range(N + 1)]
plt.subplot(211)
plt.hist(bins[:-1], bins, weights=net(get_input(best)).detach(), align='left')
plt.title('Predicted Occupation by Neural Network')
# plt.show()

stepper.set_time(0)
stepper.set_state(eigenstates[0], normalzie=False)
stepper.set_electric_field(best[0], t_max / 2, best[1], [(1, best[2]), (best[3], best[4])])
stepper.step_to(t_max, dt)
projections = stepper.projection(eigenstates)
occupations = np.real(np.conj(projections) * projections)[:N]
plt.subplot(212)
plt.hist(bins[:-1], bins, weights=occupations, align='left')
plt.title('Predicted Occupation by simulating with Crank-Nicolson')
plt.gcf().tight_layout()
plt.savefig('harmonic_oscillator/genetic/optim.pdf')
plt.show()
