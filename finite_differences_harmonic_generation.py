import numpy as np
from crank_nicolson import Stepper
from fourier_transform import fourier_transform
import matplotlib.pyplot as plt


# Initialization of time propagator

def V(x, t):
    return 0.5 * np.square(x)


parameters_max = np.array([1., .5, 10., 1., 10.])
# parameters_max = np.array([1., 5., 1., 5.])

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
energys, eigenstates = stepper.get_stationary_solutions(k=250)

target_freq = 4.


def spectrum(parameters):
    time_shift = t_max / 2

    stepper.set_time(0)
    stepper.set_state(eigenstates[0], normalzie=False)

    frequencies = [(1, parameters[2]), (parameters[3], parameters[4])]
    stepper.set_electric_field(parameters[0], time_shift, parameters[1], frequencies)

    times = [stepper.t]
    mean_x = [stepper.mean_x().real]
    while stepper.t < t_max:
        stepper.step(dt)
        times.append(stepper.t)
        mean_x.append(stepper.mean_x().real)

    return fourier_transform(times, mean_x)


def get_intensity(parameters):
    freqs, spec = spectrum(parameters)
    index = np.argmin(np.abs(freqs - target_freq))
    return np.abs(spec[index])


def grad(parameters, dp=1e-3):
    occ0 = get_intensity(parameters)
    occs = np.zeros_like(parameters)
    for i in range(len(parameters)):
        parameters[i] += dp
        occs[i] = get_intensity(parameters)
        parameters[i] -= dp
    return (occs - occ0) / dp, occ0


parameters = np.random.random(len(parameters_max)) * parameters_max

values = []
learning_rate = .1
for i in range(100):
    g, v = grad(parameters)
    parameters += learning_rate * g
    values.append(v)
    print(i)

plt.plot(values)
plt.show()

freqs, spec = spectrum(parameters)
print(parameters)
plt.plot(freqs, np.abs(spec))
max_int = np.max(np.abs(spec))
for i in range(0, 10, 2):
    plt.plot([i+1, i+1], [0, max_int], 'k--', alpha=.5)
plt.show()
