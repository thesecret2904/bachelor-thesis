from jax.config import config

# enable float64
config.update("jax_enable_x64", True)

import numpy as onp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from crank_nicolson import Stepper
import jax.numpy as np
from jax import grad, jit, value_and_grad, jacfwd
from jax.lax import fori_loop
from jax.ops import index, index_update


def harmonic_pot_jax(space: np.ndarray, t: float):
    return 1 / 2 * np.square(space)


def harmonic_pot_np(space: onp.ndarray, t: float):
    return 1 / 2 * onp.square(space)


# function for electric field (checked)
def electric_field(t, parameters, t_h, space):
    t_eff = t - t_h
    a = parameters[0] / (np.sqrt(2 * np.pi) * parameters[1])
    e = a * (np.sin(parameters[2] * t_eff) + parameters[3] * np.sin(parameters[4] * t_eff)) * np.exp(
        -t_eff ** 2 / (2 * parameters[1] ** 2))
    return e


def init_hamiltonian(space, t, parameters, t_h):
    dx2 = (space[0] - space[1]) ** 2
    upper_diag = np.ones(len(space) - 1, ) / dx2 * (-0.5)
    lower_diag = np.ones(len(space) - 1, ) / dx2 * (-0.5)
    main_diag = -0.5 * np.full(len(space), -2.) / dx2 + harmonic_pot_jax(space, t) - electric_field(t, parameters, t_h,
                                                                                                    space) * space
    H = np.diag(main_diag) + np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
    return H


def mean_x(state, x, dx):
    # calculate mean x value with trapezoidal rule
    mean = (np.sum(np.conj(state) * x * state) - 0.5 * (
                np.conj(state[0]) * x[0] * state[0] + np.conj(state[-1]) * x[-1] * state[-1])).real * dx
    return mean


def occupation(state, goal, dx):
    occ = (np.sum(np.conj(goal) * state) - 0.5 * (
            np.conj(goal[0]) * state[0] + np.conj(goal[len(goal) - 1]) * state[len(goal) - 1])) * dx
    return occ


def step(t, state, dt, space, t_h, parameters):
    hamiltonian = init_hamiltonian(space, t + dt / 2, parameters, t_h)
    I = np.identity(len(state))
    H = 0.5j * hamiltonian * dt
    # get the new state by solving a system of linear equations obtained by crank-nicolson
    state = np.linalg.solve(I + H, (I - H).dot(state))
    t += dt
    return t, state


def simulate(t, state, dt, space, t_h, parameters, N_step, dx, linear_weight, target_index):
    # for i in range(N_step):
    #     t, state = step(t, state, dt, space, t_h, parameters)
    # body_fun = lambda i, val: step(*val, dt, space, t_h, parameters)
    def body_fun(i, val):
        t, state = step(*val[:2], dt, space, t_h, parameters)
        temp_moments = np.zeros(N_step)
        temp_moments = index_update(temp_moments, index[i], mean_x(state, space, dx))
        temp_moments = val[2] + temp_moments
        return t, state, temp_moments
    t, state, dipole_moments = fori_loop(0, N_step, body_fun, (t, state, np.zeros(N_step, dtype='float64')))

    transform = np.fft.fftshift(np.fft.fft(dipole_moments)) * dt
    return np.abs(transform[target_index] + linear_weight * (transform[target_index + 1] - transform[target_index]))


# grad_simulation = value_and_grad(simulate, argnums=5)
grad_simulation = jacfwd(simulate, argnums=5)
grad_simulation = jit(grad_simulation, static_argnums=6)
simulation = jit(simulate, static_argnums=6)

N = 1000
x_min = -50
x_max = 50
space = np.linspace(x_min, x_max, N, endpoint=True, dtype=np.float64)
dx = 0.1
dx2 = dx ** 2

# parameters = np.array([2.21, 1.59, 1.60, 3.14, 2.15], dtype=np.float64)
parameters = np.array([10., 1., 5., 1., 5.], dtype=np.float64)
t = 0.
t_h = 3.0
dt = 0.1

stepper = Stepper(None, 0, onp.array(space), harmonic_pot_np)
energys, eigenstates = stepper.get_stationary_solutions(k=250)


state = np.array(eigenstates[0], dtype='complex128')

target_freq = 1.

N_step = int(2 * t_h / dt)

freqs = np.fft.fftshift(np.fft.fftfreq(N_step, dt / (2 * np.pi)))
target_index = 0
for i in range(0, N_step):
    if target_freq < freqs[i]:
        target_index = i - 1
        break
linear_weight = (target_freq - freqs[target_index]) / (freqs[target_index + 1] - freqs[target_index])
# print(freqs[target_index] + linear_weight * (freqs[target_index + 1] - freqs[target_index]))

dipole_moments = np.zeros(N_step, dtype='float64')

# parameters = np.array([2.21, 1.50, 1.60, 3.14, 2.15], dtype=np.float64)
# value = simulation(t, state, dt, space, t_h, parameters, N_step, dx, linear_weight, target_index)
# print(value)
# exit()

# plt.plot(onp.array(space), eigenstates[0])
# plt.show()
# simulate(t, state, dt, space, t_h, parameters, goal, N_step, dx)
learning_rate = .1
momentum = 0.9
delta = np.zeros_like(parameters)
occs = []
try:
    for i in range(1000):
        grad = grad_simulation(t, state, dt, space, t_h, parameters, N_step, dx, linear_weight, target_index)
        occ = simulation(t, state, dt, space, t_h, parameters, N_step, dx, linear_weight, target_index)
        print(occ)
        occs.append(occ)
        old = parameters.copy()
        parameters = parameters + learning_rate * grad + momentum * delta
        delta = parameters - old
except KeyboardInterrupt:
    pass
plt.plot(occs)
plt.show()
# fig = plt.figure()
# ax = plt.axes()
# plt.ylim((-1, 1))
# plt.xlim(x_min, x_max)
# line, = plt.plot(space, eigenstates[0])
# time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
#
#
# def animate(i):
#     global t, state
#     if t < 2 * t_h:
#         t, state = step(t, state, dt, space, t_h, parameters)
#     time_text.set_text(str(t))
#     line.set_data(space, np.real(state))
#     return line, time_text
#
#
# anim = animation.FuncAnimation(fig, animate, blit=True, interval=100)
# plt.show()
# print(np.abs(occupation(goal, state, dx)) ** 2)
