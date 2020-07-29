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


def simulate(t, state, dt, space, t_h, parameters, goal, N_step, dx):
    # for i in range(N_step):
    #     t, state = step(t, state, dt, space, t_h, parameters)
    body_fun = lambda i, val: step(*val, dt, space, t_h, parameters)
    t, state = fori_loop(0, N_step, body_fun, (t, state))
    occ = occupation(state, goal, dx)
    occ = (occ * np.conj(occ)).real
    print(occ)
    return occ


# grad_simulation = value_and_grad(simulate, argnums=5)
grad_simulation = jacfwd(simulate, argnums=5)
grad_simulation = jit(grad_simulation)
simulation = jit(simulate)

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
stepper.set_state(eigenstates[0])
print(occupation(eigenstates[0], eigenstates[1], dx))

state = np.array(eigenstates[0], dtype='complex128')
goal = eigenstates[1]
N_step = int(2 * t_h / dt)

# plt.plot(onp.array(space), eigenstates[0])
# plt.show()
# simulate(t, state, dt, space, t_h, parameters, goal, N_step, dx)
learning_rate = 1
momentum = 0.9
delta = np.zeros_like(parameters)
occs = []
try:
    for i in range(1000):
        if i == 250:
            learning_rate = 0.1
        grad = grad_simulation(t, state, dt, space, t_h, parameters, goal, N_step, dx)
        occ = simulation(t, state, dt, space, t_h, parameters, goal, N_step, dx)
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
