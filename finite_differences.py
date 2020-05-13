from crank_nicolson import Stepper
import numpy as np
import matplotlib.pyplot as plt


def V(x, t):
    return 0.5 * np.square(x)


def occupation(parameters):
    time_shift = t_max / 2

    frequencies = [(1, parameters[2]), (parameters[3], parameters[4])]
    stepper.set_electric_field(parameters[0], time_shift, parameters[1], frequencies)
    stepper.set_time(0)
    stepper.set_state(eigenstates[0], normalzie=False)
    stepper.step_to(t_max, dt)
    occ = stepper.projection([eigenstates[1]])[0]
    return np.real(np.conj(occ) * occ)

def grad(parameters, dp=1e-3):
    occ0 = occupation(parameters)
    occs = np.zeros_like(parameters)
    for i in range(len(parameters)):
        parameters[i] += dp
        occs[i] = occupation(parameters)
        parameters[i] -= dp
    return (occs - occ0) / dp, occ0


parameters = np.array([10., 1., 5., 1., 5.])

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

occupations = []
learning_rate = 1.
try:
    for i in range(1000):
        g, o = grad(parameters)
        print(o)
        occupations.append(o)
        if o > 0.2:
            learning_rate = 0.1
        parameters += learning_rate * g
except KeyboardInterrupt:
    pass

plt.plot(occupations)
plt.xlabel('Iterations')
plt.ylabel('Occupation')
plt.savefig('finite_differences.pdf')
plt.show()
