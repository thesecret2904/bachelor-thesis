from crank_nicolson import Stepper
from permutator import permutations
import numpy as np
import matplotlib.pyplot as plt

def stability(parameters, func, range=0.1):
    values = []
    for shift in permutations(len(parameters), [-1, 1]):
        norm = np.linalg.norm(shift)
        if not np.isclose(norm, 0):
            shift = shift * range / np.linalg.norm(shift)
        values.append(func(parameters + shift))
    return np.mean(values), np.std(values)


if __name__ == '__main__':
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

    # def grad(parameters, dp=1e-3):
    #     occ0 = occupation(parameters)
    #     occs = np.zeros_like(parameters)
    #     for i in range(len(parameters)):
    #         parameters[i] += dp
    #         occs[i] = occupation(parameters)
    #         parameters[i] -= dp
    #     return (occs - occ0) / dp, occ0


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

    print(occupation([2.51, 1.50, 1.60, 3.14, 2.15]))
    print(stability([2.51, 1.50, 1.60, 3.14, 2.15], occupation))