import numpy as np
from crank_nicolson import Stepper
import os
from No_Interrupt import NoInterrupt

__indicies_path__ = 'current_indices'
__save_path__ = 'trainigs_data'
__targets_path__ = 'targets'


# init indices for passing through laser paramters
def init():
    indices = np.array([0, 0, 0, 0, 0])
    np.save(__indicies_path__, indices)
    os.remove(__targets_path__ + '.npy')
    exit()


def load():
    return np.load(__indicies_path__ + '.npy')


if __name__ == '__main__':
    # reset
    # init()
    # load current indices
    indices = load()
    data = np.load('parameters.npz')
    amplitudes = data['A']
    time_widths = data['s']
    # amp_freq1 = data['a1']
    freq1 = data['f1']
    amp_freq2 = data['a2']
    freq2 = data['f2']

    max_indices = [len(amplitudes), len(time_widths), len(freq1), len(amp_freq2), len(freq2)]

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
    targets = []
    goal = [i - 1 for i in max_indices]

    try:
        while indices[0] < max_indices[0]:
            print(f'Current indicies = {indices}\n End = {goal}')
            stepper.set_time(0)
            stepper.set_state(eigenstates[0], normalzie=False)
            stepper.set_electric_field(amplitudes[indices[0]], t_max / 2, time_widths[indices[1]],
                                       [(1, freq1[indices[2]]),
                                        (amp_freq2[indices[3]], freq2[indices[4]])])
            stepper.step_to(t_max, dt)
            projections = stepper.projection(eigenstates)
            target = np.real(np.conj(projections) * projections)

            # go over every permutation
            for i in range(len(indices) - 1, -1, -1):
                indices[i] += 1
                if indices[i] >= max_indices[i] and i > 0:
                    indices[i] = 0
                else:
                    break
            targets.append(target)

    except KeyboardInterrupt:
        np.save(__indicies_path__, indices)
        saved_targets = None
        try:
            saved_targets = np.load(__targets_path__ + '.npy')
            saved_targets = np.append(saved_targets, targets, axis=0)
        except IOError:
            saved_targets = np.array(targets)
        np.save(__targets_path__, saved_targets)
        exit()

    np.save(__indicies_path__, indices)
    saved_targets = None
    try:
        saved_targets = np.load(__targets_path__ + '.npy')
        saved_targets = np.append(saved_targets, targets, axis=0)
    except IOError:
        saved_targets = np.array(targets)
    np.save(__targets_path__, saved_targets)
