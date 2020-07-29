from crank_nicolson import Stepper
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib


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

def occupation2(parameters):
    frequencies = [(1, parameters[1]), (parameters[2], parameters[3])]
    stepper.set_electric_field2(parameters[0], frequencies, order=2)
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

def grad2(parameters, dp=1e-3):
    occ0 = occupation2(parameters)
    occs = np.zeros_like(parameters)
    for i in range(len(parameters)):
        parameters[i] += dp
        occs[i] = occupation2(parameters)
        parameters[i] -= dp
    return (occs - occ0) / dp, occ0


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

# print(occupation(parameters_max))
# exit()

if __name__ == '__main__':
    for j in range(10):
        o = 0
        while o < 1e-3:
            parameters = np.random.rand(len(parameters_max)) * parameters_max
            o = occupation(parameters)
        occupations = []
        learning_rate = .5
        momentum = 0.9
        delta = np.zeros_like(parameters)
        try:
            for i in range(1000):
                if i > 1:
                    d = np.abs(occupations[-1][0] - occupations[-2][0])
                    print(f'j = {j}, i = {i}')
                    print('d = ', d)
                if i > 1 and d < 1e-6 and o > 0.1:
                    break
                g, o = grad(parameters)
                print(o)
                print(parameters)
                occupations.append([o, parameters.copy()])
                if i > 0:
                    delta = occupations[-1][1] - occupations[-2][1]
                if o > 0.2:
                    learning_rate = 1e-2
                # else:
                #     learning_rate = 1.
                parameters += learning_rate * g + momentum * delta
        except KeyboardInterrupt:
            pass
        print(parameters)
        plt.figure()
        plt.plot([o[0] for o in occupations])
        plt.xlabel('Iterations')
        plt.ylabel('Occupation')
        tikzplotlib.save(f'harmonic_oscillator/finite-differences/field2/run{j+1}.tex')
        plt.savefig(f'harmonic_oscillator/finite-differences/field2/run{j+1}.pdf')
        plt.show()

    plt.plot([o[0] for o in occupations])
    plt.show()
    plt.figure()
    plt.subplot(3, 2, 1)
    plt.plot([o[0] for o in occupations])
    plt.xlabel('Iterations')
    plt.ylabel('Occupation')
    plt.subplot(3, 2, 2)
    plt.plot([o[1][0] for o in occupations])
    plt.xlabel('Iterations')
    plt.ylabel('Amplitude')
    plt.subplot(3, 2, 3)
    plt.plot([o[1][1] for o in occupations])
    plt.xlabel('Iterations')
    plt.ylabel('Time Width')
    plt.subplot(3, 2, 4)
    plt.plot([o[1][2] for o in occupations])
    plt.xlabel('Iterations')
    plt.ylabel('Frequency 1')
    plt.subplot(3, 2, 5)
    plt.plot([o[1][3] for o in occupations])
    plt.xlabel('Iterations')
    plt.ylabel('Amplitude 2')
    plt.subplot(3, 2, 6)
    plt.plot([o[1][4] for o in occupations])
    plt.xlabel('Iterations')
    plt.ylabel('Frequency 2')

    plt.gcf().tight_layout()
    plt.savefig('finite_differences_parameters.pdf')
    plt.show()
