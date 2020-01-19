import scipy.integrate
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# solving ode in time
def step_ode(current_state, V, x, t, dt, boundary='fixed'):
    # solve dy/dt = f(t, y)
    dx2_inv = 1 / (x[1] - x[0]) ** 2

    def right_hand_side(t, state):
        result = []
        for i in range(len(state)):
            sum = 0
            # 3 point rule
            '''if i > 0:
                sum += 1j * dx2_inv / 2 * state[i - 1]
            sum -= 1j * (dx2_inv + V(x[i], t)) * state[i]
            if i < len(state) - 1:
                sum += 1j * dx2_inv / 2 * state[i + 1]'''
            if boundary == 'fixed':
                if i > 1:
                    sum -= 1j * dx2_inv / 24 * state[i - 2]
                if i > 0:
                    sum += 1j * dx2_inv * 16 / 24 * state[i - 1]
                sum -= 1j * (dx2_inv * 30 / 24 + V(x[i], t)) * state[i]
                if i < len(state) - 1:
                    sum += 1j * dx2_inv * 16 / 24 * state[i + 1]
                if i < len(state) - 2:
                    sum -= 1j * dx2_inv / 24 * state[i + 2]
                result.append(sum)
            elif boundary == 'periodic':
                sum -= 1j * dx2_inv / 24 * state[i - 2]
                sum += 1j * dx2_inv * 16 / 24 * state[i - 1]
                sum -= 1j * (dx2_inv * 30 / 24 + V(x[i], t)) * state[i]
                sum += 1j * dx2_inv * 16 / 24 * state[(i + 1) % len(state)]
                sum -= 1j * dx2_inv / 24 * state[(i + 2) % len(state)]
                result.append(sum)
            else:
                raise ValueError(f'Unknown boundary condition: {boundary}')
        return result

    result = scipy.integrate.solve_ivp(right_hand_side, (t, t + dt), current_state)
    return result.y[:, -1]


def step_ode_full(current_state, V, x, t, dt, boundary='fixed'):
    # solve dy/dt = f(t, y)
    dx2_inv = 1 / (x[1] - x[0]) ** 2

    def right_hand_side(t, state):
        result = []
        for i in range(len(state)):
            sum = 0
            # 3 point rule
            '''if i > 0:
                sum += 1j * dx2_inv / 2 * state[i - 1]
            sum -= 1j * (dx2_inv + V(x[i], t)) * state[i]
            if i < len(state) - 1:
                sum += 1j * dx2_inv / 2 * state[i + 1]'''
            if boundary == 'fixed':
                if i > 1:
                    sum -= 1j * dx2_inv / 24 * state[i - 2]
                if i > 0:
                    sum += 1j * dx2_inv * 16 / 24 * state[i - 1]
                sum -= 1j * (dx2_inv * 30 / 24 + V(x[i], t)) * state[i]
                if i < len(state) - 1:
                    sum += 1j * dx2_inv * 16 / 24 * state[i + 1]
                if i < len(state) - 2:
                    sum -= 1j * dx2_inv / 24 * state[i + 2]
                result.append(sum)
            elif boundary == 'periodic':
                sum -= 1j * dx2_inv / 24 * state[i - 2]
                sum += 1j * dx2_inv * 16 / 24 * state[i - 1]
                sum -= 1j * (dx2_inv * 30 / 24 + V(x[i], t)) * state[i]
                sum += 1j * dx2_inv * 16 / 24 * state[(i + 1) % len(state)]
                sum -= 1j * dx2_inv / 24 * state[(i + 2) % len(state)]
                result.append(sum)
            else:
                raise ValueError(f'Unknown boundary condition: {boundary}')
        return result

    result = scipy.integrate.solve_ivp(right_hand_side, (t, t + dt), current_state)
    return result


def crank_nicolson(current_state: np.ndarray, potential: callable, x: list, t: float, dt: float):
    # calculate distance square
    dx2 = (x[1] - x[0]) ** 2
    # calculate distance / time ratio
    ratio = 4j * dx2 / dt
    # create an array for the right hand side of the equation
    b = np.zeros_like(current_state, dtype='complex128')
    # create lists for the diagonals of the matrix
    main_diag = np.zeros_like(current_state, dtype='complex128')
    upper_diag = np.ones((len(current_state) - 1,))
    lower_diag = np.ones((len(current_state) - 1,))
    # fill main diagonal and right hand side with values
    for i in range(len(current_state)):
        # evalute the right hand side of the equation
        # b[i] = psi(i-1) - (2 + 2 * dx2 * V(i, t) + i * 4 * dx2/dt) + psi(i+1) (right hand side)
        # (A*x)[i] = -1 * psi'(i-1) + (2 + 2 * dx2 * V(i, t+dt/2) - i * 4 * dx2 / dt) * psi'(i) - psi'(i+1)

        # if i - 1 < 0 assume zeroes in the wave function (fixed boundary condition)
        if i > 0:
            b[i] -= current_state[i - 1]
        b[i] += 2 + 2 * dx2 * potential(x[i], t + dt / 2) + ratio
        main_diag[i] = -2 - 2 * dx2 * potential(x[i], t + dt / 2) + ratio
        if i < len(current_state) - 1:
            b[i] -= current_state[i + 1]
    A = scipy.sparse.diags([lower_diag, main_diag, upper_diag], [-1, 0, 1])
    solution = scipy.sparse.linalg.spsolve(A, b)
    # Testing correctness
    print(np.linalg.norm(solution - np.linalg.solve(A.toarray(), b)))
    # print(solution[1])
    return solution


def crank_nicolson2(current_state, V, x, t, dt):
    b = np.zeros_like(current_state, dtype='complex128')
    A = np.zeros((len(current_state), len(current_state)), dtype='complex128')
    dx2 = (x[1] - x[0]) ** 2
    complex_ratio = 4j * dx2 / dt
    for i in range(len(current_state)):
        if i > 0:
            b[i] -= current_state[i - 1]
            A[i][i - 1] = 1.
        b[i] += complex_ratio + 2 + dx2 * V(x[i], t + dt / 2)
        A[i][i] = complex_ratio + complex_ratio - 2 - 2 * dx2 * V(x[i], t + dt / 2)
        if i < len(current_state) - 1:
            b[i] -= current_state[i + 1]
            A[i][i + 1] = 1.
    return np.linalg.solve(A, b)


def hamiltonian(V, x, t):
    dx2 = (x[1] - x[0]) ** 2
    V = scipy.sparse.diags(V(x, t))
    upper_diag = np.ones((len(x) - 1,))
    lower_diag = np.ones((len(x) - 1,))
    main_diag = -2 * np.ones((len(x),))
    derivative = scipy.sparse.diags([lower_diag, main_diag, upper_diag], [-1, 0, 1]) / dx2
    return -0.5 * derivative + V



def crank_nicolson3(current_state, V, x, t, dt):
    '''dx2 = (x[1] - x[0]) ** 2
    I = 2 * scipy.sparse.identity(len(x))
    V = scipy.sparse.diags(1j * dt * V(x, t))
    upper_diag = np.ones((len(x) - 1,))
    lower_diag = np.ones((len(x) - 1,))
    main_diag = -2 * np.ones((len(x),))
    A = 1j / 2 * dt / dx2 * scipy.sparse.diags([lower_diag, main_diag, upper_diag], [-1, 0, 1])
    return scipy.sparse.linalg.spsolve(I - A + V, (I + A - V).dot(current_state))'''
    I = scipy.sparse.identity(len(x))
    return scipy.sparse.linalg.spsolve(I + 0.5j * dt * hamiltonian(V, x, t), (I - 0.5j * dt * hamiltonian(V, x, t)).dot(current_state))



if __name__ == '__main__':
    def V(x, t):
        '''if 1 < t < 4:
            return 100 - x
        else:
            return 0'''
        # return 0 * x
        # oscillator:
        # return np.square(x)
        return 5 * np.where(x < 2.5, 1, 0) * np.where(x > -2.5, 1, 0)
        # if 30 <= t <= 40:
        #    20 * np.exp(-np.square(x) / (2 * 10 ** 2)) * (4 * np.cos(x) + 2 * np.cos(2 * x) + np.cos(3 * x))
        return total


    # inital state
    def psi(x):
        # stationary function for potential box
        # a = x[-1] - x[0]
        # n = 10
        # return np.sqrt(2 / a)  * np.sin(np.pi * n / a * (x - x[0]))
        # gauß function
        x0 = -15
        p0 = 5
        a = 2.5
        a2 = a ** 2
        return 1 / (np.pi * a2) ** (1 / 4) * np.exp(1j * p0 * (x - x0)) * np.exp(-(x - x0) ** 2 / (2 * a2))


    def mean_place(current_state, x):
        if len(current_state) != len(x):
            raise ValueError('Samples size of state, must correspond to size of x')
        to_integrate = np.conj(current_state) * x * current_state
        return scipy.integrate.simps(to_integrate, x).real


    # number of sample points
    N = 500
    # x domain
    x_min = -100
    x_max = 100
    # starting point
    t = 0
    # number of frames for saving animation
    frames = 0
    # create list of all x points
    x = np.linspace(x_min, x_max, N, endpoint=True)
    # time step
    dt = 0.1
    # calculate initial state
    current_state = psi(x)
    # ode_solution = step_ode_full(current_state, V, x, t, dt, 'fixed')
    # frames = len(ode_solution.t)
    # counter = 0
    # counter_max = -1
    # plot initial state
    line1, line2 = plt.plot(x, np.real(current_state), x, np.abs(current_state))
    # set axis limits
    plt.ylim((-1, 1))
    plt.xlim((x_min, x_max))
    # set text for time and probability
    time_text = plt.gca().text(0.02, 0.95, '', transform=plt.gca().transAxes)
    probability_text = plt.gca().text(0.02, 0.9, '', transform=plt.gca().transAxes)
    energy_text = plt.gca().text(0.02, 0.85, '', transform=plt.gca().transAxes)
    print(scipy.integrate.cumtrapz(np.square(np.abs(current_state)), x)[-1])

    # Test of crank nicolson
    # plt.show()
    # new_state /= np.sqrt(scipy.integrate.simps(new_state, dx=x[1] - x[0]))
    '''for i in range(200):
        t += dt
        current_state = crank_nicolson3(current_state, V, x, t, dt)
        plt.plot(x, np.real(current_state), x, np.abs(current_state))
        plt.ylim((-1, 1))
        plt.xlim((x_min, x_max))
        time_text = plt.gca().text(0.02, 0.95, f't = {t}', transform=plt.gca().transAxes)
        plt.show()'''


    # init function for animation
    def init():
        line1.set_data(x, np.real(current_state))
        line2.set_data(x, np.abs(current_state))
        return line1, line2, time_text, probability_text, energy_text


    # animation function for using runge kutta methode
    def animate_runge_kutta(i):
        global ode_solution
        # display new time
        time_text.set_text(f't = {ode_solution.t[i]}')
        # set new state
        current_state = ode_solution.y[:, i]
        # calculate absolute values of the current state
        state_abs = np.abs(current_state)
        # plot the absolute value und the real part of the current wave function
        line1.set_data(x, np.real(current_state))
        line2.set_data(x, state_abs)
        # print mean place
        probability_text.set_text(f'<x> = {mean_place(current_state, x)}')
        # print animation progress
        print(f'{i} / {frames - 1}')
        return line1, line2, time_text, probability_text


    def animate_crank_nicolson(i):
        global current_state, t, x
        # calculate new state with old state using crank nicolson method
        current_state = crank_nicolson3(current_state, V, x, t, dt)
        # update time
        t += dt
        # calculate absolute value of wave function and plot it and the real part
        state_abs = np.abs(current_state)
        line1.set_data(x, np.real(current_state))
        line2.set_data(x, state_abs)
        # display current time and norm (squared) of wave function
        time_text.set_text(f't = {t}')
        probability_text.set_text(f'<psi|psi> = {scipy.integrate.simps(np.square(state_abs), x)}')
        energy_text.set_text(f'E = {scipy.integrate.simps(np.conj(current_state) * hamiltonian(V, x, t).dot(current_state), x).real}')
        return line1, line2, time_text, probability_text,energy_text


    # animate
    anim = animation.FuncAnimation(plt.gcf(), animate_crank_nicolson, init_func=init, interval=100, blit=True)
    name = 'animation'
    # save animation
    # anim.save(f'{name}.gif', writer='imagemagick', fps=10)
    # display animation
    plt.show()
