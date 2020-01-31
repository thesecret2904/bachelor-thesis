import scipy.integrate
import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Class for time propagation
class Stepper:
    def __init__(self, initial_state: np.ndarray, start_time: float, x_domain: np.ndarray, potential: callable):
        '''
        initialize time propagator
        '''
        # set current state
        self.state = initial_state
        # set current time
        self.t = start_time
        # set space domain
        self.x = x_domain
        # set a function for the potential energy
        self.V = potential
        # initialize variable for electric field
        self.E = None
        # initialize derivative matrix with fixed boundary conditions
        upper_diag = np.ones((len(self.x) - 1,))
        lower_diag = np.ones((len(self.x) - 1,))
        main_diag = np.full((len(self.x),), -2.)
        self.dx2 = (self.x[1] - self.x[0]) ** 2
        self.derivative = scipy.sparse.diags([lower_diag, main_diag, upper_diag], [-1, 0, 1]) / self.dx2
        # set end time
        self.t_max = np.inf

    def hamiltonian(self, t):
        '''
        Get the hamilton operator (a matrix in discretized space) at time t
        :param t: time at which the hamilton operator should be evaluated
        :return: hamilton operator in matrix form
        '''
        # calculate potential energy at every point at time t
        V = scipy.sparse.diags(self.V(self.x, t))
        # add electric field in dipole approximation if it is defined
        if self.E is not None:
            E = scipy.sparse.diags(self.E(self.x, t) * self.x)
            V = V - E
        return -0.5 * self.derivative + V

    def step(self, dt):
        '''
        propagate time by dt
        :param dt: time step for time propagation
        :return: a tuple consisting of the new time and the new state
        '''
        # only propagate if current time < maximum time
        if self.t < self.t_max:
            # get identity matrix I and hamilton at half of the time step operator (mid-point rule for integration)
            I = scipy.sparse.identity(len(self.state))
            H = 0.5j * self.hamiltonian(self.t + dt / 2) * dt
            # get the new state by solving a system of linear equations obtained by crank-nicolson
            self.state = scipy.sparse.linalg.spsolve(I + H, (I - H).dot(self.state))
            self.t += dt
        return self.t, self.state

    def step_to(self, target_time: float, dt: float):
        '''
        Propagates time until target time is reached.
        :param target_time: target time
        :param dt: time step
        :return: a tuple consisting of the new time and the new state
        '''
        while self.t < target_time and self.t < self.t_max:
            self.step(dt)
        return self.t, self.state

    def mean(self, operator, state=None):
        '''
        Calculates the mean value of an operator of either the current state or a given state
        :param operator: Operator in matrix form of which the mean shall be calculated
        :param state: optional state
        :return: mean
        '''
        if state is None:
            return scipy.integrate.simps(np.conj(self.state) * operator.dot(self.state), self.x)
        else:
            return scipy.integrate.simps(np.conj(state) * operator.dot(state), self.x)

    def mean_x(self, state=None):
        return self.mean(scipy.sparse.diags(self.x), state)

    def mean_energy(self, state=None):
        return self.mean(self.hamiltonian(self.t), state)

    def norm2(self, state=None):
        return self.mean(scipy.sparse.identity(len(self.x)), state)

    def norm(self, state=None):
        return np.sqrt(self.norm2(state))

    def get_real(self):
        return np.real(self.state)

    def get_imag(self):
        return np.imag(self.state)

    def get_abs(self):
        return np.abs(self.state)

    def get_time(self):
        return self.t

    def get_stationary_solutions(self, t: float = 0, k: int = None):
        eigenvalues, eigenstates = scipy.sparse.linalg.eigsh(self.hamiltonian(t), len(self.x) // 2 if k is None else k,
                                                             which='SM')
        eigenstates = [eigenstates[:, i] for i in range(len(eigenstates[0]))]
        self.normalize(eigenstates)
        return eigenvalues, eigenstates

    def set_state(self, state: np.ndarray, normalzie=True):
        self.state = state
        if normalzie:
            self.normalize()

    def set_time(self, t: float):
        self.t = t

    def normalize(self, list=None):
        if list is None:
            self.state /= self.norm()
        else:
            for i in range(len(list)):
                list[i] /= self.norm(list[i])

    def projection(self, list):
        parameters = np.zeros(len(list), dtype='complex128')
        for i in range(len(list)):
            parameters[i] = scipy.integrate.simps(np.conj(list[i]) * self.state, self.x)
        return parameters

    def add_electric_field(self, E: callable):
        self.E = E

    def set_electric_field(self, amplitude: float, time_shift: float, time_width: float, frequencies: list):
        frequencies = frequencies.copy()
        amplitude /= np.sqrt(2 * np.pi) * time_width
        time_width = time_width ** 2

        def E(x: np.ndarray, t: float):
            return np.full_like(x, amplitude * np.exp(-(t - time_shift) ** 2 / (2 * time_width)) * sum(
                [a * np.sin(f * (t - time_shift)) for a, f in frequencies]))

        self.add_electric_field(E)


class Animator:
    def __init__(self, stepper: Stepper, dt: float = 0.05):
        self.dt = dt
        self.stepper = stepper
        # create figure and axis
        self.fig = plt.figure()
        self.ax = plt.axes()
        plt.ylim((-1, 1))
        plt.xlim((x_min, x_max))
        self.line1, self.line2 = plt.plot(x, stepper.get_real(), x, stepper.get_abs())
        # set text for time and probability
        self.time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)
        self.norm_text = self.ax.text(0.02, 0.9, '', transform=self.ax.transAxes)
        self.energy_text = self.ax.text(0.02, 0.85, '', transform=self.ax.transAxes)

    def __animate__(self, i):
        t, current = self.stepper.step(self.dt)
        self.line1.set_data(x, np.real(current))
        self.line2.set_data(x, np.abs(current))
        self.time_text.set_text(f't = {t}')
        self.norm_text.set_text(f'<psi|psi> = {stepper.norm2().real}')
        self.energy_text.set_text(f'E = {stepper.mean_energy().real}')
        return self.line1, self.line2, self.time_text, self.energy_text, self.norm_text

    def animate(self, path: str = None, frames: int = None, fps: int = 10):
        anim = animation.FuncAnimation(self.fig, self.__animate__, frames=frames, blit=True, interval=100)
        if path is None:
            plt.show()
        else:
            if frames is None:
                raise ValueError(f'frames must be not None if animation should be stored!')
            anim.save(path, writer='imagemagick', fps=fps)


if __name__ == '__main__':
    w = 1.


    def V(x: np.ndarray, t: float):
        '''if 1 < t < 4:
            return 100 - x
        else:
            return 0'''
        # return 0 * x
        # oscillator:
        return (w ** 2) / 2 * np.square(x)
        return 5 * np.where(x < 2.5, 1, 0) * np.where(x > -2.5, 1, 0)


    electric_field_time = (.5, 1.5)


    def E(x: np.ndarray, t: float):
        if electric_field_time[0] < t < electric_field_time[1]:
            return np.full_like(x, 2.5)
        else:
            return np.zeros_like(x)


    def fac(n):
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result


    def stationary_osci(x0, n):
        def hermit_norm(n, x0):
            return 1 / np.sqrt(np.sqrt(np.pi) * 2 ** n * fac(n) * x0)

        def hermit(n):
            if n == 0:
                def H(x):
                    return 1

                return H
            if n == 1:
                def H(x):
                    return 2 * x

                return H
            else:
                def H(x):
                    return 2 * x * hermit(n - 1)(x) - 2 * (n - 1) * hermit(n - 2)(x)

                return H

        def psi(x):
            a = hermit_norm(n, x0)
            return a * hermit(n)(x / x0) * np.exp(-w / 2 * np.square(x))

        return psi


    def coherent_state(x0, l, n=20):
        def psi(x):
            result = np.zeros_like(x)
            for i in range(n):
                result += l ** n / (fac(i) ** (1 / 2)) * stationary_osci(x0, i)(x)
            return result * np.exp(-l * np.conj(l) / 2)

        return psi


    # inital state
    def psi(x: np.ndarray):
        # stationary function for potential box
        # a = x[-1] - x[0]
        # n = 2
        # return np.sqrt(2 / a)  * np.sin(np.pi * n / a * (x - x[0]))
        # gauß function
        # x0 = 0
        # p0 = 0
        # a = 1
        # a2 = a ** 2
        # return 1 / (np.pi * a2) ** (1 / 4) * np.exp(1j * p0 * (x - x0)) * np.exp(-(x - x0) ** 2 / (2 * a2))
        # stationary function for oscillator
        n = -1
        x0 = 1 / np.sqrt(w)
        return coherent_state(x0, n, 30)(x)


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
    # add electric field to the Hamiltonian
    amp = 10
    time_shift = t_max / 2
    time_width = 1
    frequencies = [(1, 5), (1, 5)]
    stepper.set_electric_field(amp, time_shift, time_width, frequencies)
    # T = np.linspace(t, t_max, 500)
    # plt.plot(T, [stepper.E(x, t)[0] for t in T])
    # plt.show()

    # calculate eigensates and eigenvalues of the Hamiltonian
    energys, eigenstates = stepper.get_stationary_solutions(k=250)

    # get excited eigenstates
    # excited_energys, excited_states = stepper.get_stationary_solutions(
    #    (electric_field_time[0] + electric_field_time[1]) / 2, k=250)

    stepper.set_time(0)
    stepper.set_state(eigenstates[0], normalzie=False)
    anim = Animator(stepper, dt)
    anim.animate()
    # stepper.step_to(t_max, dt)
    projections = stepper.projection(eigenstates)
    projections = np.real(np.conj(projections) * projections)
    N = 5
    bins = [i for i in range(N + 1)]
    plt.hist(bins[:-1], bins, weights=projections[:N])
    plt.show()

    '''n = 0
    for n in range(100, 201, 10):
        # start with an eigenstate
        stepper.set_state(eigenstates[n], normalzie=False)
        # reset time to 0
        stepper.set_time(0)
    
        # Animator (uncomment for animation)
        # animator = Animator(stepper, dt)
        # animator.animate(None, frames=60)
    
        bins = [i for i in range(len(energys) + 1)]
    
        # energy spectrum (before electric field)
        projections = stepper.projection(eigenstates)
        plt.figure()
        plt.subplot(311)
        plt.hist(bins[:-1], bins, weights=np.real(np.conj(projections) * projections))
        plt.title(f'Initial Energy Spectrum (n = {n})')
    
        # Step forward till after electric field
        # stepper.step_to(electric_field_time[1] + 1, dt)
        # energy spectrum (after electric field)
        projections = stepper.projection(eigenstates)
        plt.subplot(312)
        plt.hist(bins[:-1], bins, weights=np.real(np.conj(projections) * projections))
        plt.title(f'Excited Energy Spectrum (Simulation, n = {n})')
    
        # theoretical energy spectrum (after electric field)
        probabilities = []
        for i in range(len(excited_states)):
            probabilities.append(scipy.integrate.simps(np.conj(excited_states[i]) * eigenstates[n], stepper.x))
        plt.subplot(313)
        plt.hist(bins[:-1], bins, weights=np.real(np.conj(probabilities) * probabilities))
        plt.title(f'Excited Energy Spectrum (Theory, n = {n})')
    
        plt.gcf().tight_layout()
        plt.savefig(f'harmonic_oscillator_energy_spectrums/energy_spectrums{n:02d}.pdf')
        plt.show()
        plt.close()'''
