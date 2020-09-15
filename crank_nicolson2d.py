import scipy.sparse
import scipy.sparse.linalg
import scipy.integrate
import numpy as np


def get_index(row, col, rows, cols):
    if row < 0 or row >= rows:
        raise ValueError('row out of range')
    if col < 0 or col >= cols:
        raise ValueError('col out of range')
    return row * cols + col


def get_row_col(index, cols):
    row = index // cols
    col = index - row * cols
    return row, col


def list_to_matrix(list, cols):
    rows = len(list) // cols
    matrix = np.zeros((rows, cols), dtype='complex128')
    for k in range(len(list)):
        i, j = get_row_col(k, cols)
        matrix[i][j] = list[k]
    return matrix


class Stepper2d:
    def __init__(self, initial_state: np.ndarray, start_time: float, x: np.ndarray, y: np.ndarray, potential: callable):
        self.state = initial_state
        self.t = start_time
        self.x = x
        self.y = y
        self.V = potential
        self.dx2 = (x[1] - x[0]) ** 2
        self.dy2 = (y[1] - y[0]) ** 2
        main_diag = np.full_like(initial_state, -2.)
        sub_diag = np.ones_like(initial_state)
        for i in range(len(y) - 1, len(sub_diag), len(y)):
            sub_diag[i] = 0
        derivative_y = scipy.sparse.diags([sub_diag, main_diag, sub_diag], [-1, 0, 1],
                                          (len(initial_state), len(initial_state))) / self.dx2
        sub_diag = np.ones_like(initial_state)
        derivative_x = scipy.sparse.diags([sub_diag, main_diag, sub_diag], [-len(y), 0, len(y)],
                                          (len(initial_state), len(initial_state))) / self.dy2
        self.derivative = derivative_x + derivative_y

    def hamiltonian(self, t):
        diags = np.zeros_like(self.state)
        for k in range(len(self.state)):
            i, j = get_row_col(k, len(self.y))
            diags[k] = self.V(self.x[i], self.y[j], t)
        V = scipy.sparse.diags(diags)
        return -0.5 * self.derivative + V

    def step(self, dt):
        I = scipy.sparse.identity(len(self.state))
        H = 0.5j * self.hamiltonian(self.t + dt / 2) * dt
        self.state = scipy.sparse.linalg.spsolve(I + H, (I - H).dot(self.state))
        self.t += dt
        return self.t, self.state


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits import mplot3d

    Nx = 200
    Ny = 200
    x_min = -10
    x_max = 10
    y_min = -10
    y_max = 10
    t = 0
    dt = 0.1
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    Y, X = np.meshgrid(y, x)

    def V(x, y, t):
        # return 0
        return (x ** 2 + y ** 2)

    def psi0(xx, yy):
        # stationary function for potential box
        # a = x[-1] - x[0]
        # b = y[-1] - y[0]
        # n = 2
        # m = 2
        # return np.sqrt(2 / a) * np.sin(np.pi * n / a * (xx - x[0])) * np.sqrt(2 / b) * np.sin(np.pi * m / b * (yy - y[0]))
        x0, y0 = 0, 0
        p0x, p0y = 0, 0
        a = 1
        a2 = a ** 2
        f = 1 / (np.pi * a2) ** (1 / 4)
        return f * np.exp(1j * (p0x * (xx - x0) + p0y * (yy - y0))) * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * a2))


    current_state = []
    for i in range(len(x)):
        for j in range(len(y)):
            current_state.append(psi0(x[i], y[j]))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlim(-.5, .5)
    # ax.plot_wireframe(X, Y, np.abs(list_to_matrix(current_state, len(y))))
    time_text = ax.text(0.95, 0.95, 0.95, '', transform=ax.transAxes)

    frame = None

    stepper = Stepper2d(current_state, t, x, y, V)

    def animate(i):
        global frame
        t, current_state = stepper.step(dt)
        if frame is not None:
            ax.collections.remove(frame)
        frame = ax.plot_surface(X, Y, np.real(list_to_matrix(current_state, len(y))), cmap='viridis', edgecolor=None)
        time_text.set_text(f't = {t}')
        print(f'{i} / {frames}')

    frames = 600
    anim = animation.FuncAnimation(fig, animate, interval=100, frames=frames)
    plt.show()
    # anim.save('cranknicolson2d.gif', writer='imagemagick', fps=10)
