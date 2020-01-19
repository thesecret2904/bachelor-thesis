import scipy.integrate
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


def get_index(row, col, rows, cols):
    if row < 0:
        row = rows + row
    elif row >= rows:
        row %= rows
    if col < 0:
        col = cols + col
    elif col >= cols:
        col %= cols
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


def step_ode(current_state, potential, x, y, t, dt, boundary='fixed'):
    # assuming current_state has the following form: current_state[(x0,y0),(x0, y1),..., (x0, yn),(x1,y0),(x1,y1),...]
    dx2_inv = 1 / (x[1] - x[0]) ** 2
    dy2_inv = 1 / (y[1] - y[0]) ** 2

    def right_hand_side(t, state):
        result = []
        for k in range(len(state)):
            i, j = get_row_col(k, len(y))
            sum = 0
            if boundary == 'fixed':
                if i > 1:
                    sum -= 1j * dx2_inv / 24 * state[get_index(i - 2, j, len(x), len(y))]
                if j > 1:
                    sum -= 1j * dy2_inv / 24 * state[get_index(i, j - 2, len(x), len(y))]
                if i > 0:
                    sum += 2j / 3 * dx2_inv * state[get_index(i - 1, j, len(x), len(y))]
                if j > 0:
                    sum += 2j / 3 * dy2_inv * state[get_index(i, j - 1, len(x), len(y))]
                sum -= 1j * (5 / 4 * (dx2_inv + dy2_inv) + potential(x[i], y[j], t)) * state[k]
                if i < len(x) - 1:
                    sum += 2j / 3 * dx2_inv * state[get_index(i + 1, j, len(x), len(y))]
                if j < len(y) - 1:
                    sum += 2j / 3 * dy2_inv * state[get_index(i, j + 1, len(x), len(y))]
                if i < len(x) - 2:
                    sum -= 1j * dx2_inv / 24 * state[get_index(i + 2, j, len(x), len(y))]
                if j < len(y) - 2:
                    sum -= 1j * dy2_inv / 24 * state[get_index(i, j + 2, len(x), len(y))]
            elif boundary == 'periodic':
                sum -= 1j * dx2_inv / 24 * state[get_index(i - 2, j, len(x), len(y))]
                sum -= 1j * dy2_inv / 24 * state[get_index(i, j - 2, len(x), len(y))]
                sum += 2j / 3 * dx2_inv * state[get_index(i - 1, j, len(x), len(y))]
                sum += 2j / 3 * dy2_inv * state[get_index(i, j - 1, len(x), len(y))]
                sum -= 1j * (5 / 4 * (dx2_inv + dy2_inv) + potential(x[i], y[j], t)) * state[k]
                sum += 2j / 3 * dx2_inv * state[get_index(i + 1, j, len(x), len(y))]
                sum += 2j / 3 * dy2_inv * state[get_index(i, j + 1, len(x), len(y))]
                sum -= 1j * dx2_inv / 24 * state[get_index(i + 2, j, len(x), len(y))]
                sum -= 1j * dy2_inv / 24 * state[get_index(i, j + 2, len(x), len(y))]
            else:
                raise ValueError(f'Unknown boundary condition: {boundary}')
            result.append(sum)
        return result

    result = scipy.integrate.solve_ivp(right_hand_side, (t, t + dt), current_state)
    return result.y[:, -1]


if __name__ == '__main__':
    Nx = 300
    Ny = 300
    x_min = -75
    y_min = -75
    x_max = 75
    y_max = 75
    t = 0
    dt = 0.1
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    Y, X = np.meshgrid(y, x)


    def V(x, y, t):
        return (x ** 2 + y ** 2) / 100


    def psi0(x, y):
        x0, y0 = 0, 0
        p0x, p0y = 5, 0
        a = 5
        a2 = a ** 2
        f = 1 / (np.pi * a2) ** (1 / 4)
        return f * np.exp(1j * (p0x * (x - x0) + p0y * (y - y0))) * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * a2))


    current_state = []
    for i in range(len(x)):
        for j in range(len(y)):
            current_state.append(psi0(x[i], y[j]))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_zlim(0, 1)
    # ax.plot_wireframe(X, Y, np.abs(list_to_matrix(current_state, len(y))))
    time_text = ax.text(0.95, 0.95, 0.95, '', transform=ax.transAxes)

    frame = None
    n = 0
    t_m = 0
    def animate(i):
        t0 = time.time()
        global frame, current_state, t, frames, n, t_m
        t += dt
        if frame is not None:
            ax.collections.remove(frame)
        current_state = step_ode(current_state, V, x, y, t, dt, 'periodic')
        frame = ax.plot_surface(X, Y, np.abs(list_to_matrix(current_state, len(y))), cmap='viridis', edgecolor='none')
        time_text.set_text(f't = {t}')

        t1 = time.time()
        progress = i / frames
        n += 1
        t_m = 1 / n * ((n - 1) * t_m + (t1 - t0))
        remaining_time = t_m * (frames - i)
        hours, sec = remaining_time // 3600, remaining_time % 3600
        min, sec = sec // 60, sec % 60
        print(f'Progress: {progress:.1%}')
        print(f'remaining times: {int(hours)}:{int(min)}:{int(sec)}')

    # animation
    frames = 400
    # starting animation
    anim = animation.FuncAnimation(fig, animate, interval=10, frames=frames)
    anim.save('animation2d.gif', writer='imagemagick', fps=10)
    # plt.show()
