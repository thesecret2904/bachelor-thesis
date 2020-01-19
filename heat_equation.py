import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def step(current, x, dt):
    dx2 = (x[1] - x[0]) ** 2
    I = scipy.sparse.identity(len(current))
    main_diag = np.ones((len(current),))
    lower_diag = np.ones((len(current) - 1,))
    upper_diag = np.ones((len(current) - 1,))
    main_diag = -2 * main_diag
    derivative2 = scipy.sparse.diags([lower_diag, main_diag, upper_diag], [-1, 0, 1]) / dx2
    A = I - dt / 2 * derivative2
    b = (I + dt / 2 * derivative2).dot(current)
    solution = scipy.sparse.linalg.spsolve(A, b)
    return solution


def initial(x):
    return 10 * np.sin(np.pi * x)

fig = plt.figure()
ax = plt.axes()

x = np.linspace(0, 1, 100)
dt = (x[1] - x[0]) / 2
current = initial(x)

line, = plt.plot(x, current)


def animate(i):
    global current
    current = step(current, x, dt)
    line.set_data(x, current)

anim = animation.FuncAnimation(fig, animate, interval=10)
plt.show()