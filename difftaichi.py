import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from crank_nicolson import Stepper
# for testing
import scipy.sparse


# function for harmonic oscillator (checked)
@ti.kernel
def harmonic_pot():
    w = 1
    for i in space:
        pot[i] = ti.sqr(w) / 2 * ti.sqr(space[i])


# function for electric field (checked)
@ti.kernel
def electric_field(t: ti.f64):
    t_eff = t - t_h
    a = parameters[0] / ((ti.sqrt(2 * np.pi) * parameters[1]))
    w = ti.sqr(parameters[1])
    e = a * ti.exp(-ti.sqr(t_eff) / (2 * w)) * (
            ti.sin(parameters[2] * t_eff) + parameters[3] * ti.sin(parameters[4] * t_eff))
    for i in e_field:
        e_field[i] = e


# init matrix for crank-nicolson I +- i/2*H*dt 
@ti.kernel
def init_hamiltonian(temp: ti.f64):
    for i in right_main_diag:
        right_main_diag[i][0] = 1
        right_main_diag[i][1] = 0
        left_main_diag[i][0] = 1
        left_main_diag[i][1] = 0
        if i < N - 1:
            left_upper_diag[i][1] = -temp
            left_upper_diag[i][0] = 0
            left_lower_diag[i][1] = -temp
            left_lower_diag[i][0] = 0
            right_upper_diag[i][1] = temp
            right_upper_diag[i][0] = 0
            right_lower_diag[i][1] = temp
            right_lower_diag[i][0] = 0


# complete hamilton operator 
@ti.kernel
def hamiltonian():
    for i in left_main_diag:
        left_main_diag[i][1] = (1 / h2 + pot[i] - e_field[i] * space[i]) * dt / 2
        right_main_diag[i][1] = -left_main_diag[i][1]


# complex multiplication and division (checked)
@ti.func
def complex_mult(z1, z2):
    result = ti.Vector([0., 0.], dt=ti.f64)
    result[0] = z1[0] * z2[0] - z1[1] * z2[1]
    result[1] = z1[0] * z2[1] + z1[1] * z2[0]
    return result


@ti.func
def complex_diff(z1, z2):
    result = ti.Vector([0., 0.], dt=ti.f64)
    l = ti.sqr(z2[0]) + ti.sqr(z2[1])
    result[0] = (z1[0] * z2[0] + z1[1] * z2[1]) / l
    result[1] = (z1[1] * z2[0] - z1[0] * z2[1]) / l
    return result


@ti.func
def complex_conj(z):
    z[1] = -z[1]
    return z


# calculating the right side of the system of equations (I-i/2*H*dt) * current (checked)
@ti.kernel
def right_hand_side():
    for i in current:
        if i == 0:
            right_side[i] = complex_mult(right_main_diag[i], current[i]) + complex_mult(right_upper_diag[i],
                                                                                        current[i + 1])
        elif i == N - 1:
            right_side[i] = complex_mult(right_main_diag[i], current[i]) + complex_mult(right_lower_diag[i - 1],
                                                                                        current[i - 1])
        else:
            right_side[i] = complex_mult(right_main_diag[i], current[i]) + complex_mult(right_upper_diag[i],
                                                                                        current[i + 1]) + complex_mult(
                right_lower_diag[i - 1], current[i - 1])


# thomas algorithm for solving linear system of equations (checked)
def thomas():
    thomas1()
    thomas2()


@ti.kernel
def thomas1():
    # prevent parallelization
    if True:
        for i in range(N):
            if i == 0:
                left_upper_diag[i] = complex_diff(left_upper_diag[i], left_main_diag[i])
                right_side[i] = complex_diff(right_side[i], left_main_diag[i])
            else:
                if i < N - 1:
                    left_upper_diag[i] = complex_diff(left_upper_diag[i],
                                                      left_main_diag[i] - complex_mult(left_upper_diag[i - 1],
                                                                                       left_lower_diag[i - 1]))
                right_side[i] = complex_diff(right_side[i] - complex_mult(right_side[i - 1], left_lower_diag[i - 1]),
                                             left_main_diag[i] - complex_mult(left_upper_diag[i - 1],
                                                                              left_lower_diag[i - 1]))


@ti.kernel
def thomas2():
    if True:
        for i in range(N):
            j = N - i - 1
            if j == N - 1:
                current[j] = right_side[j]
            else:
                current[j] = right_side[j] - complex_mult(left_upper_diag[j], current[j + 1])


def step():
    init_hamiltonian(dt / (4 * h2))
    electric_field(t + dt / 2)
    hamiltonian()
    right_hand_side()
    thomas()


'''def control_step():
    global t
    init_hamiltonian(dt / (4 * h2))
    electric_field(t + dt / 2)
    hamiltonian()
    right_hand_side()

    H = stepper.hamiltonian(t + dt / 2).todense()
    I = np.eye(N)
    left_crank = I + 0.5j * H * dt
    right_crank = I - 0.5j * H * dt

    lm = left_main_diag.to_numpy()[:, :, 0]
    lu = left_upper_diag.to_numpy()[:, :, 0]
    ll = left_lower_diag.to_numpy()[:, :, 0]
    rm = right_main_diag.to_numpy()[:, :, 0]
    ru = right_upper_diag.to_numpy()[:, :, 0]
    rl = right_lower_diag.to_numpy()[:, :, 0]

    lm = np.asarray([complex(i[0], i[1]) for i in lm])
    lu = np.asarray([complex(i[0], i[1]) for i in lu])
    ll = np.asarray([complex(i[0], i[1]) for i in ll])
    left_taichi = scipy.sparse.diags([ll, lm, lu], [-1, 0, 1]).todense()
    rm = np.asarray([complex(i[0], i[1]) for i in rm])
    ru = np.asarray([complex(i[0], i[1]) for i in ru])
    rl = np.asarray([complex(i[0], i[1]) for i in rl])
    right_taichi = scipy.sparse.diags([rl, rm, ru], [-1, 0, 1]).todense()

    print(np.max(left_crank - left_taichi))
    print(np.max(right_crank - right_taichi))

    right_side_taichi = np.asarray([complex(i[0], i[1]) for i in right_side.to_numpy()[:, :, 0]])
    right_side_crank = right_crank.dot(stepper.state)

    print(np.max(right_side_taichi - right_side_crank))

    thomas()

    stepper.step(dt)
    current_taichi = np.asarray([complex(i[0], i[1]) for i in current.to_numpy()[:, :, 0]])
    print(np.max(current_taichi - stepper.state))
    t += dt'''


@ti.kernel
def init():
    for i in current:
        current[i][0] = initial[i]


@ti.kernel
def get_occupation():
    projection = (goal[0] * current[0] + goal[N - 1] * current[N - 1]) / 2
    for i in range(1, N - 1):
        projection += goal[i] * current[i]
        if i == N - 2:
            projection *= h
            occupation[None] = projection.norm_sqr()
            print(occupation[None])


w = 1.


def V(x: np.ndarray, t: float):
    return (w ** 2) / 2 * np.square(x)


ti.init(arch=ti.cuda, default_fp=ti.f64)
# number of sample points
N = 1000
# x domain
x_min = -50
x_max = 50
# starting point
t = 0
t_max = 6
t_h = (t_max + t) / 2
# create list of all x points
x = np.linspace(x_min, x_max, N, endpoint=True, dtype=np.float64)
h = x[1] - x[0]
h2 = (x[1] - x[0]) ** 2
# time step
dt = 0.1

space = ti.var(dt=ti.f64, shape=N)
pot = ti.var(dt=ti.f64, shape=N)

e_field = ti.var(dt=ti.f64, shape=N)
p = np.array([10., 1., 5., 1., 5.])
parameters = ti.var(dt=ti.f64, shape=5, needs_grad=True)

initial = ti.var(dt=ti.f64, shape=N)
goal = ti.var(dt=ti.f64, shape=N)
occupation = ti.var(dt=ti.f64, shape=(), needs_grad=True)

left_main_diag = ti.Vector(2, dt=ti.f64, shape=N)
left_upper_diag = ti.Vector(2, dt=ti.f64, shape=N - 1)
left_lower_diag = ti.Vector(2, dt=ti.f64, shape=N - 1)

right_main_diag = ti.Vector(2, dt=ti.f64, shape=N)
right_upper_diag = ti.Vector(2, dt=ti.f64, shape=N - 1)
right_lower_diag = ti.Vector(2, dt=ti.f64, shape=N - 1)
right_side = ti.Vector(2, dt=ti.f64, shape=N)

current = ti.Vector(2, dt=ti.f64, shape=N)

stepper = Stepper(None, t, x, V)
energys, eigenstates = stepper.get_stationary_solutions(k=250)

parameters.from_numpy(p)
space.from_numpy(x)
initial.from_numpy(np.real(eigenstates[0]))
goal.from_numpy(np.real(eigenstates[1]))

harmonic_pot()

init()
t = 0
'''while t < t_max:
    step()
    t += dt
get_occupation()'''

with ti.Tape(occupation):
    step()

'''fig = plt.figure()
ax = plt.axes()
plt.ylim((-1, 1))
plt.xlim(x_min, x_max)
line, = plt.plot(x, eigenstates[0])
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)


def animate(i):
    global t
    step()
    t += dt
    time_text.set_text(str(t))
    line.set_data(x, current.to_numpy()[:, 0, 0])
    return line, time_text


anim = animation.FuncAnimation(fig, animate, blit=True, interval=100)
plt.show()'''

# Checking matrices
'''stepper.set_state(eigenstates[0], normalzie=False)
for i in range(5):
    print(f'step: {i}')
    control_step()'''
