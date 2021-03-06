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
    a = parameters[0] / (ti.sqrt(2 * np.pi) * parameters[1])
    w = ti.sqr(parameters[1])
    e = a * ti.exp(-ti.sqr(t_eff) / (2 * w)) * (
            ti.sin(parameters[2] * t_eff) + parameters[3] * ti.sin(parameters[4] * t_eff))
    e_field[None] = e
    grad[0] = e / parameters[0]
    grad[1] = - e / parameters[1] + e * ti.sqr(t_eff) / (parameters[1] ** 3)
    grad[2] = a * ti.exp(-ti.sqr(t_eff) / (2 * w)) * ti.cos(parameters[2] * t_eff) * t_eff
    grad[3] = a * ti.exp(-ti.sqr(t_eff) / (2 * w)) * ti.sin(parameters[4] * t_eff)
    grad[4] = a * ti.exp(-ti.sqr(t_eff) / (2 * w)) * parameters[3] * ti.cos(parameters[4] * t_eff) * t_eff


# init matrix for crank-nicolson I +- i/2*H*dt
@ti.kernel
def init_main_diags():
    for i in range(N):
        right_main_diag[i][0] += 1 - right_main_diag[i][0]
        right_main_diag[i][1] += 0 - right_main_diag[i][1]
        left_main_diag[i][0] += 1 - left_main_diag[i][0]
        left_main_diag[i][1] += 0 - left_main_diag[i][1]


@ti.kernel
def init_side_diags(temp: ti.f64):
    for i in range(N - 1):
        left_upper_diag[i][1] += -temp - left_upper_diag[i][1]
        left_upper_diag[i][0] += 0 - left_upper_diag[i][0]
        left_lower_diag[i][1] += -temp - left_lower_diag[i][1]
        left_lower_diag[i][0] += 0 - left_lower_diag[i][0]
        right_upper_diag[i][1] += temp - right_upper_diag[i][1]
        right_upper_diag[i][0] += 0 - right_upper_diag[i][0]
        right_lower_diag[i][1] += temp - right_lower_diag[i][1]
        right_lower_diag[i][0] += 0 - right_lower_diag[i][0]


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
    for i in range(N):
        # both are set to 0 in a previous step
        left_main_diag[i][1] += (1 / h2 + pot[i] - e_field[None] * space[i]) * dt / 2
        right_main_diag[i][1] += -left_main_diag[i][1]


# complex multiplication and division (checked)
@ti.func
def complex_mult(z1, z2):
    result = ti.Vector([0., 0.], dt=ti.f64, needs_grad=True)
    result[0] = z1[0] * z2[0] - z1[1] * z2[1]
    result[1] = z1[0] * z2[1] + z1[1] * z2[0]
    return result


@ti.func
def complex_diff(z1, z2):
    result = ti.Vector([0., 0.], dt=ti.f64, needs_grad=True)
    result[0] = (z1[0] * z2[0] + z1[1] * z2[1]) / (ti.sqr(z2[0]) + ti.sqr(z2[1]))
    result[1] = (z1[1] * z2[0] - z1[0] * z2[1]) / (ti.sqr(z2[0]) + ti.sqr(z2[1]))
    return result


@ti.func
def complex_conj(z):
    z[1] = -z[1]
    return z


# calculating the right side of the system of equations (I-i/2*H*dt) * current (checked)
@ti.kernel
def right_hand_side_init():
    right_side[0] += complex_mult(right_main_diag[0], current[0]) + complex_mult(right_upper_diag[0], current[1]) - \
                     right_side[0]


@ti.kernel
def right_hand_side_main():
    for i in range(1, N - 1):
        right_side[i] += complex_mult(right_main_diag[i], current[i]) + complex_mult(right_upper_diag[i],
                                                                                     current[i + 1]) + complex_mult(
            right_lower_diag[i - 1], current[i - 1]) - right_side[i]


@ti.kernel
def right_hand_side_end():
    right_side[N - 1] += complex_mult(right_main_diag[N - 1], current[N - 1]) + complex_mult(right_lower_diag[N - 2],
                                                                                             current[N - 2]) - \
                         right_side[N - 1]


# @ti.kernel
def right_hand_side():
    right_hand_side_init()
    right_hand_side_main()
    right_hand_side_end()
    '''for i in current:
        if i == 0:
            right_side[i] = complex_mult(right_main_diag[i], current[i]) + complex_mult(right_upper_diag[i],
                                                                                        current[i + 1])
        elif i == N - 1:
            right_side[i] = complex_mult(right_main_diag[i], current[i]) + complex_mult(right_lower_diag[i - 1],
                                                                                        current[i - 1])
        else:
            right_side[i] = complex_mult(right_main_diag[i], current[i]) + complex_mult(right_upper_diag[i],
                                                                                        current[i + 1]) + complex_mult(
                right_lower_diag[i - 1], current[i - 1])'''


# thomas algorithm for solving linear system of equations (checked)
def thomas():
    thomas1()
    thomas2()


@ti.kernel
def thomas1_init():
    # left_upper_diag[0] += complex_diff(left_upper_diag[0], left_main_diag[0]) - left_upper_diag[0]
    left_upper_diag[0] += left_upper_diag[0] * left_main_diag[0] - left_upper_diag[0]
    right_side[0] += complex_diff(right_side[0], left_main_diag[0]) - right_side[0]


@ti.kernel
def thomas1_main():
    # prevent parallelization
    for _ in range(1):
        for i in range(1, N - 1):
            left_upper_diag[i] += complex_diff(left_upper_diag[i],
                                               left_main_diag[i] - complex_mult(left_upper_diag[i - 1],
                                                                                left_lower_diag[i - 1])) - \
                                  left_upper_diag[i]
            right_side[i] += complex_diff(right_side[i] - complex_mult(right_side[i - 1], left_lower_diag[i - 1]),
                                          left_main_diag[i] - complex_mult(left_upper_diag[i - 1],
                                                                           left_lower_diag[i - 1])) - right_side[i]


@ti.kernel
def thomas1_end():
    right_side[N - 1] += complex_diff(right_side[N - 1] - complex_mult(right_side[N - 2], left_lower_diag[N - 2]),
                                      left_main_diag[N - 1] - complex_mult(left_upper_diag[N - 2],
                                                                           left_lower_diag[N - 2])) - right_side[N - 1]


# @ti.kernel
def thomas1():
    thomas1_init()
    thomas1_main()
    thomas1_end()
    # prevent parallelization
    '''for _ in range(1):
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
                                                                              left_lower_diag[i - 1]))'''


@ti.kernel
def thomas2_init():
    current[N - 1] += right_side[N - 1] - current[N - 1]


@ti.kernel
def thomas2_main():
    # prevent parallelization
    for _ in range(1):
        for i in range(1, N):
            j = N - i - 1
            current[j] += right_side[j] - complex_mult(left_upper_diag[j], current[j + 1]) - current[j]


# @ti.kernel
def thomas2():
    thomas2_init()
    thomas2_main()
    '''if True:
        for i in range(N):
            j = N - i - 1
            if j == N - 1:
                current[j] = right_side[j]
            else:
                current[j] = right_side[j] - complex_mult(left_upper_diag[j], current[j + 1])'''


def step():
    # init_hamiltonian()
    init_main_diags()
    init_side_diags(dt / (4 * h2))
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
def get_grad(i: ti.i32):
    for k in range(5):
        # derivative of left_main_diag[i][1]
        grad[k] = grad[k] * (-space[i] * dt / 2)
        # derivative of left_upper_diag[0][0] after thomas1_init()
        # grad[k] = grad[k] * (-(dt / (4 * h2)) - 2 * left_upper_diag[i][0] * left_main_diag[i][1]) / (
        #             ti.sqr(left_main_diag[i][0]) + ti.sqr(left_main_diag[i][1]))
        grad[k] = grad[k] * left_upper_diag[i][1]


@ti.kernel
def set_var(i: ti.i32):
    # test_var[None] = (1 / h2 + pot[i] - e_field[None] * space[i]) * dt / 2
    test_var[None] = left_upper_diag[i][1]


def test_grad(i):
    init_main_diags()
    init_side_diags(dt / (4 * h2))
    electric_field(t + dt / 2)
    hamiltonian()
    right_hand_side()
    thomas1_init()
    set_var(i)


@ti.kernel
def init():
    for i in current:
        current[i][0] = initial[i]
        current[i][1] = 0
    # occupation[None] = 0


@ti.kernel
def get_occupation_init():
    projection[None] = (goal[0] * current[0] + goal[N - 1] * current[N - 1]) / 2


@ti.kernel
def get_occupation_main():
    for i in range(1, N - 1):
        projection[None] += goal[i] * current[i]


@ti.kernel
def get_occupation_end():
    projection[None] *= h
    occupation[None] = projection.norm_sqr()
    print(occupation[None])


# @ti.kernel
def get_occupation():
    get_occupation_init()
    get_occupation_main()
    get_occupation_end()
    '''projection = (goal[0] * current[0] + goal[N - 1] * current[N - 1]) / 2
    for i in range(1, N - 1):
        projection += goal[i] * current[i]
        if i == N - 2:
            projection *= h
            occupation[None] = projection.norm_sqr()
            print(occupation[None])'''


@ti.kernel
def optimze():
    for i in range(5):
        parameters[i] += parameters.grad[i] * learning_rate


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

e_field = ti.var(dt=ti.f64, shape=(), needs_grad=True)
# p = np.array([4.814029335249125, 0.841850321595887, 2.0975267122577352, 1.0035008897668007, 4.794475884008799])
p = np.asarray([5.120897103827091, 0.45307230437739066, 2.005192724801804, 1.2527029251083832, 4.535321375318067])
# p = np.array([10, .5, 5, 3, 7])
parameters = ti.var(dt=ti.f64, shape=5, needs_grad=True)

test_var = ti.var(dt=ti.f64, shape=(), needs_grad=True)
grad = ti.var(dt=ti.f64, shape=5)

initial = ti.var(dt=ti.f64, shape=N)
goal = ti.var(dt=ti.f64, shape=N, needs_grad=True)
occupation = ti.var(dt=ti.f64, shape=(), needs_grad=True)
projection = ti.Vector(2, dt=ti.f64, shape=(), needs_grad=True)

left_main_diag = ti.Vector(2, dt=ti.f64, shape=N, needs_grad=True)
left_upper_diag = ti.Vector(2, dt=ti.f64, shape=N - 1, needs_grad=True)
left_lower_diag = ti.Vector(2, dt=ti.f64, shape=N - 1, needs_grad=True)

right_main_diag = ti.Vector(2, dt=ti.f64, shape=N, needs_grad=True)
right_upper_diag = ti.Vector(2, dt=ti.f64, shape=N - 1, needs_grad=True)
right_lower_diag = ti.Vector(2, dt=ti.f64, shape=N - 1, needs_grad=True)
right_side = ti.Vector(2, dt=ti.f64, shape=N, needs_grad=True)

current = ti.Vector(2, dt=ti.f64, shape=N, needs_grad=True)

stepper = Stepper(None, t, x, V)
energys, eigenstates = stepper.get_stationary_solutions(k=250)

parameters.from_numpy(p)
space.from_numpy(x)
initial.from_numpy(np.real(eigenstates[0]))
goal.from_numpy(np.real(eigenstates[1]))

harmonic_pot()
# init()
# while t < t_max:
#     step()
#     t += dt
# get_occupation()

# testing grad
with ti.Tape(test_var):
    test_grad(0)
get_grad(0)
print(parameters.grad[0], parameters.grad[1], parameters.grad[2], parameters.grad[3], parameters.grad[4])
print(grad[0], grad[1], grad[2], grad[3], grad[4])
t += dt
exit()

# optimizing
learning_rate = 1e-5
occupations = []
grad = []
for i in range(1000):
    init()
    t = 0
    with ti.Tape(occupation):
        while t < t_max:
            step()
            t += dt
        get_occupation()
    occupations.append(occupation[None])
    grad.append(parameters.grad[0])
    print(parameters[0], ',', parameters[1], ',', parameters[2], ',', parameters[3], ',', parameters[4])
    print(parameters.grad[0], parameters.grad[1], parameters.grad[2], parameters.grad[3], parameters.grad[4])
    print(i)
    optimze()
plt.plot(occupations)
plt.xlabel('Iterations')
plt.ylabel('Occupation')
# plt.savefig('difftaichi3.pdf')
plt.show()
plt.plot(grad)
plt.show()
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

''''# Checking matrices
stepper.set_state(eigenstates[0], normalzie=False)
for i in range(5):
    print(f'step: {i}')
    control_step()'''
