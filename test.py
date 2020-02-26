import taichi as ti

b = ti.Vector(2, dt=ti.f32, shape=10)
a = ti.Vector(2, dt=ti.f32, shape=10)


@ti.func
def print_(v):
    print(v[0])
    print(v[1])


@ti.kernel
def t():
    for i in a:
        a[i][0] = i
        a[i][1] = -i


@ti.func
def complex_mult(z1, z2):
    result = ti.Vector([0, 0], dt=ti.f64)
    result[0] = z1[0] * z2[0] - z1[1] * z2[1]
    result[1] = z1[0] * z2[1] + z1[1] * z2[0]
    return result


@ti.func
def complex_diff(z1, z2):
    result = ti.Vector([0, 0], dt=ti.f64)
    l = ti.sqr(z2[0]) + ti.sqr(z2[1])
    result[0] = (z1[0] * z2[0] + z1[1] * z2[1]) / l
    result[1] = (z1[1] * z2[0] - z1[0] * z2[1]) / l
    return result


@ti.kernel
def thomas():
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
        for i in range(N):
            j = N - i - 1
            if j == N - 1:
                current[j] = right_side[j]
            else:
                current[j] = right_side[j] - complex_mult(left_upper_diag[j], current[j + 1])


@ti.kernel
def init():
    for i in current:
        current[i][0] = i
        current[i][1] = -i
        right_main_diag[i][0] = 1
        right_main_diag[i][1] = -1
        if i < N - 1:
            right_upper_diag[i][0] = i
            right_lower_diag[i][1] = i + 1

@ti.kernel
def test():
    z1 = ti.Vector([2, 5], dt=ti.f64)
    z2 = ti.Vector([3, -4], dt=ti.f64)
    print_(complex_mult(z1, z2))
    print_(complex_diff(z1, z2))
    
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

N = 4
left_upper_diag = ti.Vector(2, dt=ti.f64, shape=N - 1)
left_lower_diag = ti.Vector(2, dt=ti.f64, shape=N - 1)
left_main_diag = ti.Vector(2, dt=ti.f64, shape=N)
right_upper_diag = ti.Vector(2, dt=ti.f64, shape=N - 1)
right_lower_diag = ti.Vector(2, dt=ti.f64, shape=N - 1)
right_main_diag = ti.Vector(2, dt=ti.f64, shape=N)


current = ti.Vector(2, dt=ti.f64, shape=N)
right_side = ti.Vector(2, dt=ti.f64, shape=N)

# test()
init()
right_hand_side()
print(current.to_numpy()[:, :, 0])
print(right_side.to_numpy()[:, :, 0])
