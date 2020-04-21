import taichi as ti

a = ti.Vector(2, dt=ti.f64, needs_grad=True, shape=3)
x = ti.var(dt=ti.f64, shape=(), needs_grad=True)
t = ti.var(dt=ti.f64, needs_grad=True, shape=2)

@ti.func
def complex_mult(z1, z2):
    result = ti.Vector([0., 0.], dt=ti.f64, needs_grad=True)
    print(z1[0])
    print(z2[0])
    result[0] = z1[0] * z2[0] - z1[1] * z2[1]
    result[1] = z1[0] * z2[1] + z1[1] * z2[0]
    return result


@ti.func
def complex_diff(z1, z2):
    result = ti.Vector([0., 0.], dt=ti.f64, needs_grad=True)
    result[0] = (z1[0] * z2[0] + z1[1] * z2[1]) / (ti.sqr(z2[0]) + ti.sqr(z2[1]))
    result[1] = (z1[1] * z2[0] - z1[0] * z2[1]) / (ti.sqr(z2[0]) + ti.sqr(z2[1]))
    return result

@ti.kernel
def init():
    a[0][0] = 4
    a[1][0] = 2


@ti.kernel
def test():
    # for i in range(2):
    #     a[None][i] =
    a[2] = complex_diff(a[0], a[1])
    x[None] = a[2][0]


init()
with ti.Tape(x):
    test()
print(x[None])
print(a.grad[0][0])
print(a.grad[1][0])
