import taichi as ti

a = ti.Vector(2, dt=ti.f64, needs_grad=True, shape=())
x = ti.var(dt=ti.f64, shape=(), needs_grad=True)
t = ti.var(dt=ti.f64, needs_grad=True, shape=2)


@ti.kernel
def init():
    x[None] = 2


@ti.kernel
def test():
    # for i in range(2):
    #     a[None][i] = 2
    a[None][0] = 2 * x[None]
    a[None][1] = 3 * x[None]
    t[0] = a[None][0]
    t[1] = a[None][1]


init()
with ti.Tape(t):
    test()
# print(x.grad[None])
