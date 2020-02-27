import taichi as ti

a = ti.Vector(2, dt=ti.f32, shape=10)
b = ti.var(dt=ti.f32, shape=10)
sum = ti.Vector(2, dt=ti.f32, shape=())
total = ti.var(dt=ti.f32, shape=())



@ti.kernel
def test():
    for i in a:
        a[i][0] = i + 1
        a[i][1] = -a[i][0]
        b[i] = i + 1
    for i in a:
        sum[None] += a[i] * b[i]
    print(sum[None][0])
    total[None] = sum.norm_sqr()
    print(total[None])
    

test()
