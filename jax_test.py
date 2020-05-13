import jax.numpy as np
from jax import grad

def test(x):
    f1 = x ** 2
    f2 = 4
    f2 = f2 * f1
    return f2

g = grad(test)
print(test(2))
print(g(2.))

