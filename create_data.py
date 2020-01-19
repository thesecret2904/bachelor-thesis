import numpy as np

A = np.linspace(10, 0, 10, endpoint=False)[::-1]
s = np.linspace(1, 0, 5, endpoint=False)[::-1]
a1 = np.linspace(0, 10, 10)
f1 = np.linspace(5, 0, 10, endpoint=False)[::-1]
a2 = a1.copy()
f2 = np.linspace(10, 5, 10, endpoint=False)[::-1]

data = {'A': A,
        's': s,
        'a1': a1,
        'f1': f1,
        'a2': a2,
        'f2': f2}

np.savez('parameters', **data)
