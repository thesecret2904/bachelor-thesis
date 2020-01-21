import numpy as np

A = np.linspace(10, 0, 10, endpoint=False)[::-1]
s = np.linspace(2, 0, 10, endpoint=False)[::-1]
f1 = np.linspace(0, 5, 10, endpoint=False)[::-1]
a2 = np.linspace(0, 5, 10, endpoint=True)
f2 = np.linspace(0, 5, 10, endpoint=True)

data = {'A': A,
        's': s,
        'f1': f1,
        'a2': a2,
        'f2': f2}

np.savez('parameters', **data)
