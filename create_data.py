import numpy as np

A = np.linspace(10, 0, 10, endpoint=False)[::-1]
s = np.linspace(2, 0, 10, endpoint=False)[::-1]
f1 = np.linspace(0, 5, 10, endpoint=False)[::-1]
a2 = np.linspace(0, 5, 10, endpoint=True)
f2 = np.linspace(0, 5, 10, endpoint=True)

# data = {'A': A,
#        's': s,
#        'f1': f1,
#        'a2': a2,
#        'f2': f2}

# np.savez('parameters', **data)

max_indices = [len(A), len(s), len(f1), len(a2), len(f2)]
indices = [0] * 5
inputs = []

while indices[0] < max_indices[0]:
    inputs.append([A[indices[0]], s[indices[1]], f1[indices[2]], a2[indices[3]], f2[indices[4]]])
    # go over every permutation
    for i in range(len(indices) - 1, -1, -1):
        indices[i] += 1
        if indices[i] >= max_indices[i] and i > 0:
            indices[i] = 0
        else:
            break

inputs = np.asarray(inputs)
print(inputs.shape)
np.save('inputs', inputs)
