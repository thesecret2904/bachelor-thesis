import numpy as np

def permutations(n, limits):
    current = np.full(n, limits[0])
    yield current
    while True:
        for i in range(n-1, -1, -1):
            current[i] += 1
            if current[i] > limits[1] and i > 0:
                current[i] = limits[0]
            else:
                break
        if current[0] > 1:
            return
        yield current

if __name__ == '__main__':
    for i in permutations(2, (-1, 1)):
        print(i)