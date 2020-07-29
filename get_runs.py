import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import tikzplotlib

examples = [i for i in range(1000, 51000, 1000)]
loss = []

for i in examples:
    __Path__ = f'runs/set_size/run-set_size_{i}-tag-cross_validation_loss__mse_.csv'
    data = np.loadtxt(__Path__, skiprows=1, usecols=2, delimiter=',')
    loss.append(data[-1])

# plt.plot(examples, loss)
loss = savgol_filter(loss, 15, 3)
plt.plot(examples, loss)
plt.xlabel('Training set size')
plt.ylabel('MSE on testing set')
tikzplotlib.save('harmonic_oscillator/set_size.tex')
plt.savefig('harmonic_oscillator/set_size.pdf')
plt.show()
