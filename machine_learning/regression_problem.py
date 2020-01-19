import numpy as np
import machine_learning.multilayer_perceptron as mlp
import matplotlib.pyplot as plt

N = 100
x = np.ones((1, N)) * np.linspace(0, 1, N)
t = np.sin(2 * np.pi * x) + np.cos(4 * np.pi * x) + np.random.randn(N) * 0.2

x = x.T
t = t.T

x = (x - x.mean(axis=0)) / x.var(axis=0)
t = (t - t.mean(axis=0)) / t.var(axis=0)

plt.plot(x, t, '.')
plt.show()
plt.figure()

training_inputs = x[0::2, :]
testing_inputs = x[1::4, :]
validation_inputs = x[3::4, :]

training_targets = t[0::2, :]
testing_targets = t[1::4, :]
validation_targets = t[3::4, :]

network = mlp.Multi_layer_perceptron(1, 3, 1)

network.train(training_inputs, training_targets, N=5000, learning_rate=0.01)
print(network.output_weights)
print(network.hidden_weights)
plt.plot(training_inputs, training_targets, '.', training_inputs, network.get_output(training_inputs), '.')
plt.show()
