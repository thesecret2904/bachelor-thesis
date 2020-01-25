import numpy as np
import sklearn.ensemble
import time
import matplotlib.pyplot as plt
from crank_nicolson import Stepper

inputs = np.load('shuffled_E_inputs.npy')
targets = np.load('shuffled_targets.npy')

x = np.zeros((2,))
stepper = Stepper(None, 0, x, None)
t0 = 0
dt = 0.1
t_max = 6
t = np.arange(t0, t_max + dt, dt)

training_in = inputs[:3 * len(inputs) // 4]
training_out = targets[:3 * len(targets) // 4]

testing_in = inputs[3 * len(inputs) // 4:]
testing_out = targets[3 * len(targets) // 4:]

regressors = []
predictions = []
training_predictions = []
N = 5
for i in range(N):
    t = time.time()
    regressors.append(sklearn.ensemble.GradientBoostingRegressor(learning_rate=0.05))
    regressors[-1].fit(training_in, training_out[:, i])
    predictions.append(regressors[-1].predict(testing_in))
    training_predictions.append(regressors[-1].predict(training_in))
    print(time.time() - t)

predictions = np.array(predictions).T
training_predictions = np.array(training_predictions).T
diff = np.abs(predictions - testing_out[:, :N])
print(predictions.shape, diff.shape)
print(diff.mean())
print(np.abs(training_predictions - training_out[:, :N]).mean())

bins = [i for i in range(N + 1)]
for i in range(10):
    plt.figure()
    plt.subplot(211)
    plt.hist(bins[:-1], bins, weights=testing_out[i, :N])
    plt.title('Testing Data')

    plt.subplot(212)
    plt.hist(bins[:-1], bins, weights=predictions[i, :N])
    plt.title('Regression Data')

    plt.gcf().tight_layout()
    plt.show()
