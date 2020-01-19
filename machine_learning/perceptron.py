import numpy as np


class Perceptron():
    def __init__(self, number_of_inputs, number_of_outputs, activation_func=None):
        number_of_inputs += 1  # add one for bias node
        self.weights = np.random.rand(number_of_inputs, number_of_outputs)
        self.weights -= 0.5
        self.activation_func = activation_func

    def get_output(self, inputs):
        inputs = np.concatenate((inputs, -np.ones((len(inputs), 1))), axis=1)
        out = np.dot(inputs, self.weights)
        if self.activation_func is not None:
            out[:, :] = self.activation_func(out[:, :])
        else:
            out = np.where(out > 0, 1, 0)
        return out

    def train_step(self, inputs, targets, learning_rate):
        out = self.get_output(inputs)
        inputs = np.concatenate((inputs, -np.ones((len(inputs), 1))), axis=1)
        self.weights -= learning_rate * np.dot(np.transpose(inputs), out - targets)

    def train(self, inputs, targets, learing_rate=0.25, tol=1e-3, N=1000):
        for i in range(N):
            old_weights = self.weights.copy()
            self.train_step(inputs, targets, learing_rate)
            max_change = -1
            for j in range(len(self.weights)):
                for k in range(len(self.weights[j])):
                    max_change = max(max_change, abs(self.weights[j][k] - old_weights[j][k]))
            if max_change < tol:
                break


if __name__ == '__main__':
    # Test Perceptron on logical OR-Function
    inputs = []
    targets = []
    for i in range(2):
        for j in range(2):
            inputs.append([i, j])
            targets.append([i or j])
    inputs = np.asarray(inputs)
    targets = np.asarray(targets)
    print(inputs)
    print(targets)
    perceptron = Perceptron(2, 1)
    perceptron.train(inputs, targets)
    print(perceptron.get_output(inputs))
