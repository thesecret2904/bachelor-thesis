import numpy as np


class Network:
    def __init__(self, architectur: tuple, random_bound=0.5):
        self.weights = []
        for i in range(1, len(architectur)):
            # the +1 accommodates for the bias node
            self.weights.append(2 * np.random.rand(architectur[i], architectur[i - 1] + 1) - random_bound)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_gradient(x):
        return Network.sigmoid(x) * (1 - Network.sigmoid(x))

    def get_output(self, input):
        shape = input.shape
        current = input
        if len(shape) == 1:
            for i in range(0, len(self.weights)):
                # append bias node
                current = np.concatenate(([1], current))
                # calculate the activation for every node
                current = Network.sigmoid(np.dot(self.weights[i], current))
        else:
            for i in range(0, len(self.weights)):
                # append bias node
                current = np.concatenate((np.ones((1, current.shape[1])), current))
                # calculate the activation for every node
                current = Network.sigmoid(np.dot(self.weights[i], current))
        return current.T

    def get_activations(self, input):
        shape = input.shape
        accumulated_weights = []
        activations = [input]
        current = input
        if len(shape) == 1:
            for i in range(0, len(self.weights)):
                # append bias node
                current = np.concatenate(([1], current))
                # calculate the activation for every node
                current = np.dot(self.weights[i], current)
                accumulated_weights.append(current)
                current = Network.sigmoid(current)
                activations.append(current)
        else:
            for i in range(0, len(self.weights)):
                # append bias node
                current = np.concatenate((np.ones((1, current.shape[1])), current))
                # calculate the activation for every node
                current = np.dot(self.weights[i], current)
                accumulated_weights.append(current)
                current = Network.sigmoid(current)
                activations.append(current)
        return accumulated_weights, activations

    def cost(self, inputs, targets, regularization=0):
        m = inputs.shape[1]  # number of training exmaples
        outputs = self.get_output(inputs)
        cost = 0
        for i in range(m):
            cost += -np.dot(targets[i], np.log(outputs[i])) - np.dot(1 - targets[i], np.log(1 - outputs[i]))
        cost /= m
        if regularization > 0:
            extra_cost = 0
            for weight in self.weights:
                extra_cost += np.sum(np.square(weight[:, 1:]))
            cost += regularization / (2 * m) * extra_cost
        return cost

    def cost_gradient(self, inputs, targets, reuglarization=0):
        accumulated_weights, activations = self.get_activations(inputs)
        m = inputs.shape[1]
        gradients = [np.zeros_like(w) for w in self.weights]
        for i in range(m):
            deltas = []
            deltas.append(activations[-1][:, i] - targets[i])
            gradients[-1] += np.dot(deltas[-1], activations[-1][:, i])
            for j in range(len(activations) - 2, 0, -1):
                deltas.append(
                    np.dot(weights[j - 1], deltas[-1]) * Network.sigmoid_gradient(accumulated_weights[j - 1][:, i]))
                deltas[-1] = deltas[-1][1:]
                gradients[j] += np.dot(deltas[-1], activations[-1][:, i].T)
        gradients = [gradient / m for gradient in gradients]
        return gradients

    def get_cost_function(self, inputs, targets, regularization):
        def J(weigths_array):
            weights = []
            current_index = 0
            for w in self.weights:
                shape = w.shape
                weights.append(np.reshape(weigths_array[current_index:current_index + shape[0] * shape[1]], shape))
                current_index += shape[0] * shape[1]
            m = inputs.shape[1]  # number of training exmaples
            outputs = self.get_output(inputs)
            cost = 0
            for i in range(m):
                cost += -np.dot(targets[i], np.log(outputs[i])) - np.dot(1 - targets[i], np.log(1 - outputs[i]))
            cost /= m
            if regularization > 0:
                extra_cost = 0
                for weight in weights:
                    extra_cost += np.sum(np.square(weight[:, 1:]))
                cost += regularization / (2 * m) * extra_cost
            return cost
        return J


if __name__ == '__main__':
    import scipy.io as sio
    import matplotlib.pyplot as plt

    trainings_data = sio.loadmat('ex4data1.mat')
    inputs = trainings_data['X'].T
    targets = trainings_data['y']
    template = [i + 1 for i in range(10)]
    targets = np.asarray([t == template for t in targets], 'float64')

    network = Network((400, 25, 10))
    weights = sio.loadmat('ex4weights.mat')
    weights = [weights['Theta1'], weights['Theta2']]
    network.weights = weights
    J = network.get_cost_function(inputs, targets, 1)
    print(f'cost = {J(np.concatenate((weights[0], weights[1]), None))}')
    gradients = network.cost_gradient(inputs, targets)
    print(f'cost = {network.cost(inputs, targets, 1)}')
