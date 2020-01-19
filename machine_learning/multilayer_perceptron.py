import numpy as np


class Multi_layer_perceptron:
    def __init__(self, number_of_inputs, number_of_hidden_nodes, number_of_output_nodes, activation=3.0,
                 outtype='linear'):
        number_of_inputs += 1
        self.hidden_weights = (2 * np.random.rand(number_of_inputs, number_of_hidden_nodes) - 1) / np.sqrt(
            number_of_inputs)
        number_of_hidden_nodes += 1
        self.output_weights = (2 * np.random.rand(number_of_hidden_nodes, number_of_output_nodes) - 1) / np.sqrt(
            number_of_hidden_nodes)
        self.outtype = outtype

        self.hidden_change = np.zeros_like(self.hidden_weights)
        self.out_change = np.zeros_like(self.output_weights)

        def activation_function(x):
            return 1 / (1 + np.exp(-activation * x))

        self.activation_function = activation_function

    def get_output(self, inputs, full=False):
        inputs = np.concatenate((inputs, -np.ones((len(inputs), 1))), axis=1)
        hidden_activation = self.activation_function(np.dot(inputs, self.hidden_weights))
        hidden_activation = np.concatenate((hidden_activation, -np.ones((len(hidden_activation), 1))), axis=1)
        if self.outtype == 'sigmoid':
            output = self.activation_function(np.dot(hidden_activation, self.output_weights))
        elif self.outtype == 'linear':
            output = np.dot(hidden_activation, self.output_weights)
        else:
            raise ValueError("Unknown outtype! Possible values are: 'linear' and 'sigmoid'")
        if full:
            return hidden_activation, output
        else:
            return output

    def trainings_step(self, inputs, targets, learning_rate):
        hidden_output, output = self.get_output(inputs, True)
        inputs = np.concatenate((inputs, -np.ones((len(inputs), 1))), axis=1)

        if self.outtype == 'sigmoid':
            out_error = (output - targets) * output * (1 - output)
        elif self.outtype == 'linear':
            out_error = (output - targets)
        print(np.max(np.abs(out_error)))
        hidden_error = hidden_output * (1 - hidden_output) * (
            np.dot(out_error, np.transpose(self.output_weights)))

        self.hidden_change = learning_rate * np.dot(np.transpose(inputs),
                                                    hidden_error[:, :-1]) + .9 * self.hidden_change
        self.out_change = learning_rate * np.dot(np.transpose(hidden_output), out_error) + 0.9 * self.out_change
        self.hidden_weights -= self.hidden_change
        self.output_weights -= self.out_change

    def train(self, inputs, targets, learning_rate=0.25, tol=1e-3, N=1000):
        change = [i for i in range(len(inputs))]
        self.hidden_change = np.zeros_like(self.hidden_weights)
        self.out_change = np.zeros_like(self.output_weights)
        for i in range(N):
            old_out_weights = self.output_weights.copy()
            old_hidden_weitghs = self.hidden_weights.copy()
            self.trainings_step(inputs, targets, learning_rate)
            max_change = -1
            for j in range(len(self.output_weights)):
                for k in range(len(self.output_weights[j])):
                    max_change = max(max_change, abs(self.output_weights[j][k] - old_out_weights[j][k]))
            for j in range(len(self.hidden_weights)):
                for k in range(len(self.hidden_weights[j])):
                    max_change = max(max_change, abs(self.hidden_weights[j][k] - old_hidden_weitghs[j][k]))
            if max_change < tol:
                break
            np.random.shuffle(change)
            inputs = inputs[change]
            targets = targets[change]


if __name__ == '__main__':
    anddata = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
    xordata = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

    p = Multi_layer_perceptron(2, 2, 1, outtype='sigmoid')
    q = Multi_layer_perceptron(2, 2, 1, outtype='sigmoid')

    p.train(anddata[:, :-1], anddata[:, 2:3], tol=1e-9, N=20000)
    q.train(xordata[:, :-1], xordata[:, 2:3], tol=1e-9, N=10000)

    P = p.get_output(anddata[:, :-1])
    Q = q.get_output(xordata[:, :-1])
    print(np.round(P))
    print(np.round(Q))
