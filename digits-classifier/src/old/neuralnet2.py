import numpy as np


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


class NeuralNet:
    def __init__(self, sizes, epochs=50, eta=0.001):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.epochs = epochs
        self.eta = eta
        self.__initWeightsAndBiases()

    def __initWeightsAndBiases(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def fit(self, X, expectedOutputs):
        for _ in range(self.epochs):
            # literally copied from the machine learning book but it still doesn't work
            # wtf?
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            for x, y in zip(X, expectedOutputs):
                x_ = np.reshape(x, (4, 1))
                y_ = np.reshape(y, (3, 1))
                activation = x_
                delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
                delta_nabla_w = [np.zeros(w.shape) for w in self.weights]

                activations = [x_]  # list to store all the activations, layer by layer
                zs = []  # list to store all the z vectors, layer by layer
                for b, w in zip(self.biases, self.weights):
                    z = np.dot(w, activation) + b
                    zs.append(z)
                    activation = sigmoid(z)
                    activations.append(activation)
                # backward pass
                delta = self.cost_derivative(activations[-1], y_) * sigmoid_prime(
                    zs[-1]
                )

                delta_nabla_w[-1] = np.dot(delta, activations[-2].transpose())
                delta_nabla_b[-1] = delta

                for l in range(2, self.num_layers):
                    z = zs[-l]
                    sp = sigmoid_prime(z)
                    delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
                    delta_nabla_b[-l] = delta
                    delta_nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

            self.weights = [
                w - (self.eta / len(X)) * nw for w, nw in zip(self.weights, nabla_w)
            ]
            self.biases = [
                b - (self.eta / len(X)) * nb for b, nb in zip(self.biases, nabla_b)
            ]
            # error = np.zeros((1, 3))
            # for x, y in zip(X, y):
            #     x_ = np.reshape(x, (4, 1))
            #     output = np.reshape(self.feedfoward(x_), (1, 3))
            #     error += (output - y)
            # a = np.dot(error, X)
            #
            # error /= len(X)
            #

    def cost_derivative(self, output, expectedOutput):
        return output - expectedOutput

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        print(test_results)
        return sum(int(x == y) for (x, y) in test_results)
