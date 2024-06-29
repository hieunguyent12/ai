import random
import math


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


class NeuralNet:
    def __init__(self, sizes, epochs=50, eta=0.001):
        self.eta = eta
        self.epochs = epochs
        self.sizes = sizes

        self.__initWeightsAndBiases()
        self.weight_layers = len(self.weights)

    def __initWeightsAndBiases(self):
        self.weights = []
        self.biases = []

        idx = 1
        for s in self.sizes[1:]:
            prev_s = self.sizes[idx - 1]
            weight_layer = []

            for _ in range(s):
                weights = [random.gauss(0, 1) for _ in range(prev_s)]
                weight_layer.append(weights)

            self.weights.append(weight_layer)
            idx += 1

            bias_layer = [0 for _ in range(s)]
            self.biases.append(bias_layer)

        print(self.weights)
        print(self.biases)

    # def fit(self, X, y):
    #     """
    #     X: training examples
    #     y: training labels
    #     """
    #     for _ in range(self.epochs):
    #         predictions = self.predict(X)
    #
    #         errors = []
    #         costs = []
    #         for prediction, actual in zip(predictions, y):
    #             error = []
    #             cost = 0
    #             for a, b in zip(prediction, actual):
    #                 print("{}, {}".format(a, b))
    #                 cost += self.cost_derivative(a, b)
    #                 error.append((b - a))
    #             errors.append(error)
    #             costs.append(cost)
    #
    #         for x, cost in zip(X, costs):
    #             for weight_layer, bias_layer in zip(self.weights, self.biases):
    #                 for i in range(len(weight_layer)):
    #                     for j in range(len(weight_layer[i])):
    #                         weight_layer[i][n] -= self.eta * 2 * cost * x[n] / len(X)
    #                         bias_layer[i] -= self.eta * 2 * cost / len(X)
    #
    # def cost(self, output, expectedOutput):
    #     error = expectedOutput - output
    #     return error**2
    #
    # def cost_derivative(self, output, expectedOutput):
    #     return expectedOutput - output
    #
    # def activation(self, z):
    #     return z
    # return 1 if z >= 0 else 0

    def fit(self, X, y):
        for _ in range(self.epochs):
            predictions = self.predict(X)
            errors = []
            for prediction, actual in zip(predictions, y):
                error = []
                for a, b in zip(prediction, actual):
                    error.append((a-b))
                errors.append(error)

            # average the errors
            errors_avg = []
            error_count = len(errors)
            output_count = len(errors[0])
            for i in range(output_count):
                total = 0
                for error in errors:
                    total += error[i]

                errors_avg.append(total / len(y))

            for x in X:
                for weight_layer, bias_layer in zip(self.weights, self.biases):
                    for i in range(len(weight_layer)):  # loop through each neuron
                        for n in range(
                            len(weight_layer[i])
                        ):  # loop through each weight that is connected to the neuron
                            z = x[n] * weight_layer[i][n] + bias_layer[i]
                            weight_layer[i][n] += (
                                self.eta * sigmoid_derivative(x[n]) * errors_avg[i] * x[n]
                            )

                            bias_layer[i] += (
                                self.eta * sigmoid_derivative(x[n]) * errors_avg[i]
                            )

    def activation(self, z):
        return sigmoid(z)

    def forward(self):
        pass

    def backprop(self):
        pass

    def predict(self, X, log=False):
        predictions = []
        for x in X:
            prediction = []
            for layerIdx in range(self.weight_layers):
                weight_layer = self.weights[layerIdx]

                for i, weights in enumerate(weight_layer):
                    net_z = 0
                    for feature, weight in zip(x, weights):
                        net_z += feature * weight + self.biases[layerIdx][i]

                    if log:
                        if self.activation(net_z) >= 0.5:
                            prediction.append(1)
                        else:
                            prediction.append(0)
                    else:
                        prediction.append(self.activation(net_z))
            predictions.append(prediction)
        return predictions
