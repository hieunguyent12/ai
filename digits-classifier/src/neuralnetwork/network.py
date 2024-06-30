import numpy as np
from cost import Cost, CostType
from activation import Activation, ActivationType
from sklearn.model_selection import train_test_split
import pickle
import os

from data_loader import load_iris, load_digits


class Neuron:
    def __init__(self, num_weights):
        self.weights = np.random.randn(num_weights)
        self.bias = np.random.randn()
        self.z = 0

    def net_input(self, X):
        self.z = np.dot(X, self.weights) + self.bias
        return self.z


class Layer:
    def __init__(self, num_neurons, num_weights):
        self.neurons = [Neuron(num_weights) for _ in range(num_neurons)]


class Network:
    def __init__(self, cost, activation):
        self.cost = cost
        self.activation = activation

    def createLayers(self, sizes):
        self.num_layers = len(sizes)
        self.layers = [Layer(sizes[s], sizes[s - 1]) for s in range(1, len(sizes))]

    def feedforward(self, X):
        activations = X
        for layer in self.layers:
            inputs = []
            for neuron in layer.neurons:
                z = neuron.net_input(activations)
                a = self.activation.activation(z)
                inputs.append(a)
            activations = np.asarray(inputs).T
        return activations

    def backprop(self, X, y):
        activations = [X]
        inputs = X
        # feedforward
        for layer in self.layers:
            activations_ = []
            for neuron in layer.neurons:
                z = neuron.net_input(inputs)
                a = self.activation.activation(z)
                activations_.append(a)
            inputs = np.asarray(activations_).T
            activations.append(inputs)
        # calculate dCdz for output layer
        dCda = self.cost.cost_prime(activations[-1], y) / y.shape[0]
        dadz = []
        for neuron in self.layers[-1].neurons:
            dadz.append(self.activation.activation_prime(neuron.z))
        error = (np.asarray(dadz).T * dCda).T

        # update output weights
        dCdw = np.dot(error, activations[-2])
        for neuron, w, e in zip(self.layers[-1].neurons, dCdw, error):
            neuron.weights -= self.eta * w
            neuron.bias -= self.eta * e.sum()

        for i in range(2, self.num_layers):
            layer = self.layers[-i]
            weights_next_layer = np.asarray(
                [neuron.weights for neuron in self.layers[-i + 1].neurons]
            )
            dadz = []
            for neuron in layer.neurons:
                dadz.append(self.activation.activation_prime(neuron.z))

            error = np.dot(weights_next_layer.T, error) * dadz

            dCdw = np.dot(error, activations[-i - 1])
            for neuron, w, e in zip(layer.neurons, dCdw, error):
                neuron.weights -= self.eta * w
                neuron.bias -= self.eta * e.sum()

    def fit(self, training_data, epochs=30, eta=0.1, minibatch_size=5):
        X_ = training_data[0]
        y_ = training_data[1]
        self.eta = eta
        for _ in range(epochs):
            minibatch_gen = self.minibatch_generator(X_, y_, minibatch_size)

            for X, y in minibatch_gen:
                self.backprop(X, y)

    def evaluate(self, test_data):
        test_results = [
            (np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data
        ]
        return sum(int(x == y) for (x, y) in test_results)

    def minibatch_generator(self, X, y, minibatch_size):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        for start_idx in range(
            0, indices.shape[0] - minibatch_size + 1, minibatch_size
        ):
            batch_idx = indices[start_idx : start_idx + minibatch_size]

            yield X[batch_idx], y[batch_idx]

    def save_model(self, path):
        output = open(path, "wb")
        pickle.dump(self.layers, output)
        output.close()

    def from_model(self, path):
        if os.path.exists(os.path.join(os.getcwd(), path)):
            model = open(path, "rb")
            data = pickle.load(model)
            self.layers = data
            model.close()
            return self
        else:
            raise FileNotFoundError("model doesn't exist at specified path")


def run():
    model_path = "model.pkl"
    cost = Cost.init(CostType.MSE)
    activation = Activation.init(ActivationType.SIGMOID)

    nn = Network(cost, activation)
    nn.createLayers([28 * 28, 10, 10])
    nn.from_model(model_path)

    training_inputs, training_results, test_inputs, test_results = load_digits()

    # X_train, X_test, y_train, y_test = train_test_split(
    #     inputs,
    #     outputs,
    #     test_size=0.2,
    #     random_state=1,
    #     shuffle=True,
    # )

    # nn.fit([training_inputs, training_results], epochs=10, eta=0.3, minibatch_size=10)
    print((nn.evaluate(list(zip(test_inputs, test_results))) / len(test_results)) * 100)
    # nn.save_model(model_path)


if __name__ == "__main__":
    run()
