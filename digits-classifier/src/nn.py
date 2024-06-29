import numpy as np
import arff
from sklearn.model_selection import train_test_split
import gzip
import pickle as cPickle
import os


def labels_to_onehot(j):
    # labels_id = []
    # for label in labels:
    #     if label not in labels_id:
    #         labels_id.append(label)

    # encoded = []

    # for label in labels:
    #     new_label = map(lambda l: 1 if l == label else 0, labels_id)
    #     encoded.append(list(new_label))

    # return encoded
    e = np.zeros(10)
    e[int(j)] = 1.0
    return e


def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx : start_idx + minibatch_size]

        yield X[batch_idx], y[batch_idx]


class Neuron:
    def __init__(self, num_weights):
        self.num_weights = num_weights
        self.weights = np.random.randn(num_weights)
        self.bias = np.random.randn()
        # self.weights = rgen.normal(loc=0.0, scale=0.1, size=num_weights)
        # self.bias = np.float_(0.0)

    def calc_z(self, X):
        self.z = np.dot(X, self.weights) + self.bias
        return self.z


class Network:
    def __init__(self, sizes, epochs=20, eta=0.1):
        self.num_layers = len(sizes)
        self.epochs = epochs
        self.eta = eta

        self.layers = []
        idx = 1
        for s in sizes[1:]:
            self.layers.append([Neuron(sizes[idx - 1]) for _ in range(s)])
            idx += 1

    def fit(self, X, y):

        losses = []
        for _ in range(self.epochs):
            for layer in self.layers:
                outputs = []
                zs = []

                # 1st pass to collect outputs
                for neuron in layer:
                    z = neuron.z(X)
                    output = self.sigmoid(z)
                    outputs.append(output)
                    zs.append(z)

                outputs = np.array(outputs).T
                zs = np.array(zs).T
                errors = y - outputs
                dadz = self.sigmoid_prime(zs)
                delta_out = (errors / y.shape[0]) * dadz

                # 2nd pass to update weights?
                for i, neuron in enumerate(layer):
                    # get the corresponding axis based on the neuron index
                    neuron_errors = errors.T[i]
                    neuron_delta_out = delta_out.T[i]
                    neuron_z = zs.T[i]

                    dcdw = X.T.dot(neuron_delta_out)

                    neuron.weights += self.eta * 2 * dcdw
                    neuron.bias += (
                        self.eta
                        * 2
                        * (self.sigmoid_prime(neuron_z) * neuron_errors).mean()
                    )

                loss = errors**2
                a = []
                for e in loss:
                    a.append(e.sum())
                losses.append(np.array(a).mean())

        return losses

    def fit2(self, X, y):
        for _ in range(self.epochs):
            minibatch_gen = minibatch_generator(X, y, 3)
            for X_, y_ in minibatch_gen:
                activations = [X_]
                zs = []
                inputs = X_
                # feedforward
                for layer in self.layers:
                    activations_ = []
                    z_ = []
                    for neuron in layer:
                        z = neuron.calc_z(inputs)
                        a = self.activation(z)
                        activations_.append(a)
                        z_.append(z)
                    inputs = np.asarray(activations_).T
                    activations.append(inputs)
                    zs.append(z_)
                # calculate dCdz for output layer
                dCda = (activations[-1] - y_) / y_.shape[0]
                dadz = []
                for neuron in self.layers[-1]:
                    dadz.append(self.activation_prime(neuron.z))
                error = (np.asarray(dadz).T * dCda).T

                # update output weights
                dCdw = np.dot(error, activations[-2])
                for neuron, w, e in zip(self.layers[-1], dCdw, error):
                    neuron.weights -= self.eta * w
                    neuron.bias -= self.eta * e.sum()

                for i in range(2, self.num_layers):
                    layer = self.layers[-i]
                    weights_next_layer = np.asarray(
                        [neuron.weights for neuron in self.layers[-i + 1]]
                    )
                    dadz = []
                    for neuron in layer:
                        dadz.append(self.activation_prime(neuron.z))

                    error = np.dot(weights_next_layer.T, error) * dadz

                    dCdw = np.dot(error, activations[-i - 1])
                    for neuron, w, e in zip(self.layers[-i], dCdw, error):
                        neuron.weights -= self.eta * w
                        neuron.bias -= self.eta * e.sum()

    def feedforward(self, X):
        activations = X
        for layer in self.layers:
            inputs = []
            for neuron in layer:
                z = neuron.calc_z(activations)
                a = self.activation(z)
                inputs.append(a)
            activations = np.asarray(inputs).T
        return activations

    def evaluate(self, test_data):
        test_results = [
            (np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data
        ]
        return sum(int(x == y) for (x, y) in test_results)

    def backprop(self):
        pass

    def activation(self, z):
        return self.sigmoid(z)

    def activation_prime(self, z):
        return self.sigmoid_prime(z)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def save_model(self, path):
        output = open(path, "wb")
        cPickle.dump(self.layers, output)
        output.close()

    @classmethod
    def from_model(cls, path):
        if os.path.exists(os.path.join(os.getcwd(), path)):
            model = open(path, "rb")
            data = cPickle.load(model)
            nn = cls([])
            nn.layers = data
            model.close()
            return nn
        else:
            raise FileNotFoundError("model doesn't exist")


def load_data():
    f = gzip.open("datasets/mnist.pkl.gz", "rb")
    training_data, validation_data, test_data = cPickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    # training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_inputs = tr_d[0]
    training_results = [labels_to_onehot(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    # validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    # validation_data = list(zip(validation_inputs, va_d[1]))
    # test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_inputs = te_d[0]
    test_results = [labels_to_onehot(y) for y in te_d[1]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_inputs, training_results, test_inputs, test_results)


def run():
    model_path = "model.pkl"
    nn = Network.from_model(model_path)
    (training_inputs, training_results, test_inputs, test_results) = load_data_wrapper()
    # data = arff.load(open("datasets/iris.arff"))["data"]
    # inputs = []
    # outputs = []
    # for item in data:
    #     inputs.append(item[:-1])
    #     outputs.append(item[-1])
    #     # outputs.append(labels_to_onehot(item[-1]))
    # outputs = labels_to_onehot(outputs)
    # inputs = np.array(inputs)
    # outputs = np.array(outputs)

    # # we MUST scale the pixel values into range of [-1, 1] for some reason?
    # # inputs = ((inputs / 255.0) - 0.5) * 2

    # X_train, X_test, y_train, y_test = train_test_split(
    #     inputs,
    #     outputs,
    #     test_size=0.2,
    #     random_state=1,
    #     shuffle=True,
    # )

    # nn = Network([4, 7, 3], epochs=25, eta=0.3)
    # nn.load_model("model.pkl")
    # print((nn.evaluate(list(zip(X_test, y_test))) / len(y_test)) * 100)
    # nn.fit2(X_train, y_train)

    # nn = Network([784, 10, 10], epochs=10, eta=3)
    # nn.fit2(np.array(training_inputs), np.array(training_results))
    # print(training_inputs[0])
    print((nn.evaluate(list(zip(test_inputs, test_results))) / len(test_results)) * 100)
    # nn.save_model("model.pkl")


if __name__ == "__main__":
    run()
