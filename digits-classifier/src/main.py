from neuralnet2 import NeuralNet
import arff
import random
from network import Network
import numpy as np


def train_test_split(X, y, train_size):
    if len(X) != len(y):
        raise Exception("inputs and outputs must be the same length")

    # shuffle the data
    for i in reversed(range(len(X))):
        j = random.randint(0, i)
        if i == j: continue

        temp = X[i]
        temp2 = y[i]

        X[i] = X[j]
        X[j] = temp
        y[i] = y[j]
        y[j] = temp2

    # split data
    train_idx = int(len(X) * train_size)
    X_train = X[0:train_idx]
    y_train = y[0:train_idx]
    X_test = X[train_idx:]
    y_test = y[train_idx:]

    return (X_train, y_train, X_test, y_test)


# do one-hot-encoding on labels
def encode_labels(labels):
    labels_id = []
    for label in labels:
        if label not in labels_id:
            labels_id.append(label)

    encoded = []

    for label in labels:
        new_label = map(lambda l: 1 if l == label else 0, labels_id)
        encoded.append(list(new_label))

    return encoded

def check(outputs, expectedOutputs):
    right = 0
    for output, expected in zip(outputs, expectedOutputs):
        if output == expected:
            right+=1 
    return right, len(outputs)


def main():
    data = arff.load(open("datasets/iris.arff"))['data']
    inputs = []
    outputs = []
    for item in data:
        inputs.append(item[:-1])
        outputs.append(item[-1])
    X_train, y_train, X_test, y_test = train_test_split(inputs, outputs, 0.8)

    nn = NeuralNet([4, 3], epochs=20)
    nn.fit(X_train, encode_labels(y_train))

    test_data = []

    for x, y in zip(X_test, encode_labels(y_test)):
        test_data.append((np.reshape(x, (4, 1)), np.reshape(y, (3, 1))))

    print(nn.evaluate(test_data))

    # print(nn.feedfoward(np.reshape([1,2, 3, 4], (4, 1))))

    #
    # nn = NeuralNet([4, 3], epochs=50)
    # nn.fit(X_train, encode_labels(y_train))
    # # print(nn.weights)
    # outputs = nn.predict(X_test, log=True)
    # right, total = check(outputs, encode_labels(y_test))
    # print("{} / {}".format(right, total))
    #


if __name__ == "__main__":
    main()
