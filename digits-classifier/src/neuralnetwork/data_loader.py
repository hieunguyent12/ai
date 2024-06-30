import arff
import numpy as np
import gzip
import pickle


def iris_labels_to_onehot(labels):
    labels_id = []
    for label in labels:
        if label not in labels_id:
            labels_id.append(label)

    encoded = []

    for label in labels:
        new_label = map(lambda l: 1 if l == label else 0, labels_id)
        encoded.append(list(new_label))

    return encoded


def int_to_onehot(i):
    e = np.zeros(10)
    e[int(i)] = 1.0
    return e


def load_iris():
    data = arff.load(open("datasets/iris.arff"))["data"]
    inputs = []
    outputs = []
    for item in data:
        inputs.append(item[:-1])
        outputs.append(item[-1])
    inputs = np.array(inputs)
    outputs = np.array(iris_labels_to_onehot(outputs))
    return inputs, outputs


def load_digits():
    f = gzip.open("datasets/mnist.pkl.gz", "rb")
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()

    training_inputs = np.array(training_data[0])
    training_results = np.array([int_to_onehot(y) for y in training_data[1]])
    test_inputs = np.array(test_data[0])
    test_results = np.array([int_to_onehot(y) for y in test_data[1]])
    return (training_inputs, training_results, test_inputs, test_results)
