# coding: utf-8


import sys
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# # Machine Learning with PyTorch and Scikit-Learn
# # -- Code Examples

# ## Package version checks

# Add folder to path in order to load from the check_packages.py script:


sys.path.insert(0, "..")


# Check recommended package versions:


# # Chapter 11 - Implementing a Multi-layer Artificial Neural Network from Scratch
#

# ### Overview

# - [Modeling complex functions with artificial neural networks](#Modeling-complex-functions-with-artificial-neural-networks)
#   - [Single-layer neural network recap](#Single-layer-neural-network-recap)
#   - [Introducing the multi-layer neural network architecture](#Introducing-the-multi-layer-neural-network-architecture)
#   - [Activating a neural network via forward propagation](#Activating-a-neural-network-via-forward-propagation)
# - [Classifying handwritten digits](#Classifying-handwritten-digits)
#   - [Obtaining the MNIST dataset](#Obtaining-the-MNIST-dataset)
#   - [Implementing a multi-layer perceptron](#Implementing-a-multi-layer-perceptron)
#   - [Coding the neural network training loop](#Coding-the-neural-network-training-loop)
#   - [Evaluating the neural network performance](#Evaluating-the-neural-network-performance)
# - [Training an artificial neural network](#Training-an-artificial-neural-network)
#   - [Computing the loss function](#Computing-the-loss-function)
#   - [Developing your intuition for backpropagation](#Developing-your-intuition-for-backpropagation)
#   - [Training neural networks via backpropagation](#Training-neural-networks-via-backpropagation)
# - [Convergence in neural networks](#Convergence-in-neural-networks)
# - [Summary](#Summary)


# # Modeling complex functions with artificial neural networks

# ...

# ## Single-layer neural network recap


# ## Introducing the multi-layer neural network architecture


# ## Activating a neural network via forward propagation


# # Classifying handwritten digits

# ...

# ## Obtaining and preparing the MNIST dataset

# The MNIST dataset is publicly available at http://yann.lecun.com/exdb/mnist/ and consists of the following four parts:
#
# - Training set images: train-images-idx3-ubyte.gz (9.9 MB, 47 MB unzipped, 60,000 examples)
# - Training set labels: train-labels-idx1-ubyte.gz (29 KB, 60 KB unzipped, 60,000 labels)
# - Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 7.8 MB, 10,000 examples)
# - Test set labels: t10k-labels-idx1-ubyte.gz (5 KB, 10 KB unzipped, 10,000 labels)
#
#

#
# X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
# X = X.values
# y = y.astype(int).values
#
#
# # Normalize to [-1, 1] range:
# X = ((X / 255.0) - 0.5) * 2
#
# X_temp, X_test, y_temp, y_test = train_test_split(
#     X, y, test_size=10000, random_state=123, stratify=y
# )
#
# X_train, X_valid, y_train, y_valid = train_test_split(
#     X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp
# )
#
#
# # optional to free up some memory by deleting non-used arrays:
# del X_temp, y_temp, X, y

s = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(s, header=None, encoding="utf-8")
# select setosa and versicolor
# y = df.iloc[0:100, 4].values
# y = np.where(y == "Iris-setosa", 0, 1)

# # extract sepal length and petal length
# X = df.iloc[0:100, [0, 2]].values

y = df.iloc[0:150, 4].values
X = df.iloc[0:150, :4].values
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)


# ## Implementing a multi-layer perceptron


##########################
### MODEL
##########################


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def int_to_onehot(labels):
    labels_id = []
    for label in labels:
        if label not in labels_id:
            labels_id.append(label)

    encoded = []

    for label in labels:
        new_label = map(lambda l: 1 if l == label else 0, labels_id)
        encoded.append(list(new_label))

    return np.array(encoded)

    # ary = np.zeros((y.shape[0], num_labels))
    # for i, val in enumerate(y):
    #     ary[i, val] = 1
    #
    # return ary


class NeuralNetMLP:

    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()

        self.num_classes = num_classes

        # hidden
        rng = np.random.RandomState(random_seed)

        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        # output
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    def forward(self, x):
        # Hidden layer
        # input dim: [n_examples, n_features] dot [n_hidden, n_features].T
        # output dim: [n_examples, n_hidden]
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # Output layer
        # input dim: [n_examples, n_hidden] dot [n_classes, n_hidden].T
        # output dim: [n_examples, n_classes]
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        return a_h, a_out

    def backward(self, x, a_h, a_out, y):

        #########################
        ### Output layer weights
        #########################

        # onehot encoding
        y_onehot = y

        # Part 1: dLoss/dOutWeights
        ## = dLoss/dOutAct * dOutAct/dOutNet * dOutNet/dOutWeight
        ## where DeltaOut = dLoss/dOutAct * dOutAct/dOutNet
        ## for convenient re-use

        # input/output dim: [n_examples, n_classes]
        d_loss__d_a_out = 2.0 * (a_out - y_onehot) / y.shape[0]

        # input/output dim: [n_examples, n_classes]
        d_a_out__d_z_out = a_out * (1.0 - a_out)  # sigmoid derivative

        # output dim: [n_examples, n_classes]
        delta_out = d_loss__d_a_out * d_a_out__d_z_out  # "delta (rule) placeholder"

        # gradient for output weights

        # [n_examples, n_hidden]
        d_z_out__dw_out = a_h

        # input dim: [n_classes, n_examples] dot [n_examples, n_hidden]
        # output dim: [n_classes, n_hidden]
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)

        #################################
        # Part 2: dLoss/dHiddenWeights
        ## = DeltaOut * dOutNet/dHiddenAct * dHiddenAct/dHiddenNet * dHiddenNet/dWeight

        # [n_classes, n_hidden]
        d_z_out__a_h = self.weight_out

        # output dim: [n_examples, n_hidden]
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)

        # [n_examples, n_hidden]
        d_a_h__d_z_h = a_h * (1.0 - a_h)  # sigmoid derivative

        # [n_examples, n_features]
        d_z_h__d_w_h = x

        # output dim: [n_hidden, n_features]
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return (d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h)


# model = NeuralNetMLP(num_features=28 * 28, num_hidden=50, num_classes=10)
model = NeuralNetMLP(num_features=4, num_hidden=3, num_classes=3)


# ## Coding the neural network training loop

# Defining data loaders:


num_epochs = 100
minibatch_size = 15


def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    # np.random.shuffle(indices)
    y_ = int_to_onehot(y)

    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx : start_idx + minibatch_size]

        yield X[batch_idx], y_[batch_idx]


# iterate over training epochs
for i in range(num_epochs):

    # iterate over minibatches
    minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)

    for X_train_mini, y_train_mini in minibatch_gen:

        break

    break

# Defining a function to compute the loss and accuracy


def mse_loss(targets, probas):
    onehot_targets = int_to_onehot(targets)
    return np.mean((onehot_targets - probas) ** 2)


def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets)


_, probas = model.forward(X_valid)
mse = mse_loss(y_valid, probas)

predicted_labels = np.argmax(probas, axis=1)
acc = accuracy(y_valid, predicted_labels)

print(f"Initial validation MSE: {mse:.1f}")
print(f"Initial validation accuracy: {acc*100:.1f}%")


def compute_mse_and_acc(nnet, X, y, minibatch_size=10):
    mse, correct_pred, num_examples = 0.0, 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)

    for i, (features, targets) in enumerate(minibatch_gen):
        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)

        onehot_targets = targets
        loss = np.mean((onehot_targets - probas) ** 2)
        targets_ = []
        for t in targets:
            targets_.append(np.argmax(t))
        correct_pred += (predicted_labels == targets_).sum()
        # print(predicted_labels)
        # print(targets_)
        # print(int_to_onehot(targets))

        num_examples += targets.shape[0]
        mse += loss
        mse = mse / (i + 1)

    acc = correct_pred / num_examples
    return mse, acc


mse, acc = compute_mse_and_acc(model, X_valid, y_valid)
print(f"Initial valid MSE: {mse:.1f}")
print(f"Initial valid accuracy: {acc*100:.1f}%")


def train(model, X_train, y_train, X_valid, y_valid, num_epochs, learning_rate=0.1):

    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for e in range(num_epochs):

        # iterate over minibatches
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)

        for X_train_mini, y_train_mini in minibatch_gen:
            #### Compute outputs ####
            a_h, a_out = model.forward(X_train_mini)

            #### Compute gradients ####
            d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h = (
                model.backward(X_train_mini, a_h, a_out, y_train_mini)
            )

            #### Update weights ####
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out

        #### Epoch Logging ####
        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)
        train_acc, valid_acc = train_acc * 100, valid_acc * 100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)
        print(
            f"Epoch: {e+1:03d}/{num_epochs:03d} "
            f"| Train MSE: {train_mse:.2f} "
            f"| Train Acc: {train_acc:.2f}% "
            f"| Valid Acc: {valid_acc:.2f}%"
        )

    return epoch_loss, epoch_train_acc, epoch_valid_acc


np.random.seed(123)  # for the training set shuffling

epoch_loss, epoch_train_acc, epoch_valid_acc = train(
    model, X_train, y_train, X_valid, y_valid, num_epochs=num_epochs, learning_rate=0.1
)
