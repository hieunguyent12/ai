from canvas import run
from neuralnetwork.network import Network, Layer, Neuron
from neuralnetwork.cost import Cost, CostType
from neuralnetwork.activation import Activation, ActivationType
from neuralnetwork.data_loader import load_digits


def main():

    cost = Cost.init(CostType.MSE)
    activation = Activation.init(ActivationType.SIGMOID)

    nn = Network(cost, activation)
    nn.createLayers([28 * 28, 10, 10])
    model_path = "model.pkl"
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
    # print((nn.evaluate(list(zip(test_inputs, test_results))) / len(test_results)) * 100)
    # nn.save_model(model_path)

    run(nn)


if __name__ == "__main__":
    main()
