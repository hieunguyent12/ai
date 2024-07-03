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

    run(nn)


if __name__ == "__main__":
    main()
