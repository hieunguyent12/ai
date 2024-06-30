from enum import Enum, auto


class CostType(Enum):
    MSE = auto()


class Cost:
    def __init__(self):
        pass

    @classmethod
    def init(cls, costType):
        match costType:
            case CostType.MSE:
                return MSE()

    def cost_function(self, predictionOutputs, expectedOutputs):
        raise NotImplementedError("Subclass must implement this method")

    def derivative(self, predictionOutputs, expectedOutputs):
        raise NotImplementedError("Subclass must implement this method")


class MSE(Cost):
    def __init__(self):
        super().__init__()

    def cost_function(self, predictionOutputs, expectedOutputs):
        error = (expectedOutputs - predictionOutputs) ** 2
        return error.sum(axis=1).sum() / expectedOutputs.shape[0]

    def cost_prime(self, predictionOutputs, expectedOutputs):
        return predictionOutputs - expectedOutputs
