from enum import Enum, auto
import numpy as np


class ActivationType(Enum):
    SIGMOID = auto()


class Activation:
    def __init__(self):
        pass

    @classmethod
    def init(cls, activationType):
        match activationType:
            case ActivationType.SIGMOID:
                return Sigmoid()

    def activation(self, z):
        raise NotImplementedError("Subclass must implement this method")

    def activation_prime(self, z):
        raise NotImplementedError("Subclass must implement this method")


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def activation(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def activation_prime(self, z):
        return self.activation(z) * (1 - self.activation(z))
