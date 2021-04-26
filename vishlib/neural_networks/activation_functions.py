from abc import abstractmethod
import math
from abc import ABC


class ActivationFunction(ABC):
    @abstractmethod
    def apply_activation_function(self, x):
        pass


class Sigmoid(ActivationFunction):
    def __init__(self):
        self.function = lambda x: 1 / (1 + math.e ** (-x))

    def apply_activation_function(self, x) -> float:
        return self.function(x)


class Relu(ActivationFunction):
    def __init__(self):
        self.function = lambda x: max(0, x)

    def apply_activation_function(self, x) -> float:
        return self.function(x)

# class Softmax(ActivationFunction):


def map_string_to_activation_function(activation_function_string: str) -> ActivationFunction:
    if activation_function_string.lower().startswith('sig'):
        return Sigmoid()




