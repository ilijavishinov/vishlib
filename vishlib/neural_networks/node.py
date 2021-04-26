from typing import List, Union
import activation_functions
from abc import ABC


class Node(ABC):
    pass


class NetworkNode(Node):

    def __init__(self, nodes_before: List[__class__],
                 activation_function: Union[str, activation_functions.ActivationFunction]):

        super().__init__()
        self.nodes_before = nodes_before
        self.net_sum = None
        self.activation_function = None
        if activation_function is str:
            self.activation_function = activation_functions.map_string_to_activation_function(activation_function)
        if activation_function is activation_functions.ActivationFunction:
            self.activation_function = activation_function
        self.output = self.calculate_output()

    def calculate_output(self):

        self.net_sum = 0
        for node in self.nodes_before:
            self.net_sum += node.output
        return self.activation_function.apply_activation_function(self.net_sum)


class InputNode(Node):

    def __init__(self, value: Union[int, float]):

        super().__init__()
        self.output = value



