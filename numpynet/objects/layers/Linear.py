"""This module defines the fully-connected linear layer.
"""

import numpy as np
np.random.seed(0)

class Linear():
    """Fully connected layer.
    
    Args:
        in_features (int): The number of features in the input vector.
        out_features (int): The number of features in the output vector.
    
    Attributes:
        in_features (int): The number of features in the input vector.
        out_features (int): The number of features in the output vector.
        self.weight (np.ndarray): The matrix with the weights of the layer, with shape
            (out_features, in_features).
        self.bias (np.ndarray): The vector with the bias of the layer, with shape
            (out_features, 1).
    
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = self._is_valid_in_features(in_features)
        self.out_features = self._is_valid_out_features(out_features)
        self.weight = np.random.uniform(size=(out_features,in_features))
        self.bias = np.random.uniform(size=(out_features, 1))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = self._is_valid_input(x)
        return np.matmul(self.weight, x) + self.bias

    def set_weight(self, weight: np.ndarray) -> None:
        """Set an arbitrary weight matrix for the layer.
        
        Args:
            weight (np.ndarray): The matrix with shape (out_features, in_features).
        
        """
        self.weight = self._is_valid_weight(weight)

    def set_bias(self, bias: np.ndarray) -> None:
        """Set an arbitrary bias vector for the layer.
        
        Args:
            bias (np.ndarray): The vector with shape (out_features, 1)
        
        """
        self.bias = self._is_valid_bias(bias)

    def _is_valid_weight(self, weight) -> np.ndarray:
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError
        return weight

    def _is_valid_bias(self, bias) -> np.ndarray:
        if bias.shape != (self.out_features, 1):
            raise ValueError
        return bias

    def _is_valid_in_features(self, in_features):
        if not isinstance(in_features, int):
            raise TypeError
        return in_features

    def _is_valid_out_features(self, out_features):
        if not isinstance(out_features, int):
            raise TypeError
        return out_features

    def _is_valid_input(self, x: np.ndarray) -> np.ndarray:
        if x.shape != (self.in_features, 1):
            raise ValueError
        return x
