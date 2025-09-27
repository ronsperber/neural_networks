"""
module with activations functions for neural networks
"""

from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):

    """
    Base class for activation functions
    """

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
            z: np.ndarray
        Returns
        -------
        np.ndarray
            the forward method applied to z
        """

    @abstractmethod
    def backward(self, z: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
            z: np.ndarray
        Returns
        -------
        np.ndarray
            the backward method applied to z
        """

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.forward(z)


class ReLU(Activation):
    """
    ReLU activation
    """
    def forward(self, z):
        return np.maximum(0, z)

    def backward(self, z):
        return (z > 0).astype(float)


class Sigmoid(Activation):
    """
    Sigmoid activation
    """
    def forward(self, z):
        return 1 / (1 + np.exp(-z))

    def backward(self, z):
        s = self.forward(z)
        return s * (1 - s)


class Tanh(Activation):
    """
    tanh activation function
    """
    def forward(self, z):
        return np.tanh(z)

    def backward(self, z):
        t = np.tanh(z)
        return 1 - t**2


class Identity(Activation):
    """
    Linear activation function
    """
    def forward(self, z):
        return z

    def backward(self, z):
        return 1.0


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    computes sofmax(logits)
    """
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_logits)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


class SoftMax(Activation):
    """
    Softmax activation
    """
    def forward(self, z):
        return softmax(z)

    def backward(self, z):
        return 1.0


class LeakyReLU(Activation):
    """
    Leaky RelU activation
    """
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, z):
        return np.where(z > 0, z, self.alpha * z)

    def backward(self, z):
        dz = np.ones_like(z)
        dz[z < 0] = self.alpha
        return dz


class ELU(Activation):
    """
    ELU Activation
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def forward(self, z):
        return np.where(z >= 0, z, self.alpha * (np.exp(z) - 1))

    def backward(self, z):
        dz = np.where(z >= 0, 1, self.forward(z) + self.alpha)
        return dz


class Swish(Activation):
    """
    Swish activation
    """
    def forward(self, z):
        return z / (1 + np.exp(-z))

    def backward(self, z):
        s = self.forward(z)/z
        return s + (1 - s) * s * z


activation_functions = {
    "relu": lambda **kwargs: ReLU(),
    "sigmoid": lambda **kwargs: Sigmoid(),
    "tanh": lambda **kwargs: Tanh(),
    "identity": lambda **kwargs: Identity(),
    "linear": lambda **kwargs: Identity(),
    "leaky_relu": lambda alpha=0.01, **kwargs: LeakyReLU(alpha=alpha),
    "elu": lambda alpha=1.0, **kwargs: ELU(alpha=alpha),
    "swish": lambda **kwargs: Swish(),
    "silu": lambda **kwargs: Swish(),
    "softmax": lambda **kwargs: SoftMax()
}

# aliases for when the same function has multiple keys
activation_aliases = {
    "relu": "relu",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "identity": "identity",
    "linear": "identity",
    "leaky_relu": "leaky_relu",
    "elu": "elu",
    "swish": "silu",
    "silu": "silu",
    "softmax": "softmax"
}