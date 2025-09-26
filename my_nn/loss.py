"""
Module of loss functions for neural networks
"""
from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    """
    Loss function abstract class
    """
    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        compute the forward pass
        """

    @abstractmethod
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        compute the forward pass
        """

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.forward(y_pred, y_true)


class SoftmaxCrossEntropyLoss(Loss):
    """
    Loss function combining Softmax and categorical
    cross entropy
    """
    def __init__(self, from_logits=True):
        self.from_logits = from_logits
        self.batch_size = None # for use later
        self.probs = None # for use later

    def forward(self, y_pred, y_true):
        """
        y_pred: shape (batch_size, num_classes),
            raw scores or softmax (depending on from_logits)
        y_true: shape (batch_size,), integer class labels (0 to num_classes-1)
        """
        self.batch_size = y_pred.shape[0]

        # Safety check
        if y_pred.shape[1] <= np.max(y_true):
            raise ValueError(
                f"SoftmaxCrossEntropyLoss: logits have shape {y_pred.shape}, "
                f"but y_true has labels up to {np.max(y_true)}. "
                "Final layer probably has wrong number of outputs."
            )
        y_true = np.asarray(y_true).ravel()  # Ensure y_true is a 1D array
        if self.from_logits:
            # Numerical stability
            shifted_logits = y_pred - np.max(y_pred, axis=1, keepdims=True)
            exp_scores = np.exp(shifted_logits)
            self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        else:
            self.probs = y_pred

        # Cross-entropy loss
        correct_logprobs = -np.log(
            self.probs[np.arange(self.batch_size), y_true] + 1e-15
            )
        loss = np.sum(correct_logprobs) / self.batch_size
        return loss

    def backward(self, y_pred, y_true):
        """
        Gradient of loss w.r.t logits
        """
        y_true = np.asarray(y_true).ravel()  # Ensure y_true is a 1D array
        grad = self.probs.copy()
        grad[np.arange(self.batch_size), y_true] -= 1
        grad /= self.batch_size
        return grad


class BinaryCrossEntropyLoss(Loss):
    """
    Binary Cross Entropy Loss
    """
    def forward(self, y_pred, y_true):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss_sum = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        return -np.mean(loss_sum)

    def backward(self, y_pred, y_true):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss_der = - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
        return loss_der / y_pred.shape[0]


class MSE(Loss):
    """
    Mean Square Error Loss
    """
    def forward(self, y_pred, y_true):
        loss_sum = (y_pred - y_true) ** 2
        return loss_sum.mean()

    def backward(self, y_pred, y_true):
        loss_der = 2 * (y_pred - y_true)
        return loss_der / y_pred.size


loss_functions = {
    "binary_cross_entropy": lambda from_logits=True: BinaryCrossEntropyLoss(),
    "categorical_cross_entropy": (
        lambda from_logits=True: SoftmaxCrossEntropyLoss(from_logits)
    ),
    "mse": lambda from_logits=True: MSE()
}
