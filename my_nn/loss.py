from abc import ABC, abstractmethod
import numpy as np

# Base class for loss functions
class Loss(ABC):
    @abstractmethod
    def forward(self, y_pred, y_true):
        pass

    @abstractmethod
    def backward(self, y_pred, y_true):
        pass

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)
    
class SoftmaxCrossEntropyLoss(Loss):
    def forward(self, logits, y_true, debug=False):
        """
        logits: shape (batch_size, num_classes), raw scores (no softmax yet)
        y_true: shape (batch_size,), integer class labels (0 to num_classes-1)
        debug: if True, print debug information
        """
        self.batch_size = logits.shape[0]

        # Safety check
        if logits.shape[1] <= np.max(y_true):
            raise ValueError(
                f"SoftmaxCrossEntropyLoss: logits have shape {logits.shape}, "
                f"but y_true has labels up to {np.max(y_true)}. "
                "Final layer probably has wrong number of outputs."
            )
        y_true =np.asarray(y_true).ravel()  # Ensure y_true is a 1D array
        # Numerical stability
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_logits)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Debug prints
        if debug:
            print("logits min/max/mean:", logits.min(), logits.max(), logits.mean())
            print("shifted_logits min/max/mean:", shifted_logits.min(), shifted_logits.max(), shifted_logits.mean())
            print("probs min/max/mean:", self.probs.min(), self.probs.max(), self.probs.mean())
            print("logits shape:", logits.shape)
            print("y_true shape:", y_true.shape)
            print("np.max(y_true):", np.max(y_true))
            print("self.probs shape before indexing:", self.probs.shape)


        # Cross-entropy loss
        correct_logprobs = -np.log(self.probs[np.arange(self.batch_size), y_true] + 1e-15)

        if debug:
            print("per-example loss min/max/mean:", correct_logprobs.min(),
                  correct_logprobs.max(), correct_logprobs.mean())
            print("logits.shape[0]:", logits.shape[0])
            print("self.batch_size:", self.batch_size)
            print("len(correct_logprobs):", len(correct_logprobs))


        loss = np.sum(correct_logprobs) / self.batch_size
        return loss


    def backward(self, logits, y_true):
        """
        Gradient of loss w.r.t logits
        """
        y_true = np.asarray(y_true).ravel()  # Ensure y_true is a 1D array
        grad = self.probs.copy()
        grad[np.arange(self.batch_size), y_true] -= 1
        grad /= self.batch_size
        return grad

class BinaryCrossEntropyLoss(Loss):
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
    def forward(self, y_pred, y_true):
        loss_sum = (y_pred - y_true) ** 2
        return loss_sum.mean()
    def backward(self, y_pred, y_true):
        loss_der = 2 * (y_pred - y_true)
        return loss_der / y_pred.size
    
loss_functions = {
    "binary_cross_entropy" : BinaryCrossEntropyLoss(),
    "categorical_cross_entropy" : SoftmaxCrossEntropyLoss(),
    "mse" : MSE()
}
