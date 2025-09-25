import numpy as np
from . import activations
from . import loss
from . import metrics
import copy
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Any, Callable , Dict
from pandas import DataFrame, Series


class Layer(ABC):
    regularizable_params: Optional[List[str]]

    def __init__(self, has_dims: bool=False, m:Optional[int]=None, n:Optional[int]=None):
        self.regularizable_params = []
        self.m = m
        self.n = n
        self.has_dims = has_dims

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_params_and_grads(self):
        """Return list of (param, grad) tuples"""
        pass

    @abstractmethod
    def set_training(self, training: bool):
        pass

    @abstractmethod
    def update_weights(self, **kwargs):
        pass


def one_hot_encode(y:np.ndarray) -> np.ndarray:
     # use np.eye() to one hot encode a vector
     return np.eye(y.max() + 1)[y]

def train_test_split(X: np.ndarray,
                     y: np.ndarray, 
                     test_size: Optional[Union[int, float]]=None,
                     train_size: Optional[Union[int, float]]=None,
                     shuffle_data: bool=True,
                     random_state=None):
    """ 
    Splitting X,y into train and test sets
    Parameters
    ----------
    X : array
        array of features 
    y : array
        target array
    test_size : float or int
        size of test data. If this is a number in (0,1) we interpret
        as a fraction. If it is an integer we treat it as the size
    train_size : float or int
        size of training data. If this is a number in (0,1) we interpret
        as a fraction. If it is an integer we treat it as the size
    shuffle_data: boolean
        whether we should shuffle the data first before splitting into train and test
        Usually set to True, but could be false if using time-series data

    Returns
    -------
        X_train: array
            array of features for training
        X_test : array
            array of features for testing
        y_train: array
            array of targets for training
        y_test: array
            array of targets for testing
    """
    if len(X) != len(y):
        raise ValueError("number of entries in X and y don't match")
    num_rows = X.shape[0]
    # first shuffle X,y if shuffle_data is True 
    if shuffle_data:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(len(X))
        X_shuffled,y_shuffled = X.copy()[indices], y.copy()[indices]
    else:
        X_shuffled,y_shuffled = X.copy(), y.copy()
    # we prioritize test_size if both test_size and train_size are given
    if test_size is not None:
        if test_size < 0:
            raise ValueError("Test size must be positive")
        # when test_size is between 0 and 1, treat as a ratio
        # multiply the size by the number of entries to get the correct number
        if test_size < 1:
            test_size = (np.floor(test_size * num_rows))
        # make sure the test_size is an integer
        test_size = int(test_size)
        X_train = X_shuffled[:num_rows - test_size]
        X_test = X_shuffled[num_rows - test_size:]
        y_train = y_shuffled[:num_rows - test_size]
        y_test = y_shuffled[num_rows - test_size:]
        return (X_train, X_test, y_train, y_test)
    if train_size is not None:
        if train_size < 0:
            raise ValueError("Train size must be positive")
        # when train_size is between 0 and 1, treat as a ratio
        # multiply the size by the number of entries to get the correct number
        if train_size < 1:
            train_size = (np.floor(train_size * num_rows))
        # make sure train_size is an integer
        train_size = int(train_size)
        X_train = X_shuffled[:train_size]
        X_test = X_shuffled[train_size:]
        y_train = y_shuffled[:train_size]
        y_test = y_shuffled[train_size:]
        return (X_train, X_test, y_train, y_test)
    raise ValueError("Must give a test size or train size")

class BatchNorm(Layer):
    def __init__(self,
                 num_features :int, 
                 gamma: Optional[float]=None, 
                 beta: Optional[float]=None,
                 eps: float=1e-5,
                 momentum: float=0.9):
        super().__init__(has_dims=True, m=num_features, n=num_features)
        # using m,n num_features so it will correctly sit between dense layers
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        

        # Initialize gamma and beta if not provided
        self.gamma = gamma if gamma is not None else np.ones(num_features)
        self.beta = beta if beta is not None else np.zeros(num_features)

        # Running estimates for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.v_gamma = np.zeros_like(self.gamma)   # velocity for Adam
        self.v_beta = np.zeros_like(self.beta)
        self.m_gamma= np.zeros_like(self.beta)   # first moment for Adam/momentum
        self.m_beta = np.zeros_like(self.beta)
        # for consistency across layers
        self.activation = activations.Identity()
        self.activation_name = "identity"
        self.training = True
        
    
    def forward(self, x:np.ndarray) -> np.ndarray:
        if self.training:
            mu = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            std = np.sqrt(var + self.eps)
            x_norm = (x - mu)/std
            self.batch_size = x.shape[0]
            self.centered_x = x - mu
            self.xn = x_norm
            self.mu = mu
            self.var = var
            self.std = np.sqrt(self.var + self.eps)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            x_norm = xc / (np.sqrt(self.running_var + self.eps))
        return self.gamma * x_norm + self.beta
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
         # grads for gamma and beta
        self.grad_beta = np.sum(grad_output, axis=0)
        self.grad_gamma = np.sum(self.xn * grad_output, axis=0)

        # gradient wrt normalized input
        dxn = grad_output * self.gamma

        # number of samples
        N = self.batch_size

        # formula for dx
        dx = (1. / N) * (1. / self.std) * (
            N * dxn
            - np.sum(dxn, axis=0)
            - self.xn * np.sum(dxn * self.xn, axis=0)
        )

        return dx
    
    def get_params_and_grads(self) -> list[tuple[str, np.ndarray]]:
        params_grads = []
        if self.grad_gamma is not None:
            params_grads.append((self.gamma, self.grad_gamma))
        if self.grad_beta is not None:
            params_grads.append((self.beta, self.grad_beta))
        return params_grads
    
    def set_training(self, mode: bool):
        self.training = mode

    def update_weights(self,
                       learning_rate: float,
                       optimizer: str="sgd",
                       t: int=1,
                       beta1:
                       float=0.9,
                       beta2: float=0.999,
                       epsilon: float=1e-8):
        # possible optimizers:
        # sgd : standard gradient descent
        # momentum : momentum optimization
        # adam : adam optimization
        if optimizer == "sgd":
            self.gamma -= learning_rate * self.grad_gamma
            self.beta -= learning_rate * self.grad_beta

        elif optimizer == "momentum":
            self.m_gamma = beta1 * self.m_gamma + (1 - beta1) * self.grad_gamma
            self.m_beta  = beta1 * self.m_beta + (1 - beta1) * self.grad_beta
            self.gamma -= learning_rate * self.m_gamma
            self.beta -= learning_rate * self.m_beta

        elif optimizer == "adam":
            self.m_gamma= beta1 * self.m_gamma + (1 - beta1) * self.grad_gamma
            self.m_beta = beta1 * self.m_beta + (1 - beta1) * self.grad_beta
            self.v_gamma = beta2 * self.v_gamma + (1 - beta2) * (self.grad_gamma ** 2)
            self.v_beta = beta2 * self.v_beta + (1 - beta2) * (self.grad_beta ** 2)

            m_hat_gamma = self.m_gamma / (1 - beta1**t)
            m_hat_beta = self.m_beta / (1 - beta1**t)
            v_hat_gamma = self.v_gamma / (1 - beta2**t)
            v_hat_beta = self.v_beta / (1 - beta2**t)

            self.gamma -= learning_rate * m_hat_gamma / (np.sqrt(v_hat_gamma) + epsilon)
            self.beta -= learning_rate * m_hat_beta / (np.sqrt(v_hat_beta) + epsilon)




class Dense(Layer):
    def __init__(self,
                 num_inputs: int,
                 num_neurons: int,
                 activation: Union[str, activations.Activation] ="identity", 
                 weights: Optional[np.ndarray]=None,
                 bias: Optional[np.ndarray]=None,
                 activation_params:Optional[List]=None):
        """
        Arguments
        ---------
        num_inputs:    int
            number of inputs to the layer
        num_neurons:   int
            number of neurons in the layer
        activation:    string or Activation class
            name of activation function (default is identity)
        weights:       array of size num_inputs x num_neurons
            optional initial weights to use. Randomized if not
            specified
        bias:          float
            bias to be used. Random if not specified
        """
        # make sure values make sense. dimensions should be positive
        if num_inputs <= 0:
            raise ValueError("Number of inputs must be positive")
        if num_neurons <=0:
            raise ValueError("Number of neurons must be positive")
        super().__init__(has_dims=True, m=num_inputs, n=num_neurons)
        activation_params = activation_params or {}
        if isinstance(activation, str):
            act_name= activation.lower()
            if act_name in activations.activation_functions:
                self.activation_name = act_name
                self.activation: activations.Activation = activations.activation_functions[act_name](**activation_params)
            else:
                raise ValueError(f"Activation function {activation} not recognized. Must be one of {list(activations.activation_functions.keys())}")
        elif isinstance(activation, activations.Activation):
            self.activation_name = type(activation).__name__
            self.activation = activation
        else:
            raise TypeError("Activation must be string or Activation")
        if weights is not None:
            self.W = weights
        else:
            if self.activation_name in ["relu", "leaky_relu", "elu", "swish"]:
                self.W = np.random.randn(self.n, self.m) * np.sqrt(2. / self.m)  # He initialization
            elif self.activation_name in ["sigmoid", "tanh"]:
                self.W = np.random.randn(self.n, self.m) * np.sqrt(1. / self.m)  # Xavier/Glorot initialization
            else:
                self.W = np.random.randn(self.n, self.m) * 0.01  # Slightly larger fallback

        if self.W.shape != (self.n, self.m):
            raise ValueError(f"Weights must be of shape ({self.n}, {self.m})")
        if bias is None:
            self.b = np.zeros((1, self.n))
        else:
            if bias.shape == (self.n,) or bias.shape == (1, self.n):
                self.b = bias.reshape(1, self.n)
            else:
                raise ValueError(f"Bias must be of shape ({self.n},) or (1, {self.n})")
        self.v_W = np.zeros_like(self.W)   # velocity for momentum/Adam
        self.v_b = np.zeros_like(self.b)
        self.m_W = np.zeros_like(self.W)   # first moment for Adam
        self.m_b = np.zeros_like(self.b)
        self.regularizable_params = ["W"]
        self.x = None
        self.z = None
        self.grad_W = None
        self.grad_b = None
        self.training = True
       

    def set_training(self, mode: bool):
        self.training = mode
        
    def forward(self, x:np.ndarray) -> np.ndarray:
        self.x = x
        self.z = x @ self.W.T + self.b
        a = self.activation.forward(self.z)
        return a

    def backward(self, grad_output:np.ndarray) -> np.ndarray:
        # grad_output: dL/da
        grad_z = self.activation.backward(self.z) * grad_output  # dL/dz = dL/da * da/dz
        batch_size = grad_z.shape[0]
        # dL/dW = (grad_z.T @ x) / batch_size → shape (n, m)
        self.grad_W = grad_z.T @ self.x / batch_size

        # dL/db = sum over batch / batch size → shape (n, 1)
        self.grad_b = grad_z.sum(axis=0, keepdims=True) / batch_size

        # dL/dx = grad_z @ W → shape (batch_size, m)
        grad_input = grad_z @ self.W

        return grad_input

    def get_params_and_grads(self) -> List[tuple[str , np.ndarray]]:
        # Only include if gradients are computed
        params_grads = []
        if self.grad_W is not None:
            params_grads.append((self.W, self.grad_W))
        if self.grad_b is not None:
            params_grads.append((self.b, self.grad_b))
        return params_grads

    def update_weights(self,
                       learning_rate: float=0.01,
                       optimizer: str="sgd", 
                       t: int=1,
                       beta1: float =0.9,
                       beta2: float =0.999,
                       epsilon: float=1e-8):
        # possible optimizers:
        # sgd : standard gradient descent
        # momentum : momentum optimization
        # adam : adam optimization
        if optimizer == "sgd":
            self.W -= learning_rate * self.grad_W
            self.b -= learning_rate * self.grad_b

        elif optimizer == "momentum":
            self.m_W = beta1 * self.m_W + (1 - beta1) * self.grad_W
            self.m_b = beta1 * self.m_b + (1 - beta1) * self.grad_b
            self.W -= learning_rate * self.m_W
            self.b -= learning_rate * self.m_b

        elif optimizer == "adam":
            self.m_W = beta1 * self.m_W + (1 - beta1) * self.grad_W
            self.m_b = beta1 * self.m_b + (1 - beta1) * self.grad_b
            self.v_W = beta2 * self.v_W + (1 - beta2) * (self.grad_W ** 2)
            self.v_b = beta2 * self.v_b + (1 - beta2) * (self.grad_b ** 2)

            m_hat_W = self.m_W / (1 - beta1**t)
            m_hat_b = self.m_b / (1 - beta1**t)
            v_hat_W = self.v_W / (1 - beta2**t)
            v_hat_b = self.v_b / (1 - beta2**t)

            self.W -= learning_rate * m_hat_W / (np.sqrt(v_hat_W) + epsilon)
            self.b -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)


class Dropout(Layer):
    def __init__(self, drop_prob:float):
        super().__init__()
        self.drop_prob = drop_prob
        self.mask = None
        self.has_dims = False  # Dropout does not have fixed input/output dimensions
        self.training = True  # Default to training mode
        self.activation = activations.Identity()
        self.activation_name = "identity"
        

    def forward(self, x:np.ndarray) -> np.ndarray:
        if self.training:
            # Generate dropout mask
            self.mask = (np.random.rand(*x.shape) > self.drop_prob) / (1.0 - self.drop_prob)
            return x * self.mask
        else:
            # During evaluation, pass values through unchanged
            return x

    def backward(self, grad_output:np.ndarray) -> np.ndarray:
        # Apply dropout mask to gradient
        return grad_output * self.mask
    
    def get_params_and_grads(self) -> List[tuple[str, np.ndarray]]:
        return []  # Dropout has no parameters

    def update_weights(self, **_unused: Any):
        # Dropout has no weights to update
        # **kwargs is there because the feed forward will pass arguments for the dense layer
        pass
    
    def set_training(self, mode: bool):
        self.training = mode

        
class FeedForward:
    # class to combine several layers and pass input to first layer then all the way through
    def __init__(self, 
                 *layers: Layer,
                 training: bool=True):
        self.layers = layers
        self.last_layer = layers[-1]
        self.t = 1  # batch-level counter
        self.training = training
        # validate that dimension match up
        # we have to skip dimensionless layers like Dropout
        last_dim_layer = None
        last_index = None
        for i, layer in enumerate(layers):
            if getattr(layer, "activation_name", "identity") == "softmax" and layer != self.last_layer:
                raise ValueError("Softmax can only be activation in final layer")
            if getattr(layer, 'has_dims', False):
                if last_dim_layer is not None:
                    if layer.m != last_dim_layer.n:
                        raise ValueError(
                            f"Mismatched dimensions: layer {last_index} has {last_dim_layer.n} outputs, "
                            f"layer {i} expects {layer.m} inputs"
                            )
                last_dim_layer = layer
                last_index = i

        self.history = None
    def forward(self, x:np.ndarray) -> np.ndarray:
        # pass x through all the layers
        x = np.array(x)  # ensure x is an array
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def backward(self, grad_output:np.ndarray):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
    def update_weights(self,
                       learning_rate: float = 0.01,
                       optimizer: str="sgd",
                       beta1: float=0.9,
                       beta2: float=0.999,
                       epsilon: float=1e-8,
                       t: Optional[int]=None):
        """
        Apply the weight updates for all layers.
        """
        if t is None:
            t = self.t
        for layer in self.layers:
            layer.update_weights(
                learning_rate=learning_rate,
                optimizer=optimizer,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
                t=t
            )
        

    def __call__(self, x:np.ndarray) -> np.ndarray:
        return self.forward(x)

    def set_training(self, mode:bool=True):
        self.training = mode
        for layer in self.layers:
            if hasattr(layer, 'set_training'):
                layer.set_training(mode)
           

    def fit(
        self,
        X: Union[np.ndarray, List, DataFrame],
        y: Union[np.ndarray, List, DataFrame, Series],
        epochs: int=10,
        loss_fn: Optional[Union[str , loss.Loss ]]=None,
        verbose: bool=False,
        learning_rate: float=0.01,
        metric: Optional[Union[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
        batch_size: int=32,
        print_every: int=10,
        val_size: Optional[Union[int, float]]=None,
        val_set: Optional[tuple[Union[np.ndarray, List, DataFrame], Union[np.ndarray, List, DataFrame, Series]]]=None,
        clip_value: Optional[float]=None,
        random_state: Optional[int]=None,
        optimizer: str="sgd",
        optimizer_config: Optional[Dict]=None,
        lambda_l1: float=0.0,
        lambda_l2: float=0.0
    ):
        """
        Fit model based on X, y.

        Parameters
        ----------
        X : np.ndarray, pd.DataFrame, or list
            Features used to train the model.
        y : np.ndarray, pd.DataFrame, pd.Series, or list
            Target values used to train the model.
        epochs : int
            Number of epochs to use during training.
        loss_fn : str or loss.Loss
            Loss function to use during training. If a string, must be a key in `loss.loss_functions`.
        verbose : bool
            Whether to print intermediate losses during training.
        learning_rate : float
            Learning rate to use for gradient descent.
        metric : str, callable, or list of str or callable
            Metric(s) to evaluate during training. Can be:
             - A single string key referring to a built-in metric in `metrics.metrics_dict`,
             - A single callable taking `(y_pred, y_true)` and returning a scalar,
             - A list containing any combination of strings and callables.
        batch_size : int
            Number of samples per batch.
        print_every : int
            Print training loss and metrics every N batches (only applies if `verbose=True`).
        val_size : int or float
            Size of the validation set. If a float in (0,1), treated as a fraction of the dataset; otherwise, treated as the number of samples.
        val_set : tuple of (array-like, array-like)
            Pair `(X_val, y_val)` to be used for validation.
        clip_value : float, optional
            If provided, gradients are clipped to the range [-clip_value, clip_value].
        random_state : int, optional
            Random state used when shuffling data for train/validation split.
        optimizer : str
            Optimizer to use for gradient descent.
        optimizer_configs : dict, optional
            Dictionary of optional configuration values for the optimizer.
        lambda_l1 : float, optional
            coeffecient for L1 regularization (sum of absolute value of weights)
            Default is 0.0 (no L1 regularization)
        lambda_l2 : float, optional
            coefficient for L2 regularization (sum of squares of weights)
            Default is 0.0 (no L2 regularization)
        
        """

        optimizer = optimizer.lower()
        if optimizer not in ["sgd", "momentum", "adam"]:
            raise ValueError(f"{optimizer} is not a valid optimizer. Choose one of sgd, momentum, or adam")
        # create default configs
        if optimizer_config is None:
            optimizer_config = {
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8
                }
        # set the configurations
        beta1 = optimizer_config.get("beta1", 0.9)
        beta2 = optimizer_config.get("beta2", 0.999)
        epsilon = optimizer_config.get("epsilon", 1e-8)
        # set up rng to use for shuffling data
        rng = np.random.default_rng(random_state)
        # get the number of neurons in the output layer
        output_size = self.layers[-1].n
        final_activation = getattr(self.layers[-1], "activation_name", None)
        if final_activation == "softmax":
            self.from_logits = False
        else:
            self.from_logits = True
        # set up loss function. If none is given, use mse or softmax based on output size
        if loss_fn is None:
            if output_size == 1:
                self.loss_fn = loss.MSE()
                print("No loss function specified, using mse, warning: this may not be appropriate for classification")
            else:   
                self.loss_fn = loss.SoftmaxCrossEntropyLoss(from_logits=self.from_logits)
                print("No loss function specified, using categorical_cross_entropy, warning: this may not be appropriate for regression")
        elif isinstance(loss_fn, str):
            #if a string is given, look up in loss_functions
            if loss_fn in loss.loss_functions: 
                self.loss_fn = loss.loss_functions[loss_fn](self.from_logits)
            else:
                raise ValueError(f"Loss function {loss_fn} not recognized. Must be a Loss class or one of {list(loss.loss_functions.keys())}")
        elif isinstance(loss_fn, loss.Loss):
            self.loss_fn = loss_fn
        else:
            # if not a string or loss function,  raise error
            raise ValueError("Loss must be a Loss class or a string key in loss_functions")
        if metric is None:
            # if no metric is given, use mse for regression and accuracy for classification
            if output_size == 1:
                metric = [metrics.mse]
            else:   
                metric = [metrics.accuracy]
        metric_names = []
        if not isinstance(metric, list):
            metric_list = [metric]
        else:
            metric_list = metric
        for (i,metric) in enumerate(metric_list):
            if isinstance(metric, str):
             # if a string is given, look up in metrics_dict
                if metric in metrics.metrics_dict:
                    metric = metrics.metrics_dict[metric]
                    metric_list[i] = metric
                else:
                    raise ValueError(f"Metric {metric} not recognized. Must be a callable or one of {list(metrics.metrics_dict.keys())}")
            elif not callable(metric):
                raise ValueError("Metric must be a callable or a string key in metrics_dict")
            # add the name of the metric for history tracking
            metric_names.append(metric.__name__)
        metric_dict = dict(zip(metric_names, metric_list))
        # create flag to see if validation set exists
        has_val = val_size is not None or val_set is not None
        # create history if it doesn't exist yet 
        if self.history is None:
            train_history = {metric_name:[] for metric_name in metric_names}
            train_history["loss"] = []
            val_history={}
            if has_val:
                val_history = {f"val_{metric_name}":[] for metric_name in metric_names}
                val_history["val_loss"] = []
            history = {**train_history, **val_history}
            self.history = history
            # no previous training occurred so the history can start with the epoch count
            last_epoch = 0
        else:
            history = self.history
            if has_val:
                # if there was not a previous validation set add val_loss
                if "val_loss" not in history:
                    history["val_loss"] = []
            # add history keys for any new metrics 
            for metric_name in metric_dict:
                if metric_name not in history:
                    history[metric_name] = []
                if has_val and f"val_{metric_name}" not in history:
                    # if we have validation set and missing val_metric add the key to history
                    history[f"val_{metric_name}"] = []
            # we want to increase the epochs as we go
            last_epoch = history["loss"][-1][0]
        # make sure X,y are arrays for training. 
        # if they are already arrays, make a copy to avoid modifying the original data    
        X_train = np.array(X)
        y_train = np.array(y)
        
        
        self.set_training(True) # make sure we're in training mode to fit (mostly for dropout)
        # deal with validation set
        # if a validation set is given separately, use X, y to train
        if val_set is not None:
            X_val, y_val = np.array(val_set[0]), np.array(val_set[1]) 
            use_validate = True
        # if no validation set is given, but a size is given, use train_test_split
        # to split into training and validation sets
        elif val_size is not None:
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=val_size,
                random_state=random_state
                )
            use_validate = True
        # if nothing about validation is given use X, y as training data
        else:
            use_validate = False
        # make sure all y are the correct shape
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1,1)
        if use_validate and y_val.ndim == 1:
            y_val = y_val.reshape(-1,1)
        # loop through epochs
        num_samples=X_train.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        for epoch in range(1, epochs+1):
            # shuffle X so that we aren't using the same batches every time
            
            permutation = rng.permutation(num_samples)
            # use one permutation so we match X and y properly
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]
            # variables to track loss/accuracy during training
            epoch_loss = 0
            # loop through the batches
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, num_samples)
                # get current batch
                X_batch = X_shuffled[start : end]
                y_batch = y_shuffled[start : end]
                # get output from model

                y_pred = self.forward(X_batch)
                # compute loss
                
                loss_batch = self.loss_fn.forward(y_pred, y_batch)
                # add regularization loss if it exists
                if lambda_l1 > 0 or lambda_l2 >0:
                    reg_loss = 0.0
                    for layer in self.layers:
                        for attr in layer.regularizable_params:
                            param = getattr(layer,attr)
                            if lambda_l1 > 0:
                                reg_loss += (lambda_l1 * np.sum(np.abs(param)))/(end - start)
                            if lambda_l2 > 0:
                                reg_loss += (0.5 * lambda_l2 * np.sum(param**2))/(end - start)
                    loss_batch += reg_loss

                epoch_loss += loss_batch * (end - start)  # weighted sum for averaging later

                
                
                # backward pass
                loss_grad = self.loss_fn.backward(y_pred, y_batch)
                self.backward(loss_grad)
                # adjust gradients if there is regularization.
                for layer in self.layers:
                    for attr in layer.regularizable_params:
                        param = getattr(layer, attr)
                        grad = getattr(layer, "grad_" + attr)
                        if lambda_l1 > 0:
                            grad += lambda_l1 * np.sign(param) / (end - start)
                        if lambda_l2 > 0:
                            grad +=  lambda_l2 * param /(end - start)
                # clip gradients if requested
                if clip_value is not None:
                    for layer in self.layers:
                        for _, grad in layer.get_params_and_grads():
                            np.clip(grad, -clip_value, clip_value, out=grad)

                # update weights
                self.update_weights(
                    learning_rate=learning_rate,
                    optimizer=optimizer,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon,
                    t=self.t
                    )
                self.t+=1

                # show output if verbose is set to True based on print_every
                if verbose:
                    # Optional: print loss every print_every batches
                    if batch_idx % print_every == 0:
                        running_ave_loss = epoch_loss/end
                        print(f"Epoch {epoch}/{epochs}, Batch {batch_idx+1}/{num_batches}, Loss: {running_ave_loss:.4f}")
            # at the end of the epoch print out the epoch loss/metric
            train_metric_values ={name: func(self.predict(X_train), y_train) for name, func in metric_dict.items()}
            for name, val in train_metric_values.items():
                history[name].append((epoch+last_epoch,val))
            epoch_loss /= num_samples
            history["loss"].append((epoch+last_epoch,epoch_loss))
            train_metrics_str = ", ".join(f"{name}: {val:.4f}" for name, val in train_metric_values.items())
            print(f"Epoch {epoch}/{epochs}: loss: {epoch_loss:.4f}, {train_metrics_str}", end = " ")
            # if no validation set we stop
            if not use_validate:
                print("")
            #if there is we compute the loss/accuracy on the validation set from that epoch and report it
            else:
                y_test_pred = self.predict(X_val)
                val_metric_values = {name : func(y_test_pred, y_val) for name, func in metric_dict.items()}
                for name, val in val_metric_values.items():
                    history[f"val_{name}"].append((epoch + last_epoch,val))
                val_loss = self.loss_fn.forward(y_test_pred, y_val)
                if lambda_l1 > 0 or lambda_l2 > 0:
                    val_reg_loss = 0.0
                    for layer in self.layers:
                        for attr in layer.regularizable_params:
                            param = getattr(layer, attr)
                            if lambda_l1 > 0:
                                val_reg_loss += lambda_l1 * np.sum(np.abs(param)) / X_val.shape[0]
                            if lambda_l2 > 0:
                                val_reg_loss += 0.5 * lambda_l2 * np.sum(param ** 2) / X_val.shape[0]
                    val_loss += val_reg_loss
                history["val_loss"].append((epoch+last_epoch,val_loss))
                val_metrics_str = ", ".join(f"{name}: {val:.4f}" for name, val in val_metric_values.items())
                print(f"- Validation set: loss: {val_loss:.4f}, {val_metrics_str}")

        self.history = history
        return copy.deepcopy(history)

    def plot_history(self, *keys: str):
        if self.history is None:
            print("No training has occured yet")
        else:
            plot_history(self.history, *keys)
            
    def predict(self, X: Union[DataFrame, np.ndarray, List]):
        """
        Predict target values using the trained FeedForward model.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame, shape (n_samples, n_features)
            Features for which to predict target values.

        Returns
        -------
        np.ndarray, shape (n_samples, 1) or (n_samples, n_outputs)
            Predicted values for each input sample. For models with a single output neuron, returns a 2D array with shape (n_samples, 1).
            For multi-output or multi-label models, returns a 2D array with one column per output.

        Notes
        -----
        - The model must be trained (via `.fit()`) before calling `.predict()`.
        - Input features are assumed to be in the same order and scaling as used during training.
        """

        # save current training state
        current_training = self.training
        # propagate to layers
        self.set_training(False)
        out = self.forward(X)
        # restore previous training state
        self.set_training(current_training)
        return out
    
   
def plot_history(history, *keys: str):
    """
    function to plot from history generated from a .fit()
    Parameters
    ----------
    history : Dict(List)
        a dictionary where the key is the name of the metric
        and the value is the set of ordered pairs (epoch, metric)
    keys : strings
        strings for which keys we want to plot
    """
    legend = []
    for key in keys:
        # add the name of the metric to the legend
        legend.append(key)
        # separate out epoch, metric into x,y
        x,y = zip(*history[key])
        # plot the history for this metric
        plt.plot(x,y)
    # add the legend
    plt.legend(legend)
    plt.show()



def evaluate(model: FeedForward, X: Union[DataFrame, np.ndarray, List]):
    model.set_training(False)
    return model(X)
