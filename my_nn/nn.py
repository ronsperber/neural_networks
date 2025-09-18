import numpy as np
from . import activations
from . import loss
from . import metrics
import copy
import matplotlib.pyplot as plt

def one_hot_encode(y):
     # use np.eye() to one hot encode a vector
     return np.eye(y.max() + 1)[y]

def train_test_split(X,y, test_size=None, train_size=None, shuffle_data=True, random_state=None):
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

class Dense:
    def __init__(self, num_inputs:int, num_neurons:int, activation="identity", weights=None, bias = None, activation_params = None):
        """
        simple dense layer for feed-forward
        will need to eventually add code for backpropogation
        Arguments
        ---------
        num_inputs:    int
            number of inputs to the layer
        num_neurons:   int
            number of neurons in the layer
        activation:    string
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
        self.m = num_inputs
        self.n = num_neurons
        self.has_dims = True
        if weights is not None:
            self.W = weights
        else:
            if activation in ["relu", "leaky_relu", "elu", "swish"]:
                self.W = np.random.randn(self.n, self.m) * np.sqrt(2. / self.m)  # He initialization
            elif activation in ["sigmoid", "tanh"]:
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
        self.activation_params = activation_params or {}
        if activation not in activations.activation_functions:
            raise ValueError(f"Activation function {activation} not recognized. Must be one of {list(activations.activation_functions.keys())}")
        self.activation = activations.activation_functions[activation](**self.activation_params)
        self.x = None
        self.z = None
        self.grad_W = None
        self.grad_b = None
        self.training = True
       

    def set_training(self, mode: bool):
        self.training = mode
        
    def forward(self, x):
        self.x = x
        self.z = x @ self.W.T + self.b
        a = self.activation.forward(self.z)
        return a

    def backward(self, grad_output):
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

    def get_params_and_grads(self):
        # Only include if gradients are computed
        params_grads = []
        if self.grad_W is not None:
            params_grads.append((self.W, self.grad_W))
        if self.grad_b is not None:
            params_grads.append((self.b, self.grad_b))
        return params_grads

    def update_weights(self, learning_rate):
        self.W -= learning_rate * self.grad_W
        self.b -= learning_rate * self.grad_b

class Dropout:
    def __init__(self, drop_prob):
        self.drop_prob = drop_prob
        self.mask = None
        self.has_dims = False  # Dropout does not have fixed input/output dimensions
        self.training = True  # Default to training mode

    def forward(self, x):
        if self.training:
            # Generate dropout mask
            self.mask = (np.random.rand(*x.shape) > self.drop_prob) / (1.0 - self.drop_prob)
            return x * self.mask
        else:
            # During evaluation, pass values through unchanged
            return x

    def backward(self, grad_output):
        # Apply dropout mask to gradient
        return grad_output * self.mask
    
    def get_params_and_grads(self):
        return []  # Dropout has no parameters

    def update_weights(self, learning_rate):
        # Dropout has no weights to update
        pass

        
class FeedForward:
    # class to combine several layers and pass input to first layer then all the way through
    def __init__(self, *layers, training=True):
        self.layers = layers
        self.training = training
        # validate that dimension match up
        # we have to skip dimensionless layers like Dropout
        last_dim_layer = None
        last_index = None
        for i, layer in enumerate(layers):
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
    def forward(self, x):
        # pass x through all the layers
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
    def update_weights(self, learning_rate):
        """
        Apply the weight updates for all layers.
        """
        for layer in self.layers:
            layer.update_weights(learning_rate)
    def __call__(self, x):
        return self.forward(x)

    def set_training(self, mode=True):
        self.training = mode
        for layer in self.layers:
            if hasattr(layer, 'set_training'):
                layer.set_training(mode)
            elif hasattr(layer, 'training'):
                layer.training = mode
    def fit(
        self,
        X,
        y,
        epochs=10,
        loss_fn=None,
        verbose=False,
        learning_rate=0.01,
        metric = None,
        batch_size=32,
        print_every=10,
        val_size=None,
        val_set=None,
        clip_value=None
    ):
        """
        fit model based on X,y
        Parameters
        ----------
        X : numpy array
            features used to train
        y : numpy array
            target used to train
        epochs : int
            number of epochs used in training
        loss : string or callable
            loss function to use in training
            if string, must be a key in activations.loss_functions
        verbose : boolean
            whether or not to print intermediate losses during epochs
        learning_rate: float
            learning rate to use when training
        metric : string or callable
            metric to use to evaluate accuracy during training
            if string, must be a key in metrics_dict
        batch_size : int
            number of samples in a batch
        print_every : int
            how often to print batch loss
        val_size : int or float
            size of set to consider as validation data. When this is a number in (0,1)
            this is a fraction of the set, otherwise a number to be used
        val_set: (array,array)
            A pair (X_val,y_val) to be used for validation
        clip_value: float
            if not None, clip gradients to be in the range [-clip_value, clip_value]
        """
        if loss_fn is None:
            loss_fn = loss.mse
            print("No loss function specified, using mse, warning: this may not be appropriate for classification")
        elif isinstance(loss_fn, str):
            if loss_fn in loss.loss_functions:
                loss_fn = loss.loss_functions[loss_fn]
            else:
                raise ValueError(f"Loss function {loss_fn} not recognized. Must be a callable or one of {list(loss.loss_functions.keys())}")
        elif not callable(loss_fn):
            raise ValueError("Loss must be a callable or a string key in loss_functions")
        if metric is None:
            metric = metrics.accuracy
        elif isinstance(metric, str):
            if metric in metrics.metrics_dict:
                metric = metrics.metrics_dict[metric]
            else:
                raise ValueError(f"Metric {metric} not recognized. Must be a callable or one of {list(metrics.metrics_dict.keys())}")
        elif not callable(metric):
            raise ValueError("Metric must be a callable or a string key in metrics_dict")
        metric_name = metric.__name__
        if self.history is None:
            history = {"loss":[], metric_name:[]}
            if val_set is not None or val_size is not None:
                history["val_loss"] = []
                history[f"val_{metric_name}"] = []
            self.history = history
            # no previous training occurred so the history can start with the epoch count
            last_epoch = 0
        else:
            history = self.history
            # we want to increase the epochs as we go
            last_epoch = history["loss"][-1][0]
            
        
        self.set_training(True) # make sure we're in training mode to fit (mostly for dropout)
        # deal with validation set
        # if a validation set is given separately, use X, y to train
        if val_set is not None:
            X_val,y_val = val_set
            X_train = X.copy()
            y_train = y.copy()
            use_validate = True
        # if no validation set is given, but a size is given, use train_test_split
        # to split into training and validation sets
        elif val_size is not None:
            X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=val_size)
            use_validate = True
        # if nothing about validation is given use X, y as training data
        else:
            X_train = X.copy()
            y_train = y.copy()
            use_validate = False
        # loop through epochs
        num_samples=X_train.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        for epoch in range(1, epochs+1):
            # shuffle X so that we aren't using the same batches every time
            permutation = np.random.permutation(num_samples)
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
                
                loss_batch = loss_fn.forward(y_pred, y_batch)
                epoch_loss += loss_batch * (end - start)  # weighted sum for averaging later

                
                
                # backward pass
                loss_grad = loss_fn.backward(y_pred, y_batch)
                self.backward(loss_grad)
                # clip gradients if requested
                if clip_value is not None:
                    for layer in self.layers:
                        for _, grad in layer.get_params_and_grads():
                            np.clip(grad, -clip_value, clip_value, out=grad)

                # update weights
                self.update_weights(learning_rate)

                # show output if verbose is set to True based on print_every
                if verbose:
                    # Optional: print loss every print_every batches
                    if batch_idx % print_every == 0:
                        running_ave_loss = epoch_loss/end
                        print(f"Epoch {epoch}, Batch {batch_idx+1}/{num_batches}, Loss: {running_ave_loss:.4f}")
            # at the end of the epoch print out the epoch loss/metric
            metric_value = metric(self.predict(X_train), y_train)


            # Average loss over epoch
            epoch_loss /= num_samples
            history["loss"].append((epoch+last_epoch,epoch_loss))
            history[metric_name].append((epoch+last_epoch,metric_value))
            print(f"Epoch {epoch} complete. Average Loss: {epoch_loss:.4f} , {metric_name} {metric_value:.4f}",end=" - ")
            # if no validation set we stop
            if not use_validate:
                print("\n")
            #if there is we compute the loss/accuracy on the validation set from that epoch and report it
            else:
                y_test_pred = self.predict(X_val)
                val_metric = metric(y_test_pred, y_val)
                val_loss = loss_fn.forward(y_test_pred, y_val)
                history["val_loss"].append((epoch+last_epoch,val_loss))
                history[f"val_{metric_name}"].append((epoch+last_epoch,val_metric))
                print(f" Validation Loss: {val_loss:.4f}, Validation {metric_name} {val_metric:.4f}")

        self.history = history
        return copy.deepcopy(history)

    def plot_history(self, *keys):
        if self.history is None:
            print("No training has occured yet")
        else:
            plot_history(self.history, *keys)
            
    def predict(self, X):
        # save current training state
        current_training = self.training
        # propagate to layers
        self.set_training(False)
        out = self.forward(X)
        # restore previous training state
        self.set_training(current_training)
        return out
    
   
def plot_history(history, *keys):
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



def evaluate(model, X):
    model.set_training(False)
    return model(X)
