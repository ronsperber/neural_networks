import pytest
import numpy as np
import pandas as pd
from my_nn import nn, loss, activations

"""
unit tests for the neural networks
run pytest test_nn.py to check
"""
class DummyActivation(activations.Activation):
    def forward(self, x):
        return x

    def backward(self, x):
        return x

def test_get_loss_fn_str():
    output_size = 1
    from_logits = False
    loss_fn = "binary_cross_entropy"
    loss_fn = nn.get_loss_fn(loss_fn, output_size, from_logits)
    assert(isinstance(loss_fn, loss.BinaryCrossEntropyLoss))

def test_get_loss_fn_loss():
    output_size = 1
    from_logits = False
    loss_fn = loss.BinaryCrossEntropyLoss()
    loss_fn = nn.get_loss_fn(loss_fn, output_size, from_logits)
    assert(isinstance(loss_fn, loss.BinaryCrossEntropyLoss))

def test_get_loss_fn_none():
    output_size_1 = 1
    output_size_2 = 2
    from_logits = False
    loss_fn_1 = nn.get_loss_fn(None, output_size_1, from_logits)
    loss_fn_2 = nn.get_loss_fn(None, output_size_2, from_logits)
    assert(isinstance(loss_fn_1, loss.MSE))
    assert(isinstance(loss_fn_2, loss.SoftmaxCrossEntropyLoss))

def test_custom_activation():
    layer = nn.Dense(2,2, activation = DummyActivation())
    x = np.array([[1,2]])
    assert np.array_equal(layer.activation(x),x)
    assert layer.activation_name == "dummyactivation"
    assert isinstance(layer.activation, activations.Activation)

def test_defaults_and_activation_names():
    # Dense layer sets activation_name correctly
    d = nn.Dense(2, 3, activation="ReLU")
    assert d.activation_name == "relu"
    
    d2 = nn.Dense(2, 3)
    assert d2.activation_name == "identity"

def test_invalid_loss_string():
    net = nn.FeedForward(nn.Dense(2, 3))
    with pytest.raises(ValueError):
        net.fit(np.zeros((5,2)), np.zeros(5), loss_fn="not_a_loss")

def test_non_loss_object():
    net = nn.FeedForward(nn.Dense(2, 3))
    with pytest.raises(TypeError):
        net.fit(np.zeros((5,2)), np.zeros(5), loss_fn=123)

def test_softmax_in_hidden_layer_error():
    hidden = nn.Dense(2, 3, activation="softmax")
    output = nn.Dense(3, 2, activation="identity")
    with pytest.raises(ValueError, match="Softmax can only be activation in final layer"):
        nn.FeedForward(hidden, output)


def test_softmax_output_sums_to_one():
    d = nn.Dense(2, 3, activation="softmax")
    y_pred = d.activation.forward(np.random.rand(4,3))
    np.testing.assert_allclose(y_pred.sum(axis=1), np.ones(4), rtol=1e-6)

def test_softmax_crossentropy_from_logits_and_probs():
    y_true = np.array([0, 1, 2])
    logits = np.array([[2.0, 1.0, 0.1],
                       [0.5, 2.0, 0.3],
                       [0.2, 0.1, 1.5]])
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    loss1 = loss.SoftmaxCrossEntropyLoss(from_logits=True)
    loss2 = loss.SoftmaxCrossEntropyLoss(from_logits=False)

    l1 = loss1.forward(logits, y_true)
    l2 = loss2.forward(probs, y_true)
    np.testing.assert_allclose(l1, l2, rtol=1e-6)

def test_identity_backward():
    from my_nn.activations import Identity
    act = Identity()
    z = np.array([[1.0, 2.0], [3.0, 4.0]])
    grad = act.backward(z)
    assert grad == 1.0

def test_layer_dimension_mismatch():
    l1 = nn.Dense(3, 4)
    l2 = nn.Dense(5, 2)  # incompatible
    with pytest.raises(ValueError):
        nn.FeedForward(l1, l2)


def test_activation_name_lower():
    layer = nn.Dense(2, 2, activation="ReLU")
    assert layer.activation_name == "relu"
    assert isinstance(layer.activation, activations.ReLU)

def test_invalid_activation_name():
    with pytest.raises(ValueError):
        nn.Dense(2, 2, activation="not_a_func")

def test_name_of_class_activation():
    layer = nn.Dense(2, 2, activation = activations.ReLU())
    assert layer.activation_name == "relu"

def test_activation_aliases():
    layer_1 = nn.Dense(2, 2, activation = "swish")
    layer_2 = nn.Dense(2, 2, activation = "SiLU")
    layer_3 = nn.Dense(2, 2, activation = activations.Swish())

    assert layer_1.activation_name == "silu"
    assert layer_2.activation_name == "silu"
    assert layer_3.activation_name == "silu"

def test_optimizer_config_defaults():
    layer = nn.Dense(2, 2)
    net = nn.FeedForward(layer)
    assert net.t == 1 # check counter exists
    net.fit(np.zeros((2,2)), np.zeros(2, dtype="int"), optimizer="adam")
    assert net.t == 11  # check counter is correct at end
   

def test_fit_accepts_different_input_types():
    layer = nn.Dense(2, 2)
    net = nn.FeedForward(layer)
    X_list = [[0, 1], [1, 0]]
    y_list = [0, 1]
    net.fit(X_list, y_list)  # should not raise

    
    X_df = pd.DataFrame(X_list)
    y_series = pd.Series(y_list)
    net.fit(X_df, y_series)  # should not raise

def test_large_batch_size():
    layer = nn.Dense(2, 1)
    net = nn.FeedForward(layer)
    X = np.zeros((5,2))
    y = np.zeros(5)
    net.fit(X, y, batch_size=10)  # batch_size > dataset, should work

def test_batch_counter_multiple_batches():
    
    # Create a simple network
    layer1 = nn.Dense(4, 3, activation="relu")
    layer2 = nn.Dense(3, 2, activation="identity")
    net = nn.FeedForward(layer1, layer2)

    # Small dataset with more samples than batch size
    X = np.random.rand(10, 4)
    y = np.random.randint(0, 2, size=(10,))

    # Use batch size smaller than number of samples to ensure multiple batches per epoch
    batch_size = 3
    epochs = 2
    
    net.fit(X, y, epochs=epochs, batch_size=batch_size)

    # Number of batches per epoch = ceil(10/3) = 4, total batches = 4*2 = 8
    # t increments after each batch, starting at 1, so final t = 1 + 8 = 9
    assert net.t == 9, f"Expected t to be 9, got {net.t}"
