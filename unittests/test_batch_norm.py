import numpy as np
import pytest
from my_nn.nn import BatchNorm, Dense

@pytest.fixture
def small_input():
    # 3 samples, 2 features
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])
    grad_output = np.ones_like(X)
    return X, grad_output

@pytest.fixture
def small_input_nonzero_grads():
    # 3 samples, 2 features
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])
    grad_output = np.array([[0.1, -0.2],
                            [0.5, 0.3],
                            [-0.3, 0.2]])
    return X, grad_output

def test_update_weights_momentum(small_input_nonzero_grads):
    X, grad_output = small_input_nonzero_grads
    bn = BatchNorm(num_features=2)
    bn.forward(X)
    bn.backward(grad_output)

    lr = 0.1
    bn.update_weights(learning_rate=lr, optimizer="momentum", beta1=0.9)

    # After one step, gamma and beta should have moved from initial values
    assert not np.allclose(bn.gamma, np.ones(2))
    assert not np.allclose(bn.beta, np.zeros(2))

def test_update_weights_adam(small_input_nonzero_grads):
    X, grad_output = small_input_nonzero_grads
    bn = BatchNorm(num_features=2)
    bn.forward(X)
    bn.backward(grad_output)

    lr = 0.1
    bn.update_weights(learning_rate=lr, optimizer="adam", beta1=0.9, beta2=0.999, t=1)

    # After one step, gamma and beta should have moved from initial values
    assert not np.allclose(bn.gamma, np.ones(2))
    assert not np.allclose(bn.beta, np.zeros(2))


def test_forward_training(small_input):
    X, _ = small_input
    bn = BatchNorm(num_features=2)
    
    bn.set_training(True)
    out = bn.forward(X)
    
    # Compute manually
    mu = np.mean(X, axis=0, keepdims=True)
    var = np.var(X, axis=0, keepdims=True)
    x_norm = (X - mu) / np.sqrt(var + bn.eps)
    expected = bn.gamma * x_norm + bn.beta
    
    np.testing.assert_allclose(out, expected, rtol=1e-7)
    # Running mean and var updated
    np.testing.assert_allclose(bn.running_mean, (1-bn.momentum)*mu, rtol=1e-7)
    np.testing.assert_allclose(bn.running_var, (1-bn.momentum)*var + bn.momentum, rtol=1e-7)

def test_forward_inference(small_input):
    X, _ = small_input
    bn = BatchNorm(num_features=2)
    bn.running_mean = np.array([[1.0, 2.0]])
    bn.running_var = np.array([[4.0, 9.0]])
    
    bn.set_training(False)
    out = bn.forward(X)
    expected = bn.gamma * (X - bn.running_mean) / np.sqrt(bn.running_var + bn.eps) + bn.beta
    
    np.testing.assert_allclose(out, expected, rtol=1e-7)

def test_backward_and_grads(small_input):
    X, grad_output = small_input
    bn = BatchNorm(num_features=2)
    bn.forward(X)
    dx = bn.backward(grad_output)
    
    # Check that gradients are the right shape
    assert bn.grad_gamma.shape == (2,)
    assert bn.grad_beta.shape == (2,)
    assert dx.shape == X.shape
    
    # get_params_and_grads returns correct list
    params_grads = bn.get_params_and_grads()
    assert len(params_grads) == 2
    gamma_param, gamma_grad = params_grads[0]
    beta_param, beta_grad = params_grads[1]
    np.testing.assert_array_equal(gamma_param, bn.gamma)
    np.testing.assert_array_equal(gamma_grad, bn.grad_gamma)
    np.testing.assert_array_equal(beta_param, bn.beta)
    np.testing.assert_array_equal(beta_grad, bn.grad_beta)

def test_update_weights_sgd(small_input):
    X, grad_output = small_input
    bn = BatchNorm(num_features=2)
    bn.forward(X)
    bn.backward(grad_output)
    
    gamma_before = bn.gamma.copy()
    beta_before = bn.beta.copy()
    
    lr = 0.1
    bn.update_weights(learning_rate=lr, optimizer="sgd")
    
    np.testing.assert_allclose(bn.gamma, gamma_before - lr * bn.grad_gamma)
    np.testing.assert_allclose(bn.beta, beta_before - lr * bn.grad_beta)

def test_batchnorm_inference_mode():
    X = np.array([[1., 2.], [3., 4.], [5., 6.]])
    bn = BatchNorm(num_features=2)
    
    # Train mode to update running stats
    bn.forward(X)
    
    # Save running stats
    running_mean = bn.running_mean.copy()
    running_var = bn.running_var.copy()
    
    # Switch to inference
    bn.set_training(False)
    X_out = bn.forward(X)
    
    # Compute manually using running stats
    expected = bn.gamma * (X - running_mean) / np.sqrt(running_var + bn.eps) + bn.beta
    
    assert np.allclose(X_out, expected), "BatchNorm inference does not match expected running statistics"

def test_forward_inference_does_not_change_params():
    bn = BatchNorm(num_features=3)
    bn.forward(np.random.randn(5,3))  # initial forward (training=True)
    old_gamma, old_beta = bn.gamma.copy(), bn.beta.copy()
    old_running_mean, old_running_var = bn.running_mean.copy(), bn.running_var.copy()

    bn.set_training(False)
    _ = bn.forward(np.random.randn(5,3))  # inference

    assert np.allclose(bn.gamma, old_gamma)
    assert np.allclose(bn.beta, old_beta)
    assert np.allclose(bn.running_mean, old_running_mean)
    assert np.allclose(bn.running_var, old_running_var)

def test_batchnorm_dense_integration_full():
    np.random.seed(42)
    X = np.random.randn(6, 4)  # 6 samples, 4 features
    grad_output = np.random.randn(6, 3)

    # Simple network: Dense -> BatchNorm -> Dense
    dense1 = Dense(4, 5, activation="linear")
    bn = BatchNorm(num_features=5)
    dense2 = Dense(5, 3, activation="linear")

    # ----------------------
    # Training pass
    # ----------------------
    bn.set_training(True)
    
    out1 = dense1.forward(X)
    out2 = bn.forward(out1)
    out3 = dense2.forward(out2)

    # Backward pass
    grad_dense2 = dense2.backward(grad_output)
    grad_bn = bn.backward(grad_dense2)
    _ = dense1.backward(grad_bn)

    # Check output shapes
    assert out2.shape == out1.shape
    assert grad_bn.shape == out1.shape
    assert bn.gamma.shape[0] == out1.shape[1]
    assert bn.beta.shape[0] == out1.shape[1]

    # Store original parameters
    gamma_before = bn.gamma.copy()
    beta_before = bn.beta.copy()

    # Update weights
    bn.update_weights(learning_rate=0.01, optimizer="sgd")
    dense1.update_weights(learning_rate=0.01)
    dense2.update_weights(learning_rate=0.01)

    # Check gamma/beta actually changed
    assert not np.allclose(gamma_before, bn.gamma)
    assert not np.allclose(beta_before, bn.beta)

    # ----------------------
    # Inference pass
    # ----------------------
    bn.set_training(False)
    out_inf = bn.forward(out1)

    # Running mean/var used, not batch statistics
    # Mean and variance should not match the batch statistics exactly
    batch_mean = np.mean(out1, axis=0, keepdims=True)
    batch_var = np.var(out1, axis=0, keepdims=True)
    inf_mean = np.mean(out_inf, axis=0, keepdims=True)
    inf_var = np.var(out_inf, axis=0, keepdims=True)

    # Confirm inference uses running estimates
    assert not np.allclose(batch_mean, inf_mean)
    assert not np.allclose(batch_var, inf_var)

def test_batchnorm_running_stats_multiple_batches():
    np.random.seed(42)
    X = np.random.randn(12, 4)  # 12 samples
    grad_output = np.ones((6, 4))  # 6 samples per batch

    bn = BatchNorm(num_features=4)
    bn.set_training(True)

    running_mean_history = []
    running_var_history = []

    batch_size = 6
    for i in range(0, X.shape[0], batch_size):
        X_batch = X[i:i+batch_size]
        out = bn.forward(X_batch)
        _ = bn.backward(grad_output)
        bn.update_weights(learning_rate=0.01)
        running_mean_history.append(bn.running_mean.copy())
        running_var_history.append(bn.running_var.copy())

    # Check that running mean/var actually change over batches
    assert not np.allclose(running_mean_history[0], running_mean_history[-1])
    assert not np.allclose(running_var_history[0], running_var_history[-1])

    # Inference uses running stats
    bn.set_training(False)
    out_inf = bn.forward(X)
    assert out_inf.shape == X.shape

