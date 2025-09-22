import numpy as np
import my_nn.nn as nn

def test_batchnorm_dense_integration():
    np.random.seed(42)

    # Small but diverse toy input
    X = np.random.randn(8, 3)  # 8 samples, 3 features
    

    # Layers: Dense -> BatchNorm -> Dense
    dense1 = nn.Dense(3, 5, activation="relu")
    bn = nn.BatchNorm(num_features=5)
    dense2 = nn.Dense(5, 2, activation="linear")  # logits for binary classification

    # Create network
    model = nn.FeedForward(dense1, bn, dense2)

    # Forward pass
    y_pred = model.forward(X)
    assert y_pred.shape == (8, 2)  # shape matches final Dense layer output

    # Dummy gradient from loss: use random values to ensure gamma/beta gradients aren't zero
    grad_output = np.random.randn(*y_pred.shape)
    model.backward(grad_output)

    # Check that Dense and BatchNorm have gradients
    for layer in model.layers:
        if hasattr(layer, "get_params_and_grads"):
            params_grads = layer.get_params_and_grads()
            for param, grad in params_grads:
                assert grad is not None
                assert param.shape == grad.shape



    # Update weights using all optimizers
    for opt in ["sgd", "momentum", "adam"]:
        # Reset parameters to initial state
        for layer in model.layers:
            if isinstance(layer, nn.BatchNorm):
                layer.gamma = np.ones_like(layer.gamma)
                layer.beta = np.zeros_like(layer.beta)

        # Update weights
        model.update_weights(learning_rate=0.01, optimizer=opt)

        # Check parameters changed
        for layer in model.layers:
            if isinstance(layer, nn.BatchNorm):
                assert not np.allclose(layer.gamma, np.ones_like(layer.gamma))
                assert not np.allclose(layer.beta, np.zeros_like(layer.beta))

    

