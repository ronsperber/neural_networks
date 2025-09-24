
import numpy as np
import pytest
from my_nn.nn import FeedForward, Dense, BatchNorm, Dropout
from my_nn import loss

def test_l2_regularization_gradients():
    # Simple 1-in, 1-out model with linear activation
    model = FeedForward(
        Dense(1, 1, activation="linear")
    )
    model.layers[0].W = np.array([[2.0]])  # fixed weight
    model.layers[0].b = np.array([[0.0]])  # no bias

    X = np.array([[1.0]])
    y = np.array([[1.0]])

    lambda_l2 = 0.1

    # Train 1 epoch with L2 regularization
    model.fit(
        X,
        y,
        epochs=1,
        loss_fn=loss.MSE(),
        learning_rate=0.01,
        batch_size=1,
        lambda_l2=lambda_l2,
        verbose=False,
    )

    # ---- Recompute expected gradient manually ----
    W = np.array([[2.0]])
    b = np.array([[0.0]])

    # forward
    z = X @ W.T + b
    y_pred = z  # linear activation

    # base loss gradient wrt y_pred
    loss_fn = loss.MSE()
    grad_output = loss_fn.backward(y_pred, y)

    # match Dense.backward math
    grad_z = grad_output * 1.0  # derivative of linear = 1
    grad_from_loss = (grad_z.T @ X) / X.shape[0]  # averaged over batch

    # add reg term (scaled by batch size just like in your fit)
    grad_expected = grad_from_loss +  lambda_l2 * W / X.shape[0]

    grad_W = model.layers[0].grad_W

    # ---- Assertion ----
    np.testing.assert_allclose(
        grad_W,
        grad_expected,
        rtol=1e-6,
        atol=1e-6,
        err_msg=f"Expected {grad_expected}, got {grad_W}"
    )


def test_l1_regularization_gradients():
    # Simple 1-in, 1-out model with linear activation
    model = FeedForward(
        Dense(1, 1, activation="linear")
    )
    model.layers[0].W = np.array([[2.0]])  # fixed weight
    model.layers[0].b = np.array([[0.0]])  # no bias

    X = np.array([[1.0]])
    y = np.array([[1.0]])

    lambda_l1 = 0.1

    # Train 1 epoch with L1 regularization
    model.fit(
        X,
        y,
        epochs=1,
        loss_fn=loss.MSE(),
        learning_rate=0.01,
        batch_size=1,
        lambda_l1=lambda_l1,
        verbose=False,
    )

    # ---- Recompute expected gradient manually ----
    W = np.array([[2.0]])
    b = np.array([[0.0]])

    # forward
    z = X @ W.T + b
    y_pred = z  # linear activation

    # base loss gradient wrt y_pred
    loss_fn = loss.MSE()
    grad_output = loss_fn.backward(y_pred, y)

    # match Dense.backward math
    grad_z = grad_output * 1.0  # derivative of linear = 1
    grad_from_loss = (grad_z.T @ X) / X.shape[0]

    # add L1 reg term
    grad_expected = grad_from_loss + lambda_l1 * np.sign(W) / X.shape[0]

    grad_W = model.layers[0].grad_W

    # ---- Assertion ----
    np.testing.assert_allclose(
        grad_W,
        grad_expected,
        rtol=1e-6,
        atol=1e-6,
        err_msg=f"Expected {grad_expected}, got {grad_W}"
    )


def test_dense_l1_l2_regularization():
    # Simple 1→1 linear model
    model = FeedForward(Dense(1, 1, activation="linear"))
    model.loss_fn = loss.MSE()

    # Fix weights to known value
    W_val = np.array([[2.0]])
    b_val = np.array([[0.0]])
    model.layers[0].W = W_val.copy()
    model.layers[0].b = b_val.copy()

    # Single training example
    X = np.array([[1.0]])
    y = np.array([[0.0]])

    # Regularization strengths
    lambda_l1 = 0.1
    lambda_l2 = 0.2

    y_pred = model.forward(X)

    # Run one training step
    model.fit(
        X, y,
        epochs=1,
        learning_rate=0.01,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        batch_size=1,
        loss_fn="mse"

    )

    # Extract gradients
    grad_W = model.layers[0].grad_W

    # --- Expected gradient ---
    # Base gradient from MSE loss
    
    
    loss_grad = model.loss_fn.backward(y_pred, y)
    grad_base = loss_grad.T @ X / X.shape[0]

    # Add L1 + L2 contributions
    grad_expected = grad_base \
        + lambda_l1 * np.sign(W_val) / X.shape[0] \
        + lambda_l2 * W_val / X.shape[0]

    assert np.allclose(grad_W, grad_expected), \
        f"Expected {grad_expected}, got {grad_W}"

def test_dense_l1_l2_multi_feature_fixed():
    # Simple 2→1 linear model
    model = FeedForward(Dense(2, 1, activation="linear"))
    model.loss_fn = loss.MSE()

    # Fix weights and bias
    W_val = np.array([[1.0, -1.0]])
    b_val = np.array([[0.0]])
    model.layers[0].W = W_val.copy()
    model.layers[0].b = b_val.copy()

    # Single training example
    X = np.array([[3.0, 2.0]])
    y = np.array([[1.0]])

    # Regularization strengths
    lambda_l1 = 0.1
    lambda_l2 = 0.2

    # Compute base gradient before training
    y_pred = model.forward(X)
    loss_grad = model.loss_fn.backward(y_pred, y)
    grad_base = loss_grad.T @ X / X.shape[0]

    # Expected gradient including regularization
    grad_expected = grad_base \
        + lambda_l1 * np.sign(W_val) / X.shape[0] \
        + lambda_l2 * W_val / X.shape[0]  # match how .fit() applies it

    # Run one training step
    model.fit(
        X, y,
        epochs=1,
        learning_rate=0.01,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        batch_size=1,
        loss_fn = "mse"
    )

    # Extract gradient after fit
    grad_W = model.layers[0].grad_W

    assert np.allclose(grad_W, grad_expected), f"Expected {grad_expected}, got {grad_W}"

def test_dense_regularization_fit_loss():
    # Simple 1→1 linear model
    model = FeedForward(Dense(1, 1, activation="linear"))
    model.loss_fn = loss.MSE()

    # Fix weights to known values
    W_val = np.array([[2.0]])
    b_val = np.array([[0.0]])
    model.layers[0].W = W_val.copy()
    model.layers[0].b = b_val.copy()

    # Single training example
    X = np.array([[1.0]])
    y = np.array([[0.0]])
    y_pred = model.predict(X)
    # Regularization strengths
    lambda_l1 = 0.1
    lambda_l2 = 0.2
    
    

    # Manually compute expected loss
    
    base_loss = model.loss_fn.forward(y_pred, y)
    
    reg_loss = lambda_l1 * np.sum(W_val) + 0.5 * lambda_l2 * np.sum(W_val**2)
    
    expected_total_loss = base_loss + reg_loss
    # Run one training step
    history = model.fit(
        X, y,
        epochs=1,
        learning_rate=0.01,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        batch_size=1,
        loss_fn="mse"

    )

    # Extract the loss from history
    total_loss_recorded = history["loss"][0][1]
    # Compare
    assert np.isclose(total_loss_recorded, expected_total_loss), \
        f"Expected {expected_total_loss}, got {total_loss_recorded}"
    
def test_non_regularizable_layer():
    # test to make sure no error occurs if we include layers with no 
    # regularizable parameters
    X = np.array([[1]])
    y = np.array([[0]])

    model = FeedForward(
        Dense(1,2, activation="relu"),
        BatchNorm(2),
        Dense(2,2),
        Dropout(0.5),
        Dense(2,1, activation="sigmoid")
    )
    lambda_l1 = 0.1
    lambda_l2 = 0.05

    history = model.fit(
        X, y,
        epochs=1,
        learning_rate=0.01,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        batch_size=1,
        loss_fn="binary_cross_entropy"
        )