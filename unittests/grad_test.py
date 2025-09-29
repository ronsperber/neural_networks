import numpy as np
import pytest
from my_nn import activations as act

# Test inputs
X_normal = np.array([[1.0, -1.0, 2.0]])
X_edges  = np.array([[0.0, 50.0, -50.0]])

# All activations you want to test
activations_to_test = [
    act.Sigmoid,
    act.ReLU,
    act.Tanh,
    act.LeakyReLU,
    act.Identity,
    act.ELU,
    act.Swish,
]

# Activations with a corner at x=0
corner_cases = {act.ReLU, act.LeakyReLU, act.ELU}

@pytest.mark.parametrize("activation_cls", activations_to_test)
@pytest.mark.parametrize("X", [X_normal, X_edges])
def test_backward_matches_numeric(activation_cls, X):
    activation = activation_cls()

    y_exact  = activation.backward(X)
    y_approx = act.grad(activation.forward, X)

    if activation_cls in corner_cases:
        # Mask out nondifferentiable points (X == 0)
        mask = (X != 0.0)
        assert np.allclose(y_exact[mask], y_approx[mask], atol=1e-6, rtol=1e-6)
    else:
        assert np.allclose(y_exact, y_approx, atol=1e-6, rtol=1e-6)
