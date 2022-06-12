import pennylane.numpy as np


# First-order taylor's approximation to the model whose value and gradient are given as parameters.
def linearized_model(
    f_x_w0: np.ndarray, grad_x_w0: np.ndarray, w: np.ndarray, w_0: np.ndarray
):
    f_x = f_x_w0 + np.dot((w - w_0).flatten(), grad_x_w0.flatten())
    return f_x


# This is basically MSE, but for the linearized model so we need the gradients as well to compute this one.
def linearized_model_cost_fn(
    f_x_w0_s: np.ndarray,
    grad_x_w0_s: np.ndarray,
    w: np.ndarray,
    w_0: np.ndarray,
    y_s: np.ndarray,
    n_data: int,
):
    data_range = range(n_data)
    f_x_s = [linearized_model(f_x_w0_s[i], grad_x_w0_s[i], w, w_0) for i in data_range]
    return np.mean(np.array([(y_s[i] - f_x_s[i]) ** 2 for i in data_range]))
