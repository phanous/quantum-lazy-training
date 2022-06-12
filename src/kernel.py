import time

import pennylane.numpy as np
from pennylane.numpy.linalg import norm as np_norm

from local_model import local_results


# Kappa is the _relative_ change in the model's Jacobian
# Which equals:
# the distance moved in the weight (w) space times (d)
# times the rate of change of the model's Jacobian (|| ∇^2_w y(w_0) ||)
# divided by the norm of the jacobian (|| ∇_w y(w_0) ||)
# If kappa << 1, the model is very close to its linear approximation
def compute_kappa(x, weights, y, n_qubits: int, n_layers: int):
    start_time = time.time()
    print("Started kappa calculations...")

    f_x, gradient, hessian = local_results(x, weights, n_qubits, n_layers)
    gradient = gradient.flatten()

    kappa_x = np_norm(f_x - y) * np_norm(hessian) / np.power(np_norm(gradient), 2)

    elapsed_time = time.time() - start_time
    print("Elapsed time: ", elapsed_time)

    return kappa_x


# Computes the kernel entry for the given x and x'
# If gradients are given, the kernel entry is computed using the gradients instead of computing the gradients from scratch
# K_Θ(x, x') = < ∇_w f(x), ∇_w f(x') >
def compute_scalar_kernel(
    n_qubits,
    n_layers,
    x=None,
    xprime=None,
    weights=None,
    gradient_x=None,
    gradient_xprime=None,
):
    if gradient_x is None:
        _, gradient_x, _ = local_results(
            x, weights, n_qubits, n_layers, do_fx=False, do_grad=True, do_hess=False
        )

    if gradient_xprime is None:
        _, gradient_xprime, _ = local_results(
            xprime,
            weights,
            n_qubits,
            n_layers,
            do_fx=False,
            do_grad=True,
            do_hess=False,
        )

    gradient_x = gradient_x.flatten()
    gradient_xprime = gradient_xprime.flatten()

    kernel_entry = np.dot(gradient_x, gradient_xprime)

    return kernel_entry


# Computes the kernel matrix for the given x and x' if we're taking the the entire expectation vector to be the model output
# Doesn't work in the current form because we have only implemented the single-output version of the local model, but this function is left here for reference.
def compute_vector_kernel(
    x, xprime, w, n_qubits, n_layers, n_data, gradient_x=None, gradient_xprime=None
):
    kernel = np.zeros((n_data, n_data))

    print("Computing Kernel...")
    start_time = time.time()

    if gradient_x is None:
        _, gradient_x, _ = local_results(
            x, w, n_qubits, n_layers, do_fx=False, do_grad=True, do_hess=False
        )

    if gradient_xprime is None:
        _, gradient_xprime, _ = local_results(
            xprime, w, n_qubits, n_layers, do_fx=False, do_grad=True, do_hess=False
        )

    kernel = np.outer(gradient_x, gradient_xprime)

    elapsed_time = time.time() - start_time
    print("Elapsed time: ", elapsed_time)

    return kernel
