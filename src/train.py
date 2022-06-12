from functools import partial
from typing import Callable, Tuple

import pennylane as qml
import pennylane.numpy as np
from pennylane.numpy.linalg import norm as np_norm

from data import create_random_dataset
from global_model import global_model, global_results
from linearized_model import linearized_model, linearized_model_cost_fn
from local_model import local_model, local_results

# Initialize the weights, array of errors and the array of model outputs over time and different data outputs for the local and the global model.
def init_model(
    w_0: np.ndarray, steps: int, n_data: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = w_0.copy()
    errs = np.zeros((steps))
    fs = np.zeros((steps, n_data))
    return w, errs, fs


# Initialize the array of model outputs over different data inputs and the array of gradients over time for the linearzied versions of the models.
def init_linear_model(
    n_data: int, n_layers: int, n_qubits: int
) -> Tuple[np.ndarray, np.ndarray]:
    f_w0_s = np.zeros((n_data))
    grad_w0_s = np.zeros((n_data, n_layers, n_qubits))
    return f_w0_s, grad_w0_s


# Computes the MSE cost function for the model provided in the arguments.
def model_cost_fn(
    x_s: np.ndarray,
    w: np.ndarray,
    y_s: np.ndarray,
    model: Callable,
    n_qubits: int,
    n_layers: int,
    n_data: int,
) -> np.ndarray:
    data_range = range(n_data)
    f_x_s = [model(x_s[i], w, n_qubits, n_layers) for i in data_range]
    return np.mean(np.array([(y_s[i] - f_x_s[i]) ** 2 for i in data_range]))


# Takes one step in the training of the local and the global model.
def model_step(
    model: Callable,
    cost_fn: Callable,
    opt,
    x_s: np.ndarray,
    w: np.ndarray,
    y_s: np.ndarray,
    n_data: int,
    n_layers: int,
    n_qubits: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # opt.step optimizes the first argument `cost_fn` given the other arguments, and returns a tuple of gradients w.r.t the said arguments.
    # Here we only want to optimize `w` of `f(x, w)` so we take the second element of the returned tuple.
    w = opt.step(cost_fn, x_s, w, y_s)[1]
    # The model's output for every data input for the current step.
    f_step = np.array(
        [model(x_s[i], w, n_qubits, n_layers) for i in range(n_data)]
    ).flatten()
    errs = np_norm(f_step - y_s) / n_data
    return w, f_step, errs


# Takes one step in the training of the linearized models.
def linearized_step(
    cost_fn: Callable,
    opt,
    f_x_w0_s: np.ndarray,
    grad_x_w0_s: np.ndarray,
    w: np.ndarray,
    w_0: np.ndarray,
    y_s: np.ndarray,
    n_data: int,
):
    # Just like in the `model_step` function,
    # We only want to optimize `w` of `f(x, gradient, w)` so we take the third element of the returned tuple.
    w = opt.step(cost_fn, f_x_w0_s, grad_x_w0_s, w, w_0, y_s)[2]
    # The model's output for every data input for the current step.
    f_step = np.array(
        [linearized_model(f_x_w0_s[i], grad_x_w0_s[i], w, w_0) for i in range(n_data)]
    ).flatten()
    errs = np_norm(f_step - y_s) / n_data

    return w, f_step, errs


# Train the models specified by the arguments, which could be any subset of (local, global, linearized local, linearized global).
def train_models(
    x: np.ndarray,
    w_0: np.ndarray,
    y: np.ndarray,
    n_data: int,
    n_qubits: int,
    n_layers: int,
    steps: int = 100,
    conv_threshold: float = 1e-3,
    do_local: bool = True,
    do_global: bool = True,
    do_local_linear: bool = True,
    do_global_linear: bool = True,
    verbose: bool = False,
):
    print("Started training models...")
    # Avoid not defined errors
    (
        local_grad_w0_s,
        local_errs,
        linear_local_errs,
        global_errs,
        linear_global_errs,
        w_local_over_time,
    ) = [None] * 6
    steps_until_convergence = steps

    # Initialize the local model if it is to be trained.
    if do_local:
        w_local, local_errs, local_fs = init_model(w_0, steps, n_data)

        cost_fn_local = partial(
            model_cost_fn,
            model=local_model,
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_data=n_data,
        )

        w_local_over_time = [w_local]

    # Initialize the linearized local model if it is to be trained.
    if do_local_linear:
        w_local_linear, linear_local_errs, linear_local_fs = init_model(
            w_0, steps, n_data
        )
        local_f_w0_s, local_grad_w0_s = init_linear_model(n_data, n_layers, n_qubits)

        for i in range(n_data):
            local_f_w0_s[i], local_grad_w0_s[i], _ = local_results(
                x[i], w_0, n_qubits, n_layers, do_fx=True, do_grad=True, do_hess=False
            )

    # Initialize the global model if it is to be trained.
    if do_global:
        w_global, global_errs, global_fs = init_model(w_0, steps, n_data)

        cost_fn_global = partial(
            model_cost_fn,
            model=global_model,
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_data=n_data,
        )

    # Initialize the linearized global model if it is to be trained.
    if do_global_linear:
        w_global_linear, linear_global_errs, linear_global_fs = init_model(
            w_0, steps, n_data
        )
        global_f_w0_s, global_grad_w0_s = init_linear_model(n_data, n_layers, n_qubits)

        for i in range(n_data):
            global_f_w0_s[i], global_grad_w0_s[i], _ = global_results(
                x[i], w_0, n_qubits, n_layers, do_fx=True, do_grad=True, do_hess=False
            )

    # NOTE: There might be a way to use a better way to find a good step size.
    # Here `1` is used only to avoid the issue of vanishing gradients.
    opt = qml.GradientDescentOptimizer(stepsize=1)

    # Here (and in the `if do_global` section) we make use of partial evaluations to avoid unnecessary duplicate codes
    # As much of the things done is similar w.r.t the initializations and the cost functions of the models.

    model_step_particular = partial(
        model_step,
        opt=opt,
        x_s=x,
        y_s=y,
        n_data=n_data,
        n_layers=n_layers,
        n_qubits=n_qubits,
    )

    cost_fn_linear = partial(linearized_model_cost_fn, n_data=n_data)

    linearized_step_particular = partial(
        linearized_step,
        cost_fn=cost_fn_linear,
        opt=opt,
        w_0=w_0,
        y_s=y,
        n_data=n_data,
    )

    for j in range(steps):

        if verbose:
            print(f"On step {j} out of {steps}")

        if do_local:

            w_local, local_fs[j], local_errs[j] = model_step_particular(
                w=w_local, model=local_model, cost_fn=cost_fn_local
            )

            w_local_over_time.append(w_local)
            # In case the local model converges faster than expected, we note the number of steps it took to converge.
            if local_errs[j] < conv_threshold:
                steps_until_convergence = min(j, steps_until_convergence)

        if do_local_linear:

            (
                w_local_linear,
                linear_local_fs[j],
                linear_local_errs[j],
            ) = linearized_step_particular(
                w=w_local_linear,
                f_x_w0_s=local_f_w0_s,
                grad_x_w0_s=local_grad_w0_s,
            )

        if do_global:
            w_global, global_fs[j], global_errs[j] = model_step_particular(
                w=w_global, model=global_model, cost_fn=cost_fn_global
            )

        if do_global_linear:
            (
                w_global_linear,
                linear_global_fs[j],
                linear_global_errs[j],
            ) = linearized_step_particular(
                w=w_global_linear,
                f_x_w0_s=global_f_w0_s,
                grad_x_w0_s=global_grad_w0_s,
            )

    return (
        steps_until_convergence,
        local_grad_w0_s,
        local_errs,
        linear_local_errs,
        global_errs,
        linear_global_errs,
        w_local_over_time,
    )


# Computes the laziness of the local models with different numbers of qubits and the same training iterations.
# The laziness at iteration `t` is defined as ||Θ^(t) - Θ^(0)|| / ||Θ^(0)||.
def compute_laziness_over_qubits(
    qubits_list: list,
    n_data: int,
    data_dim: int,
    n_layers: int,
    steps: int,
    load_data: bool = False,
) -> dict:
    norm_diffs = dict()

    for n_qubits in qubits_list:
        config_str = f"q{n_qubits}-l{n_layers}-d{n_data}-m{data_dim}-s{steps}"
        saveaddr = f"data/weights-{config_str}.npy"
        if not load_data:
            x, y = create_random_dataset(n_data, n_qubits, data_dim)
            theta = np.random.uniform(
                0, 4 * np.pi, (n_layers, n_qubits), requires_grad=True
            )
            print(f"Training lazy model with {n_qubits} qubits")
            config_str = f"q{n_qubits}-l{n_layers}-d{n_data}-m{data_dim}-s{steps}"
            (_, _, _, _, _, _, w_local_over_time,) = train_models(
                x,
                theta,
                y,
                n_data,
                n_qubits,
                n_layers,
                steps=steps,
                do_local=True,
                do_local_linear=False,
                do_global=False,
                do_global_linear=False,
            )
            np.save(saveaddr, w_local_over_time)

        else:
            w_local_over_time = np.load(saveaddr)

        norm_diffs[n_qubits] = np_norm(
            w_local_over_time[0] - w_local_over_time[-1]
        ) / np_norm(w_local_over_time[0])

    return norm_diffs


# Computes the laziness of a local model with the given numbers of qubits and over each iteration step.
# The laziness at iteration `t` is defined as ||Θ^(t) - Θ^(0)|| / ||Θ^(0)||.
def compute_laziness_over_iterations(
    n_qubits: int,
    n_layers: int,
    n_data: int,
    data_dim: int,
    steps: int,
    config_str: str,
    load_data: bool = False,
) -> dict:
    norm_diffs = dict()

    saveaddr = f"data/weights-{config_str}.npy"

    if load_data:
        w_local_over_time = np.load(saveaddr)
    else:
        print(f"Training lazy model with {n_qubits} qubits")
        x, y = create_random_dataset(n_data, n_qubits, data_dim)
        theta = np.random.uniform(
            0, 4 * np.pi, (n_layers, n_qubits), requires_grad=True
        )
        (_, _, _, _, _, _, w_local_over_time,) = train_models(
            x,
            theta,
            y,
            n_data,
            n_qubits,
            n_layers,
            steps=steps,
            do_local_linear=False,
            do_global=False,
            do_global_linear=False,
        )
        # Save the weights of the local model over time after it's done training.
        np.save(saveaddr, w_local_over_time)

    for i in range(steps):
        norm_diffs[i] = np_norm(w_local_over_time[0] - w_local_over_time[i]) / np_norm(
            w_local_over_time[0]
        )

    return norm_diffs
