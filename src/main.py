import time

import pennylane as qml
import pennylane.numpy as np

from data import create_random_dataset, generate_theta, get_iris_dataset, get_wine_dataset
from kernel import compute_scalar_kernel
from local_model import no_lightcone_local_model
from plot import plot_errors, plot_kernel_histogram, plot_laziness
from train import (
    compute_laziness_over_iterations,
    compute_laziness_over_qubits,
    train_models,
)
from verify import verifier_1layer, verifier_2layers

# Setting the configuration of the experiment, including
# the number of qubits `n_qubits`, the number of layers `n_layers`, the number of data points `n_data`, the number of data features `data_dim`,
# The number of Gradient Descent steps `steps`,
# The rotation axis chosen to encode the data `angle_encoding_axis`,
# The rotation unitary chosen to encode the variational parameters `variational_unitary`
# The 2-qubit gate used to create entanglement between the qubits `entangling_gate`,
# The numerical range of the inputs [`min_x`, `max_x`], outputs [`min_y`, `max_y`] and the variational parameters [`min_theta`, `max_theta`].

angle_encoding_axis = "Y"
variational_unitary = qml.RX
entangling_gate = qml.CZ

##### This section is manually set for random-ish datasets
# n_qubits = 10
# n_layers = 2

# data_dim = 3
# n_data = 10

# min_y, max_y = -1, 1
# min_x, max_x = -2 * np.pi, 2 * np.pi


# # Initializing the dataset w.r.t the number of data, the number of data features and the number of qubits.
# x, y = create_random_dataset(n_data, n_qubits, data_dim, min_x, max_x, min_y, max_y)
# x_prime, y_prime = create_random_dataset(
#     n_data, n_qubits, data_dim, min_x, max_x, min_y, max_y
# )

###### Otherwise, uncomment these following lines:

n_data = 50

x, y = get_iris_dataset(n_data)

data_dim = x.shape[1]

n_qubits = data_dim
n_layers = 2

##### Ends here

# Initializing the randomly initialized variational parameters.
min_theta, max_theta = -2 * np.pi, 2 * np.pi

theta = generate_theta(min_theta, max_theta, (n_layers, n_qubits))

# Fixed number of learning steps
steps = 100

# The config string used to save the experiment data.
config_str = f"q{n_qubits}-l{n_layers}-d{n_data}-m{data_dim}-s{steps}"

# Main function to draw the kernel histogram.
def main_histogram(load_data: bool = False):
    if load_data:
        x = np.load(f"data/kernel-x-{config_str}.npy")
        x_prime = np.load(f"data/kernel-xprime-{config_str}.npy")
        kernel_values_xx = np.load(f"data/kernel-x.x-{config_str}.npy")
        kernel_values_xxprime = np.load(f"data/kernel-x.xprime-{config_str}.npy")

    else:
        x, _ = create_random_dataset(n_data, n_qubits, data_dim)
        x_prime, _ = create_random_dataset(n_data, n_qubits, data_dim)
        kernel_values_xx = []
        kernel_values_xxprime = []

        for _ in range(steps):
            theta = generate_theta(-np.pi, np.pi, (n_layers, n_qubits))

            # Since our "datasets" each have one data.
            entry_xx = compute_scalar_kernel(n_qubits, n_layers, x[0], x[0], theta)
            entry_xxprime = compute_scalar_kernel(
                n_qubits, n_layers, x[0], x_prime[0], theta
            )
            kernel_values_xx.append(float(entry_xx))
            kernel_values_xxprime.append(float(entry_xxprime))

        # Save the experiment data after it is done.
        np.save(f"data/kernel-x-{config_str}", x)
        np.save(f"data/kernel-xprime-{config_str}", x_prime)
        np.save(f"data/kernel-x.x-{config_str}", kernel_values_xx)
        np.save(f"data/kernel-x.xprime-{config_str}", kernel_values_xxprime)

    plot_kernel_histogram(kernel_values_xx, n_qubits, n_layers, n_data, config_str)
    plot_kernel_histogram(
        kernel_values_xxprime,
        n_qubits,
        n_layers,
        n_data,
        config_str,
        save_str="xxprime",
    )

    print(f"x = {x}")
    print(f"x' = {x_prime}")

    # Verifying the kernel values for the cases where we have analytically computed the expectation values.
    if n_layers == 2:
        verifier = verifier_2layers
    elif n_layers == 1:
        verifier = verifier_1layer
    else:
        verifier = lambda x, xp, n: 0

    print(f"Analytical E[K(x, x)] = {verifier(x[0], x[0], n_qubits)}")
    print(f"Empirical mean K(x, x) = {np.mean(kernel_values_xx)}")
    print("-----------")
    print(f"Analytical E[K(x, x')] = {verifier(x[0], x_prime[0], n_qubits)}")
    print(f"Empirical mean K(x, x') = {np.mean(kernel_values_xxprime)}")


# Main function to draw the error plot over the training iterations.
def main_errplot(
    load_data: bool = False,
    do_local: bool = False,
    do_global: bool = False,
    do_global_linear: bool = False,
    do_local_linear: bool = False,
):
    (local_errs, global_errs, linear_local_errs, linear_global_errs) = [None] * 4
    # Load the experiment data if it has already been done before and we're only drawing the plots again.
    if load_data:
        if do_local:
            local_errs = np.load(f"data/errs-local-{config_str}.npy")
        if do_global:
            global_errs = np.load(f"data/errs-global-{config_str}.npy")
        if do_local_linear:
            linear_local_errs = np.load(f"data/errs-lin_local-{config_str}.npy")
        if do_global_linear:
            linear_global_errs = np.load(f"data/errs-lin_global-{config_str}.npy")
    else:
        (
            steps_until_convergence,
            local_grad_w0_s,
            local_errs,
            linear_local_errs,
            global_errs,
            linear_global_errs,
            w_local_over_time,
        ) = train_models(
            x,
            theta,
            y,
            n_data,
            n_qubits,
            n_layers,
            steps,
            do_local=do_local,
            do_global=do_global,
            do_local_linear=do_local_linear,
            do_global_linear=do_global_linear,
        )

        # Save the data after completing the experiments
        np.save(f"data/errs-local-{config_str}", local_errs)
        np.save(f"data/errs-global-{config_str}", global_errs)
        np.save(f"data/errs-lin_local-{config_str}", linear_local_errs)
        np.save(f"data/errs-lin_global-{config_str}", linear_global_errs)

    plot_errors(
        n_qubits,
        n_layers,
        n_data,
        data_dim,
        config_str,
        loc_errs=local_errs,
        lin_loc_errs=linear_local_errs,
        glob_errs=global_errs,
        lin_glob_errs=linear_global_errs,
    )


# Main function to draw the laziness plot over the number of iterations / number of qubits
# Which one to plot is specified by the `over_qubits` argument.
def main_lazy(lazy_qubits: list, load_data: bool = False, over_qubits: bool = False):
    if over_qubits:
        lazy_list = [
            compute_laziness_over_qubits(
                lazy_qubits, n_data, data_dim, n_layers, steps, load_data=load_data
            )
        ]
        labels = None

    else:
        lazy_list = [
            compute_laziness_over_iterations(
                qu, n_layers, n_data, data_dim, steps, load_data=load_data
            )
            for qu in lazy_qubits
        ]
        labels = [f"{q} Qubits" for q in lazy_qubits]

    plot_laziness(
        lazy_list,
        over_qubits,
        n_qubits,
        n_layers,
        n_data,
        data_dim,
        steps,
        config_str,
        labels=labels,
    )


# Main function to compute the gradients and print the results.
def main_grad():
    t = theta.flatten()
    f = no_lightcone_local_model(x[0], theta, n_qubits, n_layers, True, True)
    grad_fn = qml.jacobian(no_lightcone_local_model, argnum=1)
    grad_t = grad_fn(x[0], t, n_qubits, n_layers, False, False)
    grad_t = np.around(grad_t, decimals=4)
    print(f"x = {x}")
    print(f"theta = {t}")
    print(f"f(x, Θ) = {f}")
    print(f"grad_t f(x, Θ) = {grad_t}")


# The main entry point of the program:
# This is the part where you change if you want to run a different aspect of the experiment.
def main():
    main_errplot(
        load_data=False,
        do_local=True,
        do_global=False,
        do_global_linear=False,
        do_local_linear=True,
    )


# Runs the program and tells you how much time it took.
if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Time Elapsed = {elapsed_time}")
