from typing import List

import matplotlib.pyplot as plt
import pennylane.numpy as np
import seaborn as sns
from matplotlib import animation
from pennylane.numpy.linalg import eig as np_eig

from kernel import compute_vector_kernel


# Plots the given errors of the quantum models over each iteration.
def plot_errors(
    n_qubits,
    n_layers,
    n_data,
    data_dim,
    config_str,
    loc_errs=None,
    lin_loc_errs=None,
    glob_errs=None,
    lin_glob_errs=None,
    extra_str="",
):

    error_fig = plt.figure()
    ax1 = error_fig.add_subplot(111)
    ax1.set_title(
        f"{n_qubits} Qubits, {n_layers} Layers, {n_data} Data, {data_dim} Features"
    )
    ax1.set_xlabel("Iterations")

    if loc_errs is not None:
        ax1.plot(loc_errs, label="Local Quantum Model")

    if lin_loc_errs is not None:
        ax1.plot(lin_loc_errs, label="Linearized Local Quantum Model")

    if glob_errs is not None:
        ax1.plot(glob_errs, label="Global Quantum Model")

    if lin_glob_errs is not None:
        ax1.plot(lin_glob_errs, label="Linearized Global Quantum Model")

    ax1.legend(loc=1)
    plt.show()
    plt.savefig(f"plots/errplot{extra_str}-{config_str}.png")


# NOTE: Two conjugated eigenvalues with order of 1e-18 might show up.
# Plots the eigenvalues of the kernel of the -unimplemented- multi-output version of the quantum model.
def plot_kernel_eigvals(
    x,
    xprime,
    w,
    n_qubits,
    n_layers,
    n_data,
    data_dim,
    gradients=None,
    steps_until_convergence=None,
):
    kernel = compute_vector_kernel(x, xprime, w, n_qubits, n_layers, n_data, gradients)

    eigvals, _ = np_eig(kernel)
    max_eig = np.max(eigvals)
    min_eig = np.min(eigvals)

    min_eig_line = [min_eig for _ in range(n_data)]
    conv_steps_line = [steps_until_convergence for _ in range(n_data)]

    # x_ticks = np.linspace(0, n_data, n_data)
    y_ticks = np.linspace(max_eig, min_eig, num=len(eigvals))
    y_ticks = np.around(y_ticks, 3)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(
        f"Eigenvalues of kernel for qubits = {n_qubits}, layers = {n_layers}, inputs = {n_data}, dim = {data_dim}"
    )
    ax1.set_xlabel("#")
    ax1.set_ylabel("Eigenvalue")
    ax1.plot(min_eig_line, label="Minimum Eigenvalue")
    if steps_until_convergence:
        ax1.plot(conv_steps_line, label="Steps until convergence")

    # ax1.scatter(x_ticks, eigvals)
    plt.yticks(y_ticks, y_ticks)
    # plt.xticks(x_ticks, x_ticks)
    plt.show()
    return eigvals


# Plots a gif of the heatmap of the model's weights over time, in order to visualize how it enters the lazy regime.
def plot_heatmap(w_over_time: list, config_str: str):
    fig = plt.figure()

    def animate(i):
        plt.clf()
        data = w_over_time[i]
        data = np.around(data, decimals=3)
        s = sns.heatmap(
            data,
            vmin=-2 * np.pi,
            vmax=2 * np.pi,
            cmap="YlGnBu",
            linewidths=0.2,
            square=True,
        )
        s.set(xlabel="Qubit", ylabel="Layer")

    anim = animation.FuncAnimation(
        fig, animate, frames=len(w_over_time), interval=500, repeat=False
    )

    plw_writer = animation.PillowWriter(fps=30)
    anim.save(
        f"plots/heatmap-{config_str}.gif",
        writer=plw_writer,
    )

    plt.show()


# Plots the laziness of the model over iterations.
def plot_laziness(
    norm_diffs: List[dict],
    over_qubits: bool,
    n_qubits,
    n_layers,
    n_data,
    data_dim,
    steps,
    config_str,
    labels: list = None,
):
    lazy_fig = plt.figure()
    ax1 = lazy_fig.add_subplot(111)
    if over_qubits:
        ax1.set_title(
            f"Laziness measure for: \n layers = {n_layers}, data = {n_data}, dim = {data_dim}, steps = {steps}"
        )
        ax1.set_xlabel("Qubits")
    else:
        ax1.set_title(
            f"Laziness measure for: \n qubits = {n_qubits}, layers = {n_layers}, data = {n_data}, dim = {data_dim}"
        )
        ax1.set_xlabel("Iterations")

    # Don't need to sort, order is already preserved by the for loop.
    for idx, diffs in enumerate(norm_diffs):
        nq_tup, lazy_tup = zip(*diffs.items())
        if labels:
            ax1.plot(nq_tup, lazy_tup, label=labels[idx])
        else:
            ax1.plot(nq_tup, lazy_tup)
    ax1.legend()
    ax1.plot()
    if over_qubits:
        plt.savefig(f"plots/lazy_qubits-{config_str}.png")
    else:
        plt.savefig(f"plots/lazy_steps-{config_str}.png")


# Histogram of the Kernel values, K_Θ(x, x') for various uniformly random chosen values of Θ between -2π and 2π
def plot_kernel_histogram(
    kernel_values,
    n_qubits,
    n_layers,
    n_data,
    config_str,
    over_str="theta",
    accuracy=0.02,
    save_str="xx",
):
    min_kernel = min(kernel_values)
    max_kernel = max(kernel_values)
    bins = int((max_kernel - min_kernel) / accuracy)
    print(f"Drawing kernel histogram with {bins} bins.")
    fig = plt.figure()
    fig.set_dpi(100)
    ax1 = fig.add_subplot(111)
    ax1.set_title(
        f"Kernel values for: \n qubits = {n_qubits}, layers = {n_layers}, data = {n_data}"
    )
    _, edges, _ = plt.hist(kernel_values, bins=bins, edgecolor="white")
    plt.xlabel("Kernel Entry")
    plt.ylabel("Occurance")
    bin_labels = [
        np.around((edges[i] + edges[i + 1]) / 2, 2) for i in range(len(edges) - 1)
    ]
    plt.xticks(bin_labels, rotation=90)
    # plt.xticks(bin_labels)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"plots/histogram-{save_str}-over{over_str}-{config_str}.png")
