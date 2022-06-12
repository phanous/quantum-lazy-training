from typing import Tuple

import pennylane as qml
import pennylane.numpy as np
from pennylane.operation import Operation as qml_Operation
from pennylane.templates.embeddings import AngleEmbedding as qml_AngleEmbedding


# This has the same structure as the local model, however, it uses the global Z^{\otimes n} observable.
def global_circuit_generator(
    dev: qml.Device,
    n_qubits: int,
    n_layers: int,
    angle_encoding_axis: str = "Y",
    variational_unitary: qml_Operation = qml.RX,
    entangling_gate: qml_Operation = qml.CZ,
) -> qml.QNode:
    qubits_range = range(n_qubits)

    def ntk_circuit(x, weights):
        qml_AngleEmbedding(x, qubits_range, angle_encoding_axis)
        for i in range(n_layers):
            layer_weights = weights[i].flatten()
            qml.broadcast(
                unitary=variational_unitary,
                wires=qubits_range,
                pattern="single",
                parameters=layer_weights,
            )
            qml.broadcast(unitary=entangling_gate, wires=qubits_range, pattern="ring")

        obs = qml.PauliZ(0)
        for i in range(1, n_qubits):
            obs = obs @ qml.PauliZ(i)

        return qml.expval(obs)

    return qml.QNode(ntk_circuit, dev, diff_method="parameter-shift")


# Given the weights and the inputs, calculates the following of the global model if asked:
# f_x: the output of the global model
# grad: the gradient of the global model
# hess: the hessian of the global model
def global_results(
    x: np.ndarray,
    w: np.ndarray,
    n_qubits: int,
    n_layers: int,
    do_fx: bool = True,
    do_grad: bool = True,
    do_hess: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    device = qml.device("qulacs.simulator", wires=range(n_qubits))
    global_circ = global_circuit_generator(device, n_qubits, n_layers)

    param_count = n_qubits * n_layers

    f_x = np.zeros(1)
    gradient = np.zeros_like(w)
    hessian = np.zeros((param_count, param_count))

    if do_fx:
        f_x = global_circ(x, w)

    if do_grad or do_hess:
        grad_fn = qml.grad(global_circ, argnum=1)

        if do_grad:
            gradient = grad_fn(x, w)

        if do_hess:
            hess_fn = qml.jacobian(grad_fn)
            hessian = hess_fn(x, w)

    return (
        f_x,
        gradient,
        hessian,
    )


# Shorthand for getting only the output f(x) of the global model
def global_model(
    x: np.ndarray,
    w: np.ndarray,
    n_qubits: int,
    n_layers: int,
) -> np.ndarray:
    f_x, _, _ = global_results(
        x, w, n_qubits, n_layers, do_fx=True, do_grad=False, do_hess=False
    )
    return f_x
