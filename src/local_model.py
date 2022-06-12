from typing import Tuple

import pennylane as qml
import pennylane.numpy as np
from pennylane.operation import Operation as qml_Operation
from pennylane.templates.embeddings import AngleEmbedding as qml_AngleEmbedding


# Extracts qubits related to the observation of the specified qubit.
# i.e. qubits in the light-cone of the qubit specified by `qubit_num`
def get_related_qubits(qubit_num: int, n_qubits: int, n_layers: int) -> dict:
    if n_qubits == 1 or n_layers == 1:
        return {qubit_num: None}

    # Avoiding self-cancelling CZ gates
    elif n_qubits == 2:
        return {qubit_num: qubit_num + 1}

    # A dictionary of control and target qubits
    related = dict()

    # The set of nodes that are expanded when going down the layers
    expanding_nodes = {qubit_num}

    for _ in range(n_layers):
        new_expanding_nodes = set()

        for qubit_node in expanding_nodes:
            # Get the neighboring qubits of the qubit_node
            before = (qubit_node - 1) % n_qubits
            after = (qubit_node + 1) % n_qubits
            related.update({qubit_node: after})
            related.update({before: qubit_node})
            # Only checking the after qubit because the before qubit is always "forwardly" connected to the current qubit_node
            if after not in related:
                related.update({after: None})
            new_expanding_nodes |= {before, after}

        expanding_nodes = new_expanding_nodes

    # Sort entangling qubits by the index of the control qubit
    # (The entangling gate might not always be CZ)
    return dict(sorted(related.items()))


# Generates a subcircuit of the local quantum model that only includes qubits related to the measurement of the specified qubit given in the `main_qubit` argument.
# This is the part which generates each lightcone-optimized subcircuit of the main local circuit.
# i.e. it breaks down the circuit into multiple subcircuits whose measurements can be computed separately.
# This breaking down is what we mean by lightcone-optimization as it reduces the amount of RAM required to simulate the circuit,
# But adds a computational overhead as some gates need to be applied multiple times.
def local_circuit_generator(
    related_qubits: dict,
    dev: qml.Device,
    main_qubit: int,
    n_layers: int,
    angle_encoding_axis: str = "Y",
    variational_unitary: qml_Operation = qml.RX,
    entangling_gate: qml_Operation = qml.CZ,
):
    def ntk_circuit(x, weights):
        sub_wires = list(related_qubits.keys())
        wires_len = len(sub_wires)
        # Encode the input data into the circuit via the specified pauli rotation axis
        qml_AngleEmbedding(x, sub_wires, angle_encoding_axis)
        for i in range(n_layers):
            # Weights are flattened to make it easier to index when differentiating
            layer_weights = weights[i * wires_len : (i + 1) * wires_len]
            qml.broadcast(
                unitary=variational_unitary,
                wires=sub_wires,
                pattern="single",
                parameters=layer_weights,
            )
            # n_layers - 1 because the last layer of CZ's always commutes with each other 
            # when measuring the around the Z-axis
            if i != n_layers - 1:
                for ctrl, target in related_qubits.items():
                    if target is not None:
                        entangling_gate(wires=[ctrl, target])

        return qml.expval(qml.PauliZ(main_qubit))

    return qml.QNode(ntk_circuit, dev, diff_method="parameter-shift")


# Constructs the complete f(x), ∇f(x), and Hf(x) for the local model by evaluating each subcircuit and putting the values together nicely.
# The output of this local model is the sum of the expected values of the PauliZ observable on each qubit.
def local_results(
    x: np.ndarray,
    weights: np.ndarray,
    n_qubits: int,
    n_layers: int,
    do_fx: bool = True,
    do_grad: bool = True,
    do_hess: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    param_count = n_qubits * n_layers

    f_x = np.zeros(1)
    gradient = np.zeros_like(weights)
    hessian = np.zeros((param_count, param_count))

    # Calculates the measurement result of the i'th qubit at each iteration
    for i in range(n_qubits):
        related_i = get_related_qubits(i, n_qubits, n_layers)
        # Can also be computed with formula (always same thing)
        n_related = len(related_i)
        wires_i = list(related_i.keys())

        dev_i = qml.device("qulacs.simulator", wires=wires_i)

        x_i = x[wires_i]
        theta_i = weights[:, wires_i].flatten()

        f = local_circuit_generator(related_i, dev_i, i, n_layers)

        if do_fx:
            f_x += f(x_i, theta_i)

        ## Sanity check to see if circuit is composed correctly.
        # print(qml.draw(f)(inputs_i, theta_i))

        if do_grad or do_hess:
            # Evaluate the gradient of the local subcircuit
            grad_fn = qml.grad(f, argnum=1)
            partial_gradient = grad_fn(x_i, theta_i)

            # Evaluate the hessian of the local subcircuit
            hess_fn = qml.jacobian(grad_fn)
            partial_hessian = hess_fn(x_i, theta_i)

            mapping = {i: wires_i[i] for i in range(n_related)}

            # Map the relevant parts of the gradient and the hessian of the local subcircuit to the overall gradient and hessian
            for j in range(n_layers):
                for k in range(n_related):

                    if do_grad:
                        gradient[j][mapping[k]] += partial_gradient[j * n_related + k]

                    if do_hess:
                        for l in range(n_related):
                            for m in range(n_layers):
                                hessian[
                                    j * n_qubits + mapping[k], m * n_qubits + mapping[l]
                                ] += partial_hessian[
                                    j * n_related + k, m * n_related + l
                                ]

    # Divide the results by the square root of the number of qubits as specified in the paper.
    gradient /= np.sqrt(n_qubits)
    hessian /= np.sqrt(n_qubits)
    f_x /= np.sqrt(n_qubits)

    return f_x, gradient, hessian


# Useful shorthand for evaluating only the output f(x) of the local model.
def local_model(
    x: np.ndarray, w: np.ndarray, n_qubits: int, n_layers: int
) -> np.ndarray:
    f_x, _, _ = local_results(
        x, w, n_qubits, n_layers, do_fx=True, do_grad=False, do_hess=False
    )
    return f_x


# Generate the local circuit without making use of the lightcone restrictions to check the validity of the lightcone-optimized code.
def no_lightcone_circuit_generator(dev, n_qubits, n_layers):
    @qml.qnode(dev, diff_method="parameter-shift")
    def ntk_circuit(x, weights):
        qml_AngleEmbedding(x, rotation="Y", wires=range(n_qubits))
        for i in range(n_layers):
            layer_weights = weights[i * n_qubits : (i + 1) * n_qubits]
            qml.broadcast(
                unitary=qml.RX,
                wires=range(n_qubits),
                pattern="single",
                parameters=layer_weights,
            )
            qml.broadcast(unitary=qml.CZ, wires=range(n_qubits), pattern="ring")
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return ntk_circuit


# Useful shorthand for getting every qubit's measurement result in the lightcone-less local model.
def no_lightcone_local_model(
    x, weights, n_qubits, n_layers, draw: bool = False, flatten: bool = True
):
    dev = qml.device("default.qubit", wires=range(n_qubits))
    if flatten:
        w = weights.flatten()
    else:
        w = weights
    nocone_circuit = no_lightcone_circuit_generator(dev, n_qubits, n_layers)

    if draw:
        print(qml.draw(nocone_circuit)(x, w))

    return nocone_circuit(x, w)


# Useful shorthand for getting the only the output f(x) of the lightcone-less local model.
def no_lightcone_f(x, weights, n_qubits, n_layers):
    ntk_sum = sum(no_lightcone_local_model(x, weights, n_qubits, n_layers))
    return ntk_sum / np.sqrt(n_qubits)


# Evaluate f(x), f(x), ∇f(x), and Hf(x) for the local model directly. (without using any lightcones)
def no_lightcone_results(
    x,
    weights,
    n_qubits,
    n_layers,
    do_fx: bool = True,
    do_grad: bool = True,
    do_hess: bool = True,
):

    f_x, grad_x, hess_x = [None] * 3

    if do_fx:
        f_x = no_lightcone_f(x, weights, n_qubits, n_layers)

    if do_grad or do_hess:
        grad_fn = qml.grad(no_lightcone_f, argnum=1)

        if do_grad:
            grad_x = grad_fn(x, weights, n_qubits, n_layers)

        if do_hess:
            hess_fn = qml.jacobian(grad_fn)
            hess_x = hess_fn(x, weights, n_qubits, n_layers)

    return f_x, grad_x, hess_x
