import pennylane.numpy as np

rng = np.random.default_rng()

# Generate random numbers in the given range and with the given shape for the initial weights of the model.
def generate_theta(min_theta, max_theta, shape):
    return rng.uniform(min_theta, max_theta, shape, requires_grad=True)


# Creates a dataset where both x (data) and y (label) are random
# x has the size of (n_data, n_qubits) whose values repeats every `data_dim` times (only `data_dim` features)
# y has the size of (n_data)
def create_random_dataset(n_data, n_qubits, data_dim, min_x, max_x, min_y, max_y):
    x = np.zeros((n_data, n_qubits), requires_grad=False)
    data = rng.uniform(min_x, max_x, (n_data, data_dim), requires_grad=False)
    for i in range(n_data):
        for j in range(n_qubits):
            x[i, j] = data[i, j % data_dim]

    y = rng.uniform(min_y, max_y, n_data, requires_grad=False)
    return x, y


# Same as create_random_dataset but
# the data has only one dimension
# the x vector built from that data is just the same number repeated multiple times, each time scaled by its `qubit index + 1`.
def create_redundant_x_dataset(n_data, n_qubits, min_x, max_x, min_y, max_y):
    x = np.zeros((n_data, n_qubits), requires_grad=False)
    data = rng.uniform(min_x, max_x, (n_data), requires_grad=False)
    for i in range(n_data):
        for j in range(n_qubits):
            x[i, j] = (j + 1) * data[i]

    y = rng.uniform(min_y, max_y, n_data, requires_grad=False)
    return x, y


# Creates a teacher-student dataset where
# x is random
# y is the output of the teacher model
# Basically it's creating a (x, f(x)) dataset where the f is the model given in the parameters.
def create_artifical_y_dataset(
    n_data,
    n_qubits,
    data_dim,
    model,
    n_layers,
    min_x,
    max_x,
    min_y,
    max_y,
    min_theta,
    max_theta,
):
    x, _ = create_random_dataset(n_data, n_qubits, data_dim, min_x, max_x, min_y, max_y)
    theta = generate_theta(min_theta, max_theta, (n_layers, n_qubits))
    y = np.zeros((n_data), requires_grad=False)
    for i in range(n_data):
        y[i] = model(x[i], theta, n_qubits, n_layers)
    return x, y
