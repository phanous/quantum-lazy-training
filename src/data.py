import pennylane.numpy as np
import pandas as pd

rng = np.random.default_rng()

# Generate random numbers in the given range and with the given shape for the initial weights of the model.
def generate_theta(min_theta, max_theta, shape):
    return rng.uniform(min_theta, max_theta, shape, requires_grad=True)

def get_wine_dataset(n_data: int = None):
    df_raw = pd.read_csv("datasets/red_wine.csv")

    if n_data is not None:
        df_raw = df_raw.sample(n=n_data)

    # Normalize output between 0 and 1
    df_raw["quality"] -= df_raw["quality"].min()
    df_raw["quality"] /= df_raw["quality"].max()

    columns = df_raw.columns.tolist()
    input_cols = columns[:-1]
    output_cols = columns[-1]

    inputs_numpy = df_raw[input_cols].to_numpy()
    outputs_numpy = df_raw[output_cols].to_numpy()

    x = np.array(inputs_numpy)
    y = np.array(outputs_numpy)

    return x, y


def get_iris_dataset(n_data: int = None):
    df_raw = pd.read_csv("datasets/iris.csv")

    if n_data is not None:
        df_raw = df_raw.sample(n=n_data)

    # Map categorical data to integers
    df_raw["variety"] = pd.factorize(df_raw["variety"])[0]

    # Normalize output between 0 and 1
    # df_raw["variety"] -= df_raw["quality"].min()
    df_raw["variety"] /= df_raw["variety"].max()

    columns = df_raw.columns.tolist()
    input_cols = columns[:-1]
    output_cols = columns[-1]

    inputs_numpy = df_raw[input_cols].to_numpy()
    outputs_numpy = df_raw[output_cols].to_numpy()

    x = np.array(inputs_numpy)
    y = np.array(outputs_numpy)

    return x, y


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
