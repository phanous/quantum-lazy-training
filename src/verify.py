from math import cos, sin

# These are the functions used to check and verify the analytical and the numerical results.

# The closed-form solution for the 2-layer quantum circuit (Appendix A).
# Computes E(K_Θ(x, x'))
def verifier_2layers(x, x_prime, n_qubits):
    psi = lambda u, k: cos(u[k % n_qubits])

    result = 0
    for k in range(n_qubits):
        result += 2 * psi(x, k) * psi(x_prime, k)
        result += (
            psi(x, k - 1)
            * psi(x, k)
            * psi(x, k + 1)
            * psi(x_prime, k - 1)
            * psi(x_prime, k)
            * psi(x_prime, k + 1)
        )
    return result / (4 * n_qubits)


# The closed-form solution for the 1-layer quantum circuit (Appendix A).
# Computes E(K_theta(x, x'))
def verifier_1layer(x, x_prime, n_qubits):
    return sum([cos(x[i]) * cos(x_prime[i]) for i in range(len(x))]) / (2 * n_qubits)


# The closed-form solution for the 2-layer quantum circuit (Appendix A - Lemma 2).
# Computes f_k(theta, x)
def verifier_2layers_f(x, theta, k, n_qubits, n_layers):
    size = n_qubits * n_layers
    psi = lambda u, i: cos(2 * u[i % n_qubits])
    phi = lambda u, i: sin(2 * u[i % size])
    t = theta.flatten()
    return psi(x, k) * psi(t, n_qubits + k) * psi(t, k) - psi(x, k - 1) * psi(x, k) * psi(x, k + 1) * psi(t, k - 1) * psi(t, k + 1) * phi(t, k) * phi(t, n_qubits + k)
