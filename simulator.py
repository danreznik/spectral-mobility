
import numpy as np
from itertools import combinations


# Shared helper

def _normalize_matrix(transition_matrix):
    matrix = np.array(transition_matrix, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"transition_matrix must be a square 2D matrix (same number "
            f"of rows and columns). Got shape {matrix.shape}."
        )

    if np.any(matrix < 0):
        raise ValueError(
            "transition_matrix contains negative values -- probabilities "
            "must be zero or positive."
        )

    row_sums = matrix.sum(axis=1)
    if np.any(row_sums == 0):
        raise ValueError(
            "transition_matrix has a row that sums to 0 -- every state "
            "must have SOME probability of transitioning somewhere "
            "(including possibly staying put). This matrix can't be "
            "normalized into valid probabilities."
        )
    return matrix / row_sums[:, np.newaxis]


# Core metrics

def gini_coefficient(incomes):
    incomes = np.sort(np.array(incomes, dtype=float))
    n = len(incomes)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * incomes) - (n + 1) * np.sum(incomes)) / (
        n * np.sum(incomes)
    )


def gini_from_distribution(distribution, income_values):
    p = np.array(distribution, dtype=float)
    x = np.array(income_values, dtype=float)
    mean_income = np.sum(p * x)

    total = 0.0
    n = len(p)
    for i in range(n):
        for j in range(n):
            total += p[i] * p[j] * abs(x[i] - x[j])

    return total / (2 * mean_income)


def spectral_gap(transition_matrix):
    matrix = _normalize_matrix(transition_matrix)
    eigenvalues = np.linalg.eigvals(matrix)
    sorted_eigs = sorted(eigenvalues, key=abs, reverse=True)
    second_largest = sorted_eigs[1]
    gap = 1 - abs(second_largest)
    return gap, second_largest


def conductance(transition_matrix, stationary_dist):
    matrix = _normalize_matrix(transition_matrix)
    pi = np.array(stationary_dist, dtype=float)
    n = len(pi)

    min_phi = float("inf")
    for size in range(1, n):
        for S in combinations(range(n), size):
            S = set(S)
            S_complement = set(range(n)) - S

            flow = sum(pi[i] * matrix[i][j] for i in S for j in S_complement)
            pi_S = sum(pi[i] for i in S)
            pi_S_comp = sum(pi[i] for i in S_complement)

            denom = min(pi_S, pi_S_comp)
            if denom > 0:
                phi = flow / denom
                min_phi = min(min_phi, phi)

    return min_phi


def distribution_entropy(distribution):
    distribution = np.array(distribution, dtype=float)
    nonzero = distribution[distribution > 0]
    return -np.sum(nonzero * np.log(nonzero))


def extreme_mass(distribution):
    distribution = np.array(distribution, dtype=float)
    return distribution[0] + distribution[-1]


def income_variance(distribution, income_values):
    distribution = np.array(distribution, dtype=float)
    income_values = np.array(income_values, dtype=float)
    mean_income = np.sum(distribution * income_values)
    return np.sum(distribution * (income_values - mean_income) ** 2)


# Simulation

def random_transition_matrix(size=3, seed=None):
    rng = np.random.default_rng(seed)
    matrix = rng.random((size, size))
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / row_sums


def simulate_population(transition_matrix, initial_distribution, generations=100):
    matrix = _normalize_matrix(transition_matrix)
    distribution = np.array(initial_distribution, dtype=float)

    history = []
    for _ in range(generations):
        history.append(distribution.copy())
        distribution = distribution @ matrix

    return history


def exact_distribution_at_time(transition_matrix, initial_distribution, t):
    matrix = _normalize_matrix(transition_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
    c = np.linalg.solve(eigenvectors, initial_distribution)
    p_t = eigenvectors @ (c * (eigenvalues ** t))
    return np.real(p_t)


def distribution_to_incomes(distribution, income_values, population_size=10000):
    incomes = []
    for fraction, income in zip(distribution, income_values):
        count = int(round(fraction * population_size))
        incomes.extend([income] * count)
    return incomes


# Convenience: run the full analysis pipeline on any matrix

def analyze_matrix(transition_matrix, income_values, generations=100, initial_distribution=None):
    n = len(income_values)
    if len(transition_matrix) != n:
        raise ValueError(
            f"transition_matrix has {len(transition_matrix)} states but "
            f"income_values has {n} entries -- these must match."
        )
    matrix = _normalize_matrix(transition_matrix)

    if initial_distribution is None:
        initial_distribution = [1.0] + [0.0] * (n - 1)

    history = simulate_population(matrix, initial_distribution, generations)
    steady_state = history[-1]

    gap, lambda2 = spectral_gap(matrix)
    phi = conductance(matrix, steady_state)
    ent = distribution_entropy(steady_state)
    ext = extreme_mass(steady_state)
    var = income_variance(steady_state, income_values)
    gini = gini_from_distribution(steady_state, income_values)

    return {
        "steady_state": steady_state,
        "gini": gini,
        "spectral_gap": gap,
        "second_eigenvalue": lambda2,
        "conductance": phi,
        "entropy": ent,
        "extreme_mass": ext,
        "income_variance": var,
    }


def print_report(results):
    print("Steady-state distribution:", [round(float(x), 4) for x in results["steady_state"]])
    print("Gini coefficient:", round(float(results["gini"]), 4))
    print("Spectral gap:", round(float(results["spectral_gap"]), 4))
    print("Conductance:", round(float(results["conductance"]), 4))
    print("Entropy:", round(float(results["entropy"]), 4))
    print("Extreme mass:", round(float(results["extreme_mass"]), 4))


# Demo / command-line entry point

if __name__ == "__main__":
    real_matrix = [
        [0.64, 0.24, 0.08, 0.03, 0.01],
        [0.23, 0.45, 0.24, 0.07, 0.02],
        [0.08, 0.20, 0.46, 0.23, 0.04],
        [0.04, 0.07, 0.19, 0.54, 0.18],
        [0.03, 0.04, 0.06, 0.16, 0.72],
    ]
    real_income_values = [25000, 45000, 70000, 130000, 265000]

    results = analyze_matrix(real_matrix, real_income_values)
    print_report(results)
