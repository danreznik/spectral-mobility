
import numpy as np
from itertools import combinations


# Shared helpers

def _normalize_matrix(transition_matrix, tol=1e-12):
    matrix = np.asarray(transition_matrix, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"transition_matrix must be a square 2D matrix (same number "
            f"of rows and columns). Got shape {matrix.shape}."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError("transition_matrix contains NaN or Inf values.")

    if np.any(matrix < 0):
        raise ValueError(
            "transition_matrix contains negative values -- probabilities "
            "must be zero or positive."
        )

    row_sums = matrix.sum(axis=1)
    if np.any(row_sums <= tol):
        raise ValueError(
            "transition_matrix has a row that sums to (approximately) 0 -- "
            "every state must have SOME probability of transitioning "
            "somewhere (including possibly staying put). This matrix "
            "can't be normalized into valid probabilities."
        )
    return matrix / row_sums[:, np.newaxis]


def compute_stationary_distribution(transition_matrix, tol=1e-10, max_iters=100000):
    matrix = _normalize_matrix(transition_matrix)
    n = matrix.shape[0]

    if n == 1:
        return np.array([1.0])

    vals, vecs = np.linalg.eig(matrix.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    vec = np.real(vecs[:, idx])

    if np.all(vec <= 0):
        vec = -vec

    if np.any(vec < -1e-8) or vec.sum() <= 0:
        vec = _power_iteration_stationary(matrix, tol, max_iters)
    else:
        vec = np.clip(vec, 0, None)

    total = vec.sum()
    if total <= 0:
        vec = _power_iteration_stationary(matrix, tol, max_iters)
        total = vec.sum()

    return vec / total


def _power_iteration_stationary(matrix, tol, max_iters):
    n = matrix.shape[0]
    p = np.full(n, 1.0 / n)
    for _ in range(max_iters):
        p_next = p @ matrix
        if np.linalg.norm(p_next - p, ord=1) < tol:
            return p_next
        p = p_next
    return p


# Core metrics

def gini_coefficient(incomes):
    incomes = np.sort(np.asarray(incomes, dtype=float))
    n = incomes.size
    if n == 0:
        raise ValueError("incomes must contain at least one element.")
    if not np.all(np.isfinite(incomes)):
        raise ValueError("incomes contains NaN or Inf values.")
    total_income = incomes.sum()
    if total_income == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * incomes) - (n + 1) * total_income) / (
        n * total_income
    )


def gini_from_distribution(distribution, income_values, normalize=True):
    p = np.asarray(distribution, dtype=float)
    x = np.asarray(income_values, dtype=float)

    if p.size != x.size:
        raise ValueError(
            "distribution and income_values must have the same length "
            f"(got {p.size} and {x.size})."
        )
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(x)):
        raise ValueError("distribution or income_values contains NaN/Inf.")
    if np.any(p < 0):
        raise ValueError("distribution cannot contain negative values.")

    if normalize:
        s = p.sum()
        if s <= 0:
            raise ValueError("distribution must sum to a positive value.")
        p = p / s

    mean_income = np.sum(p * x)
    if mean_income == 0:
        return 0.0

    total = 0.0
    n = len(p)
    for i in range(n):
        for j in range(n):
            total += p[i] * p[j] * abs(x[i] - x[j])

    return total / (2 * mean_income)


def spectral_gap(transition_matrix):
    matrix = _normalize_matrix(transition_matrix)
    n = matrix.shape[0]

    if n == 1:
        return 0.0, 0.0

    eigenvalues = np.linalg.eigvals(matrix)
    sorted_eigs = sorted(eigenvalues, key=abs, reverse=True)
    second_largest = sorted_eigs[1]
    gap = 1 - abs(second_largest)
    return gap, second_largest


def conductance(transition_matrix, stationary_dist, max_exact_n=16):
    matrix = _normalize_matrix(transition_matrix)
    pi = np.asarray(stationary_dist, dtype=float)
    n = matrix.shape[0]

    if pi.size != n:
        raise ValueError(
            f"stationary_dist length ({pi.size}) must match transition "
            f"matrix size ({n})."
        )
    if not np.all(np.isfinite(pi)):
        raise ValueError("stationary_dist contains NaN or Inf values.")
    if np.any(pi < 0):
        raise ValueError("stationary_dist cannot contain negative values.")
    if n > max_exact_n:
        raise ValueError(
            f"conductance() checks every possible split of the state "
            f"space, which is only practical up to about {max_exact_n} "
            f"states (2^n combinations). Got {n} states. Increase "
            f"max_exact_n if you accept the cost, or reduce the number "
            f"of states."
        )
    if n < 2:
        return None

    pi_total = pi.sum()
    if pi_total <= 0:
        raise ValueError("stationary_dist must sum to a positive value.")
    pi = pi / pi_total

    min_phi = None
    for size in range(1, n // 2 + 1):
        for S in combinations(range(n), size):
            S = set(S)
            S_complement = set(range(n)) - S

            pi_S = sum(pi[i] for i in S)
            pi_S_comp = sum(pi[i] for i in S_complement)
            denom = min(pi_S, pi_S_comp)
            if denom <= 0:
                continue

            flow = sum(pi[i] * matrix[i][j] for i in S for j in S_complement)
            phi = flow / denom
            if min_phi is None or phi < min_phi:
                min_phi = phi

    return min_phi if min_phi is not None else 0.0


def distribution_entropy(distribution, normalize=True):
    p = np.asarray(distribution, dtype=float)
    if p.size == 0:
        return 0.0
    if not np.all(np.isfinite(p)):
        raise ValueError("distribution contains NaN or Inf values.")
    if np.any(p < 0):
        raise ValueError("distribution cannot contain negative values.")

    if normalize:
        s = p.sum()
        if s <= 0:
            raise ValueError("distribution must sum to a positive value.")
        p = p / s

    nonzero = p[p > 0]
    if nonzero.size == 0:
        return 0.0
    return float(-np.sum(nonzero * np.log(nonzero)))


def extreme_mass(distribution):
    p = np.asarray(distribution, dtype=float)
    if p.size == 0:
        raise ValueError("distribution must be non-empty.")
    if not np.all(np.isfinite(p)):
        raise ValueError("distribution contains NaN or Inf values.")
    return float(p[0] + p[-1])


def income_variance(distribution, income_values, normalize=True):
    p = np.asarray(distribution, dtype=float)
    x = np.asarray(income_values, dtype=float)

    if p.size != x.size:
        raise ValueError(
            "distribution and income_values must have the same length "
            f"(got {p.size} and {x.size})."
        )
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(x)):
        raise ValueError("distribution or income_values contains NaN/Inf.")
    if np.any(p < 0):
        raise ValueError("distribution cannot contain negative values.")

    if normalize:
        s = p.sum()
        if s <= 0:
            raise ValueError("distribution must sum to a positive value.")
        p = p / s

    mean_income = float(np.dot(p, x))
    return float(np.dot(p, (x - mean_income) ** 2))


# Simulation

def random_transition_matrix(size=3, seed=None):
    if not isinstance(size, int) or size < 1:
        raise ValueError(f"size must be a positive integer, got {size!r}.")
    rng = np.random.default_rng(seed)
    matrix = rng.random((size, size))
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / row_sums


def simulate_population(transition_matrix, initial_distribution, generations=100):
    matrix = _normalize_matrix(transition_matrix)
    n = matrix.shape[0]

    distribution = np.asarray(initial_distribution, dtype=float).ravel()
    if distribution.size != n:
        raise ValueError(
            f"initial_distribution length ({distribution.size}) must "
            f"match transition matrix size ({n})."
        )
    if not np.all(np.isfinite(distribution)):
        raise ValueError("initial_distribution contains NaN or Inf values.")
    if generations < 0:
        raise ValueError("generations must be >= 0.")

    history = [distribution.copy()]
    for _ in range(generations):
        distribution = distribution @ matrix
        history.append(distribution.copy())

    return history


def exact_distribution_at_time(transition_matrix, initial_distribution, t):
    if not (isinstance(t, (int, np.integer)) and t >= 0):
        raise ValueError(
            f"t must be a nonnegative integer for a discrete-time chain, "
            f"got {t!r}."
        )

    matrix = _normalize_matrix(transition_matrix)
    n = matrix.shape[0]

    p0 = np.asarray(initial_distribution, dtype=float).ravel()
    if p0.size != n:
        raise ValueError(
            f"initial_distribution length ({p0.size}) must match "
            f"transition matrix size ({n})."
        )

    eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
    try:
        c = np.linalg.solve(eigenvectors, p0)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            "Could not solve for eigenbasis coefficients -- the matrix's "
            "eigenvectors may be (near-)singular (a defective matrix). "
            "Use simulate_population() instead for this matrix."
        ) from e

    p_t = eigenvectors @ (c * (eigenvalues ** t))
    return np.real(p_t)


def distribution_to_incomes(distribution, income_values, population_size=10000):
    p = np.asarray(distribution, dtype=float)
    x = list(income_values)

    if p.size != len(x):
        raise ValueError(
            f"distribution length ({p.size}) must match income_values "
            f"length ({len(x)})."
        )
    if not np.all(np.isfinite(p)):
        raise ValueError("distribution contains NaN or Inf values.")
    if np.any(p < 0):
        raise ValueError("distribution cannot contain negative values.")
    if population_size < 1:
        raise ValueError("population_size must be >= 1.")

    s = p.sum()
    if s <= 0:
        raise ValueError("distribution must sum to a positive value.")
    p = p / s

    raw_counts = p * population_size
    counts = np.floor(raw_counts).astype(int)
    remainder = population_size - counts.sum()

    if remainder > 0:
        fractional_parts = raw_counts - counts
        top_indices = np.argsort(-fractional_parts)[:remainder]
        counts[top_indices] += 1

    incomes = []
    for count, income in zip(counts, x):
        incomes.extend([income] * int(count))
    return incomes


# Convenience: run the full analysis pipeline on any matrix

def analyze_matrix(transition_matrix, income_values, generations=100, initial_distribution=None):
    income_values = list(income_values)
    n = len(income_values)

    matrix = _normalize_matrix(transition_matrix)
    if matrix.shape[0] != n:
        raise ValueError(
            f"transition_matrix has {matrix.shape[0]} states but "
            f"income_values has {n} entries -- these must match."
        )

    if initial_distribution is None:
        initial_distribution = [1.0] + [0.0] * (n - 1)

    steady_state = compute_stationary_distribution(matrix)

    gap, lambda2 = spectral_gap(matrix)
    phi = conductance(matrix, steady_state) if n <= 16 else None
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
    steady = [round(float(x), 4) for x in results["steady_state"]]
    print("Steady-state distribution:", steady)
    print("Gini coefficient:", round(float(results["gini"]), 4))
    print("Spectral gap:", round(float(results["spectral_gap"]), 4))

    cond = results.get("conductance")
    print("Conductance:", "N/A (too many states)" if cond is None else round(float(cond), 4))

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
