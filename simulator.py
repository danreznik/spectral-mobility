"""
simulator.py

A small, open-source toolkit for exploring the relationship between
Markov-chain properties (spectral gap, conductance) and inequality
(Gini coefficient) in income mobility models.

Background: economists have used the spectral gap of an income
transition matrix as a standard measure of mobility since Sommers &
Conlisk (1979). This tool lets you test whether that same spectral
gap predicts the *inequality* a mobility system settles into, or
whether other properties of the steady-state distribution (entropy,
concentration at the extremes) explain it better.

See PAPER.md in this repository for the full write-up of methods,
findings, and their limitations. See research_log.md for the dated,
honest process behind these results.

Usage (from the command line):
    python simulator.py

Usage (as a library):
    from simulator import analyze_matrix
    results = analyze_matrix(my_matrix, income_values=[30, 60, 120])
    print(results)
"""

import numpy as np
from itertools import combinations


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _normalize_matrix(transition_matrix):
    """
    Converts input to a numpy array, validates it, and normalizes
    each row to sum to exactly 1. Used internally so every function
    that accepts a transition matrix handles validation and rounding
    identically.
    """
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


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def gini_coefficient(incomes):
    """
    Computes the Gini coefficient for a list/array of individual incomes.
    0 = perfect equality, 1 = perfect inequality.

    Uses a sorting-based formula (fast, vectorized) rather than the
    naive pairwise-comparison formula, which is far too slow to run
    repeatedly on large populations.
    """
    incomes = np.sort(np.array(incomes, dtype=float))
    n = len(incomes)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * incomes) - (n + 1) * np.sum(incomes)) / (
        n * np.sum(incomes)
    )


def gini_from_distribution(distribution, income_values):
    """
    Computes the Gini coefficient directly from a probability
    distribution over income brackets and their dollar values,
    without expanding into a simulated population. Mathematically
    exact (no finite-population approximation).
    """
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
    """
    Computes the spectral gap of a transition matrix:
    1 - |second largest eigenvalue|.

    This is the quantity economists (Sommers & Conlisk, 1979) use as
    a standard measure of mobility. Close to 1 = fast mixing
    (mobile). Close to 0 = slow mixing (sticky / immobile).

    Returns (gap, second_eigenvalue).
    """
    matrix = _normalize_matrix(transition_matrix)
    eigenvalues = np.linalg.eigvals(matrix)
    sorted_eigs = sorted(eigenvalues, key=abs, reverse=True)
    second_largest = sorted_eigs[1]
    gap = 1 - abs(second_largest)
    return gap, second_largest


def conductance(transition_matrix, stationary_dist):
    """
    Computes the conductance of a Markov chain: the minimum, over
    every way of splitting the states into two groups, of the
    probability flow between the groups relative to their size.

    Related to spectral gap by Cheeger's inequality:
        gap / 2 <= conductance <= sqrt(2 * gap)

    Note: this checks every possible bipartition of the state space,
    so it scales poorly for a large number of states (2^n splits).
    Fine for the small models (3-10 brackets) this tool is designed for.
    """
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
    """
    Computes the entropy of a distribution: how spread out (high
    entropy) versus concentrated (low entropy) it is across states.
    Entropy is blind to WHICH states hold the mass, and to the
    dollar values assigned to them -- it is a summary of shape only.
    """
    distribution = np.array(distribution, dtype=float)
    nonzero = distribution[distribution > 0]
    return -np.sum(nonzero * np.log(nonzero))


def extreme_mass(distribution):
    """
    Fraction of the population in the extreme brackets (lowest and
    highest), as opposed to the middle bracket(s). Empirically the
    strongest simple predictor of steady-state Gini found in this
    project -- see PAPER.md, Section 4.3.
    """
    distribution = np.array(distribution, dtype=float)
    return distribution[0] + distribution[-1]


def income_variance(distribution, income_values):
    """
    Variance of income in dollar terms, weighted by the steady-state
    distribution. Tested as an alternative to extreme_mass; performed
    consistently worse at predicting Gini -- see PAPER.md, Section 4.3.
    """
    distribution = np.array(distribution, dtype=float)
    income_values = np.array(income_values, dtype=float)
    mean_income = np.sum(distribution * income_values)
    return np.sum(distribution * (income_values - mean_income) ** 2)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def random_transition_matrix(size=3, seed=None):
    """
    Generates a random valid transition matrix (each row sums to 1).
    """
    rng = np.random.default_rng(seed)
    matrix = rng.random((size, size))
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / row_sums


def simulate_population(transition_matrix, initial_distribution, generations=100):
    """
    Simulates a population's distribution across income brackets
    over successive generations, applying the transition matrix
    once per generation. Returns the full history (list of arrays),
    one entry per generation, including generation 0.

    Rows are re-normalized to sum to exactly 1 before simulating.
    Published real-world transition matrices are often reported to
    only 2 decimal places, so their rows can sum to e.g. 0.99 or
    1.01 -- a tiny error that compounds over many generations if
    left uncorrected.
    """
    matrix = _normalize_matrix(transition_matrix)
    distribution = np.array(initial_distribution, dtype=float)

    history = []
    for _ in range(generations):
        history.append(distribution.copy())
        distribution = distribution @ matrix

    return history


def exact_distribution_at_time(transition_matrix, initial_distribution, t):
    """
    Computes the population distribution at generation t using the
    exact eigendecomposition formula, rather than iterating step by
    step. Confirmed in this project to match simulation exactly --
    see PAPER.md, Section 4.5.
    """
    matrix = _normalize_matrix(transition_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
    c = np.linalg.solve(eigenvectors, initial_distribution)
    p_t = eigenvectors @ (c * (eigenvalues ** t))
    return np.real(p_t)


def distribution_to_incomes(distribution, income_values, population_size=10000):
    """
    Converts a group distribution (fractions) into an actual list of
    individual incomes, for use with gini_coefficient().
    """
    incomes = []
    for fraction, income in zip(distribution, income_values):
        count = int(round(fraction * population_size))
        incomes.extend([income] * count)
    return incomes


# ---------------------------------------------------------------------------
# Convenience: run the full analysis pipeline on any matrix
# ---------------------------------------------------------------------------

def analyze_matrix(transition_matrix, income_values, generations=100, initial_distribution=None):
    """
    Runs the full pipeline on a single transition matrix: simulates
    to steady state, then computes every metric explored in this
    project. This is the main entry point for exploring a new
    matrix -- paste in your own, or a real published one, and see
    how it compares to the findings in PAPER.md.

    Parameters
    ----------
    transition_matrix : list of lists, or 2D array
        A square matrix. Rows do not need to sum to exactly 1 --
        real published data is often rounded to 2 decimal places
        (e.g. rows summing to 0.99 or 1.01) and is normalized
        automatically before any calculation.
    income_values : list
        Dollar (or other unit) value assigned to each bracket, in
        the same order as the matrix's states.
    generations : int
        Number of generations to simulate before treating the
        distribution as having reached steady state.
    initial_distribution : list, optional
        Starting distribution. Defaults to "everyone starts in the
        lowest bracket."

    Returns
    -------
    dict with keys: steady_state, gini, spectral_gap, second_eigenvalue,
    conductance, entropy, extreme_mass, income_variance

    Note on consistency: the matrix is normalized ONCE, here, and
    that single normalized matrix is used for every calculation
    below (steady state, spectral gap, conductance). This matters:
    if each metric normalized the matrix independently -- or worse,
    if some did and some didn't -- results could be silently
    inconsistent with each other for real-world data with rounding.
    """
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
    """
    Prints the results from analyze_matrix() in a clean, readable
    format, rounded to sensible precision, instead of raw floats.
    """
    print("Steady-state distribution:", [round(float(x), 4) for x in results["steady_state"]])
    print("Gini coefficient:", round(float(results["gini"]), 4))
    print("Spectral gap:", round(float(results["spectral_gap"]), 4))
    print("Conductance:", round(float(results["conductance"]), 4))
    print("Entropy:", round(float(results["entropy"]), 4))
    print("Extreme mass:", round(float(results["extreme_mass"]), 4))


# ---------------------------------------------------------------------------
# Demo / command-line entry point
# ---------------------------------------------------------------------------

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
