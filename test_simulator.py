import numpy as np
import pytest

from simulator import (
    simulate_population,
    exact_distribution_at_time,
    spectral_gap,
    conductance,
    compute_stationary_distribution,
    distribution_entropy,
    gini_coefficient,
    random_transition_matrix,
    analyze_matrix,
)


def test_exact_matches_simulate_simple():
    M = [[0.9, 0.1], [0.5, 0.5]]
    p0 = [1.0, 0.0]
    for t in [0, 1, 3, 10]:
        sim = simulate_population(M, p0, generations=t)[-1]
        exact = exact_distribution_at_time(M, p0, t)
        assert np.allclose(sim, exact, atol=1e-9)


def test_exact_matches_simulate_across_many_random_matrices():
    for trial in range(30):
        size = np.random.default_rng(trial).integers(2, 6)
        M = random_transition_matrix(size=size, seed=trial)
        p0 = np.zeros(size)
        p0[0] = 1.0
        t = int(np.random.default_rng(trial + 1000).integers(1, 15))
        sim = simulate_population(M, p0, generations=t)[-1]
        exact = exact_distribution_at_time(M, p0, t)
        assert np.allclose(sim, exact, atol=1e-6), f"trial {trial} mismatch"


def test_exact_distribution_at_time_t_zero_is_initial():
    M = [[0.7, 0.3], [0.4, 0.6]]
    p0 = [0.2, 0.8]
    result = exact_distribution_at_time(M, p0, 0)
    assert np.allclose(result, p0, atol=1e-9)


def test_steady_state_independent_of_starting_point():
    M = [[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]]
    ss_a = compute_stationary_distribution(M)
    r_a = analyze_matrix(M, [10, 20, 30], initial_distribution=[1.0, 0.0, 0.0])
    r_b = analyze_matrix(M, [10, 20, 30], initial_distribution=[0.0, 0.0, 1.0])
    assert np.allclose(r_a["steady_state"], r_b["steady_state"], atol=1e-6)
    assert np.allclose(r_a["steady_state"], ss_a, atol=1e-6)


def test_cheeger_inequality_holds_for_reversible_chains():
    for seed in range(10):
        rng = np.random.default_rng(seed)
        a = rng.random((3, 3))
        a = (a + a.T) / 2
        M = a / a.sum(axis=1, keepdims=True)
        ss = compute_stationary_distribution(M)
        gap, _ = spectral_gap(M)
        phi = conductance(M, ss)
        assert gap / 2 - 1e-9 <= phi <= np.sqrt(2 * gap) + 1e-9


def test_gini_known_values():
    assert gini_coefficient([50, 50, 50, 50]) == 0.0
    assert gini_coefficient([0, 0, 0]) == 0.0
    with pytest.raises(ValueError):
        gini_coefficient([])
    with pytest.raises(ValueError):
        gini_coefficient([-5, 10])


def test_distribution_entropy_base_two():
    assert np.isclose(distribution_entropy([0.5, 0.5], base=2), 1.0)
    with pytest.raises(ValueError):
        distribution_entropy([0.5, 0.5], base=1)


def test_real_psid_matrix_matches_known_result():
    real_matrix = [
        [0.64, 0.24, 0.08, 0.03, 0.01],
        [0.23, 0.45, 0.24, 0.07, 0.02],
        [0.08, 0.20, 0.46, 0.23, 0.04],
        [0.04, 0.07, 0.19, 0.54, 0.18],
        [0.03, 0.04, 0.06, 0.16, 0.72],
    ]
    income_values = [25000, 45000, 70000, 130000, 265000]
    results = analyze_matrix(real_matrix, income_values)
    assert abs(results["gini"] - 0.4225) < 0.001


def test_analyze_matrix_rejects_mismatched_sizes():
    with pytest.raises(ValueError):
        analyze_matrix([[0.5, 0.5], [0.3, 0.7]], [10, 20, 30])


def test_analyze_matrix_rejects_degenerate_matrix():
    with pytest.raises(ValueError):
        analyze_matrix([[0.5, 0.5], [0, 0]], [10, 20])
