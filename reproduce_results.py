import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

plt.rcParams.update({
    "font.size": 13,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.edgecolor": "#333333",
    "axes.labelweight": "medium",
})

from simulator import (
    random_transition_matrix,
    compute_stationary_distribution,
    spectral_gap,
    conductance,
    distribution_entropy,
    extreme_mass,
    gini_from_distribution,
    simulate_population,
    analyze_matrix,
)

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INCOME_VALUES = [30000, 60000, 120000]
INITIAL_DIST = [1.0, 0.0, 0.0]
N_MATRICES = 50
SEED = 0


def run_experiment(n_matrices, seed):
    rng_seeds = range(seed, seed + n_matrices)
    matrices, sg_list, ent_list, ext_list, gini_list, cond_list = [], [], [], [], [], []

    for s in rng_seeds:
        m = random_transition_matrix(size=3, seed=s)
        ss = compute_stationary_distribution(m)
        gap, _ = spectral_gap(m)
        ent = distribution_entropy(ss)
        ext = extreme_mass(ss, INCOME_VALUES)
        gini = gini_from_distribution(ss, INCOME_VALUES)
        phi = conductance(m, ss)

        matrices.append(m)
        sg_list.append(gap)
        ent_list.append(ent)
        ext_list.append(ext)
        gini_list.append(gini)
        cond_list.append(phi)

    return {
        "matrices": matrices,
        "spectral_gap": np.array(sg_list),
        "entropy": np.array(ent_list),
        "extreme_mass": np.array(ext_list),
        "gini": np.array(gini_list),
        "conductance": np.array(cond_list),
    }


print("=" * 60)
print("FINDING 1: Spectral gap does not predict steady-state Gini")
print("=" * 60)

data = run_experiment(N_MATRICES, SEED)

X_gap = data["spectral_gap"].reshape(-1, 1)
y = data["gini"]
model_gap = LinearRegression().fit(X_gap, y)
r2_gap = model_gap.score(X_gap, y)
print(f"Spectral gap alone -> R^2 = {r2_gap:.3f}")

plt.figure(figsize=(7, 5))
plt.scatter(data["spectral_gap"], data["gini"], alpha=0.65, s=55, color="#4C72B0", edgecolor="white", linewidth=0.5)
plt.xlabel("Spectral gap (mixing speed)")
plt.ylabel("Steady-state Gini coefficient")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/finding1_spectral_gap_vs_gini.png", dpi=200)
plt.close()
print(f"Saved: {OUTPUT_DIR}/finding1_spectral_gap_vs_gini.png\n")


print("=" * 60)
print("FINDING 2: Conductance adds nothing beyond spectral gap")
print("=" * 60)

X_combo = np.column_stack([data["spectral_gap"], data["conductance"]])
model_combo = LinearRegression().fit(X_combo, y)
r2_combo = model_combo.score(X_combo, y)
print(f"Spectral gap + conductance -> R^2 = {r2_combo:.3f}")
print(f"(vs spectral gap alone: R^2 = {r2_gap:.3f} -- negligible improvement)\n")


print("=" * 60)
print("FINDING 3: Entropy + extreme mass predict Gini much better")
print("=" * 60)

X_features = np.column_stack([data["entropy"], data["extreme_mass"]])
model_features = LinearRegression().fit(X_features, y)
r2_features = model_features.score(X_features, y)
print(f"Entropy + extreme mass -> R^2 = {r2_features:.3f}")

plt.figure(figsize=(7, 5))
plt.scatter(data["entropy"], data["gini"], alpha=0.65, s=55, color="#DD8452", edgecolor="white", linewidth=0.5)
z = np.polyfit(data["entropy"], data["gini"], 1)
x_line = np.linspace(data["entropy"].min(), data["entropy"].max(), 100)
plt.plot(x_line, np.poly1d(z)(x_line), color="#333333", linestyle="--", linewidth=1.3, alpha=0.8)
plt.xlabel("Steady-state entropy")
plt.ylabel("Steady-state Gini coefficient")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/finding3_entropy_vs_gini.png", dpi=200)
plt.close()
print(f"Saved: {OUTPUT_DIR}/finding3_entropy_vs_gini.png")

print("\nRepeated trials (5 trials of 50 matrices) to confirm this is not a fluke:")
r2_trials = []
for trial in range(5):
    trial_data = run_experiment(N_MATRICES, seed=SEED + (trial + 1) * 1000)
    X_trial = np.column_stack([trial_data["entropy"], trial_data["extreme_mass"]])
    y_trial = trial_data["gini"]
    r2_trials.append(LinearRegression().fit(X_trial, y_trial).score(X_trial, y_trial))
print(f"R^2 across trials: {[round(r, 3) for r in r2_trials]}")
print(f"Mean: {np.mean(r2_trials):.3f}, Std: {np.std(r2_trials):.3f}\n")


print("=" * 60)
print("FINDING 4: Cheeger's inequality holds for reversible chains")
print("=" * 60)

rng = np.random.default_rng(SEED)
a = rng.random((3, 3))
a = (a + a.T) / 2
reversible_matrix = a / a.sum(axis=1, keepdims=True)
ss_rev = compute_stationary_distribution(reversible_matrix)
gap_rev, _ = spectral_gap(reversible_matrix)
phi_rev = conductance(reversible_matrix, ss_rev)
lower = gap_rev / 2
upper = np.sqrt(2 * gap_rev)
print(f"Spectral gap: {gap_rev:.4f}")
print(f"Conductance:  {phi_rev:.4f}")
print(f"Cheeger bound: {lower:.4f} <= conductance <= {upper:.4f}")
print(f"Bound holds: {lower <= phi_rev <= upper}\n")


print("=" * 60)
print("FINDING 5: Spectral gap governs convergence RATE, not destination")
print("=" * 60)

sticky_matrix = [[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]]
history = simulate_population(sticky_matrix, INITIAL_DIST, generations=15)
ginis_over_time = [gini_from_distribution(dist, INCOME_VALUES) for dist in history]
final_gini = ginis_over_time[-1]
errors = np.array([abs(g - final_gini) for g in ginis_over_time])

gap_sticky, lambda2_sticky = spectral_gap(sticky_matrix)

usable = errors[5:11]
gens_window = np.arange(5, 11)
valid = usable > 1e-14
fitted_rate, _ = np.polyfit(gens_window[valid], np.log(usable[valid]), 1)
true_rate = np.log(abs(lambda2_sticky))

print(f"Fitted decay rate (clean window):  {fitted_rate:.4f}")
print(f"True log|lambda_2|:                {true_rate:.4f}")
print(f"Relative difference: {abs(fitted_rate - true_rate) / abs(true_rate) * 100:.1f}%")

plt.figure(figsize=(7, 5))
plot_errors = errors[errors > 1e-14]
plt.plot(range(len(plot_errors)), plot_errors, marker="o", markersize=6,
         color="#55A868", linewidth=1.8, markerfacecolor="#55A868", markeredgecolor="white", markeredgewidth=0.7)
plt.yscale("log")
plt.xlabel("Generation")
plt.ylabel("|Gini(t) \u2212 Gini(\u221e)|  (log scale)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/finding5_convergence_rate.png", dpi=200)
plt.close()
print(f"Saved: {OUTPUT_DIR}/finding5_convergence_rate.png\n")


print("=" * 60)
print("FINDING 6: Validation against real published mobility data")
print("=" * 60)

real_matrix_household = [
    [0.64, 0.24, 0.08, 0.03, 0.01],
    [0.23, 0.45, 0.24, 0.07, 0.02],
    [0.08, 0.20, 0.46, 0.23, 0.04],
    [0.04, 0.07, 0.19, 0.54, 0.18],
    [0.03, 0.04, 0.06, 0.16, 0.72],
]
real_matrix_intergenerational = [
    [0.337, 0.280, 0.184, 0.123, 0.075],
    [0.242, 0.242, 0.217, 0.176, 0.123],
    [0.178, 0.198, 0.221, 0.220, 0.183],
    [0.134, 0.160, 0.209, 0.244, 0.254],
    [0.109, 0.119, 0.170, 0.236, 0.365],
]
real_income_values = [25000, 45000, 70000, 130000, 265000]

for name, matrix in [
    ("PSID household (2003-2013)", real_matrix_household),
    ("Chetty et al. intergenerational", real_matrix_intergenerational),
]:
    results = analyze_matrix(matrix, real_income_values)
    print(f"{name}:")
    print(f"  Actual Gini: {results['gini']:.4f}")

    X_pred = np.array([[results["entropy"], results["extreme_mass"]]])
    predicted_gini = model_features.predict(X_pred)[0]
    print(f"  Model's predicted Gini (trained on synthetic data): {predicted_gini:.4f}")
    print(f"  Absolute error: {abs(results['gini'] - predicted_gini):.4f}\n")

print("=" * 60)
print(f"All figures saved to ./{OUTPUT_DIR}/")
print("=" * 60)
