# Research Log

Raw notes on process.
(See README.md for the clean project summary.)

## 7/26/26

- Set up repo and Python environment (pandas, numpy, jupyter)
- Built and verified gini_coefficient() function
- Built spectral_gap() function (1 - |second eigenvalue| of transition matrix)
- Built simulate_population() to model a population moving through 
  income groups over generations
- Built distribution_to_incomes() to convert group fractions into an income list for Gini calculation
- Compared two toy 3-state matrices

Result: the more mobile matrix had a slightly higher Gini, not lower. 
This shows spectral gap and Gini coefficient are not simply correlated.

## 7/27/26

- Gini function was too slow, caused a freeze
- Rewrote using faster method
- Reran verification tests
- Generated 50 random matrices
- Spectral gap and steady-state Gini for each matrix
- Model scatter plot

Result: no clear pattern. Gini stayed roughly the same across almost all spectral gap values. A few matrices were outliers with much lower Gini, one especially low (0.05).

## 7/29/26

- Looked at outlier matrices from 50-matrix branch: low Gini happens when steady state concentrates into one bracket and high Gini happens when it stays spread across brackets
- Built entropy function to measure spread
- Regression: entropy predicts Gini much better than spectral gap (R^2=.6)
- Tested if income spacing changes how strong entropy-Gini relationship is
- First test looked like it did, but turned out to be noise after ran repeated trials
- Going to pivot to conductance and Cheeger's inequality

## 7/30/26

- Verified conductance function against Cheeger's inequality on a toy matrix
- Added conductance to regression (added almost nothing)
- Found entropy + extreme mass gets R^2 up to 0.8
- Extended to 5 income brackets (R^2 dropped to 0.5)
- Tested income variance instead of extreme mass (did worse 0.31 vs 0.6), probably because Gini is sensitive to extremes
- Tested model against 2 real matrices (PSID household, Chetty. et al intergenerational)
- Model predicted real Gini within 0.06 both times (consistent error)
- Found both real matrices give nearly the same Gini despite different mobility. Limitation since quintiles are doubly stochastic by construction

## 8/2/26

- Tracked Gini generation by generation, error = |Gini(t)-Gini(final)|
- Plotted error on a log scale (straight line = geometric decay)
- Fit the decay rate, compared to Markov chain convergence theory predicted log |lambda_2| (-0.661 vs -0.693)
- Checked ratios between consecutive errors: found early points decayed slightly slower (contamination from lambda_3) and late points got noisy
- Refit using just the clean middle window, got 0.668, within 1% of true value
- Checked across repeated matrices, pattern held
- Derived the exact formula for the distribution at any time t using eigendecomposition (sum of eigenvalue^t terms), verified simulation matched exactly

## 8/8/26

- Extracted working functions into standalone simulator.py
- Added (analyze_matrix): paste in a matrix, get metrics back
- Spectral gap and conductance weren't normalizing rows, inconsistent rounding
- Added input validation (square matrix, no negatives, no zero rows)
- Re-tested to verify match

## 8/11/26

- Reviewed simulator.py for bugs, fixed issues (off-by-one error, missing input validation, unused parameter, divide-by-zero edge case)
- Added a test suite (10 automated checks) to verify results
- Re-confirmed core findings are unaffected by fixes
