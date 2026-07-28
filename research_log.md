# Research Log

Raw, dated notes on process.
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
