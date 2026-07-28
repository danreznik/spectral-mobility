# Research Log

Raw, dated notes on process, findings, dead ends, and next steps.
(See README.md for the clean project summary.)

## 7/26/26

- Set up repo and Python environment (pandas, numpy, jupyter)
- Found key prior work: Sommers & Conlisk (1979) defined an 
  eigenvalue-based mobility measure (M2 = 1 - |second eigenvalue|) - 
  essentially the spectral gap idea already standard in economics
- Focus: connecting Gini coefficient (static inequality) to 
  spectral gap (dynamic mixing speed) of income transition matrices
- Built and verified gini_coefficient() function against known example 
  (0.226 for a standard 10-person test case)
- Built spectral_gap() function (1 - |second eigenvalue| of transition matrix)
- Built simulate_population() to model a population moving through 
  income groups over generations
- Built distribution_to_incomes() to convert group fractions into an 
  actual income list for Gini calculation

Compared two toy 3-state matrices:
- "Sticky" matrix: spectral gap 0.50, steady-state Gini 0.268
- "Mobile" matrix: spectral gap 0.99, steady-state Gini 0.286

Result: the more mobile matrix had a slightly higher Gini, not lower. 
This shows spectral gap (mixing speed) and Gini (steady-state inequality) 
are not simply correlated. 
A system can mix fast toward an unequal outcome, or mix slowly toward 
a relatively equal one.


