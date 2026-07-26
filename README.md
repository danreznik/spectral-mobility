# Spectral Gaps and Income Mobility

Investigating whether spectral gap methods from expander graph theory / 
Markov chain analysis relate to how income mobility and inequality 
(Gini coefficient) are measured in economics.

## Motivating question

Does the spectral gap of an income transition matrix (a measure of how 
fast a population "mixes" across income groups, forgetting its starting 
point) predict anything about the steady-state Gini coefficient (a 
measure of inequality) that population converges to?

## Status

Early stage. Built core tools (Gini calculator, spectral gap calculator, 
population simulator) and ran an initial small-scale comparison - 
see `research_log.md` for full process notes and findings so far.

## Structure

- `01_exploration.ipynb` - main working notebook
- `research_log.md` - dated research notes, process, and findings
- `requirements.txt` - Python dependencies

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```
