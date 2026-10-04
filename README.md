# AFL Team Analytics

**What actually drives AFL margins, and how do teams play?**
A reproducible Python analysis of public AFL data, 2019–2026.

*Written by Stefan Karydis — sport scientist and football data analyst.

---

## Questions

1. **Which team statistics are signal, not noise?** Split-half reliability of team stats: which ones are stable team traits and which mostly bounce around week to week.
2. **How do teams play?** A style map of all 18 clubs using PCA, and how the competition has shifted since 2019.
3. **What explains the margin?** A regularised regression trained on 2019–2025 and tested on the 2026 season, compared against a simple baseline.

Each section ends with a summary example for coaches.

## Data

Public data from [AFL Tables](https://afltables.com), pulled with the [fitzRoy](https://github.com/jimmyday12/fitzRoy) R package.

## Project structure

```
R/pull_data.R        # raw data pull (fitzRoy, R)
src/                 # Python: cleaning, features, models
notebooks/           # analysis walkthroughs
tests/               # data integrity checks (e.g. team scores match official results)
data/raw/            # raw CSVs from fitzRoy
data/processed/      # tidy team-match table
outputs/figures/     # charts
```

## How to run

```bash
# 1. Pull the raw data (R)
Rscript R/pull_data.R

# 2. Set up Python
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate afl-team-analytics

# 3. Build the tidy table and run the checks
python -m src.build_team_matches
pytest

# 4. Analyses
python -m src.reliability
```

## How I used an AI coding agent

I built this with an AI coding agent as a pair programmer. My rules:

- **I own the questions and the interpretation.** The agent drafts code; I decide what to measure and what the result means.
- **Every number is tested.** Automated tests check the data against known totals (e.g. every team's score matches the official result) before any modelling.
- **Every change is reviewed and committed.** The commit history shows how the analysis developed.

## Results so far

### 1. Which team stats are signal?

![Split-half reliability of AFL team stats](outputs/figures/stat_reliability.png)

Run: `python -m src.reliability` (table in `outputs/reliability.csv`).

## Status

- [x] Repository and data pipeline set up
- [x] Tidy team-match table + integrity tests (8 checks against official results)
- [x] Stat reliability
- [ ] Team style map
- [ ] Margin model
- [ ] Coach summary
