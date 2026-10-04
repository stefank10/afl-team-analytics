# AFL Team Analytics

**What actually drives AFL margins, and how do teams play?**
A reproducible Python analysis of public AFL data, 2019–2026.

*Author: Stefan Karydis — sport scientist and football data analyst. Work in progress (October 2026).*

---

## Questions

1. **Which team statistics are signal, not noise?** Split-half reliability of team stats: which ones are stable team traits and which mostly bounce around week to week.
2. **How do teams play?** A style map of all 18 clubs using PCA, and how the competition has shifted since 2019.
3. **What explains the margin?** A regularised regression trained on 2019–2025 and tested on the unseen 2026 season, compared against a simple baseline.

Each section ends with a plain-English summary for coaches.

## Data

Public data from [AFL Tables](https://afltables.com), pulled with the [fitzRoy](https://github.com/jimmyday12/fitzRoy) R package. No club, GPS or private data is used anywhere in this repository.

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

# 3. Run the checks
pytest
```

## How I used an AI coding agent

I built this with an AI coding agent (Claude) as a pair programmer. My rules:

- **I own the questions and the interpretation.** The agent drafts code; I decide what to measure and what the result means.
- **Every number is tested.** Automated tests check the data against known totals (e.g. every team's score matches the official result) before any modelling.
- **Every change is reviewed and committed.** The commit history shows how the analysis developed.

*Example of a mistake caught during review: to be added.*

## Status

- [x] Repository and data pipeline set up
- [ ] Tidy team-match table + integrity tests
- [ ] Stat reliability
- [ ] Team style map
- [ ] Margin model
- [ ] Coach summary
