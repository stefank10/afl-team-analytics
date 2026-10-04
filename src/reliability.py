"""Which team stats are stable team traits, and which are mostly noise?

Method (split-half reliability):
  For each season, take home-and-away games only. For each club, average each stat
  over its odd-numbered rounds and, separately, its even-numbered rounds. Correlate
  the two halves across the 18 clubs. A stat that reflects how a team genuinely plays
  will rank teams the same way in both halves (high r); a stat driven by noise won't.
  One season gives only 18 clubs, so we repeat it for all 8 seasons (2019-2026) and
  combine with a Fisher z average. The whiskers show the range across seasons.

Run from the repo root:  python -m src.reliability
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("data/processed/team_matches.csv")
OUT_CSV = Path("outputs/reliability.csv")
OUT_FIG = Path("outputs/figures/stat_reliability.png")

LABELS = {
    "contested_possessions": "Contested possessions",
    "uncontested_possessions": "Uncontested possessions",
    "kicks": "Kicks", "handballs": "Handballs", "disposals": "Disposals",
    "marks": "Marks", "contested_marks": "Contested marks", "marks_inside_50": "Marks inside 50",
    "clearances": "Clearances", "inside_50s": "Inside 50s", "rebounds": "Rebound 50s",
    "tackles": "Tackles", "one_percenters": "One percenters", "hit_outs": "Hit-outs",
    "clangers": "Clangers", "frees_for": "Free kicks for", "frees_against": "Free kicks against",
    "bounces": "Bounces", "goal_assists": "Goal assists", "points_for": "Points scored",
}
REFERENCE = "points_for"  # an outcome, shown for comparison only


def split_half(df: pd.DataFrame, stat: str) -> pd.Series:
    """Odd-vs-even round correlation across clubs, per season."""
    ha = df[df["is_final"] == 0].copy()
    ha["half"] = np.where(ha["round_number"] % 2 == 1, "odd", "even")
    means = ha.groupby(["season", "team", "half"])[stat].mean().unstack("half")
    return means.groupby(level="season").apply(lambda m: m["odd"].corr(m["even"]))


def fisher_mean(r: pd.Series) -> float:
    return float(np.tanh(np.arctanh(r.clip(-0.999, 0.999)).mean()))


def run() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    rows = []
    for stat in LABELS:
        r = split_half(df, stat)
        rows.append({"stat": stat, "label": LABELS[stat], "r_mean": fisher_mean(r),
                     "r_min": r.min(), "r_max": r.max(), "r_2026": r.loc[2026]})
    return pd.DataFrame(rows).sort_values("r_mean", ascending=False).reset_index(drop=True)


def plot(res: pd.DataFrame) -> None:
    ink, muted, grid, accent, ref = "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#8a8984"
    res = res.iloc[::-1]  # most stable at the top
    y = np.arange(len(res))
    fig, ax = plt.subplots(figsize=(8, 7.2), dpi=200)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")

    is_ref = res["stat"].eq(REFERENCE).to_numpy()
    colors = np.where(is_ref, ref, accent)
    ax.hlines(y, res["r_min"], res["r_max"], color=colors, lw=2, alpha=0.35)
    ax.scatter(res["r_mean"], y, s=60, color=colors, zorder=3,
               edgecolor="#fcfcfb", linewidth=2)
    for yi, (_, row) in zip(y, res.iterrows()):
        ax.text(row["r_max"] + 0.02, yi, f"{row['r_mean']:.2f}", va="center",
                fontsize=8.5, color=muted)

    labels = [f"{l} (outcome, for reference)" if s == REFERENCE else l
              for s, l in zip(res["stat"], res["label"])]
    ax.set_yticks(y, labels, fontsize=9.5, color=ink)
    ax.set_xlim(-0.4, 1.05)
    ax.axvline(0, color=muted, lw=0.8)
    ax.grid(axis="x", color=grid, lw=0.8); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", colors=muted, labelsize=8.5)
    ax.set_xlabel("Split-half reliability (r): odd rounds vs even rounds, across 18 clubs",
                  fontsize=9, color=muted)

    fig.text(0.02, 0.975, "Which AFL team stats are a genuine team trait?", fontsize=13.5,
             weight="bold", color=ink, ha="left")
    fig.text(0.02, 0.945, "Dot = average across 2019–2026 seasons; line = range across seasons. "
             "Higher = a team's numbers repeat.", fontsize=9, color=muted, ha="left")
    fig.text(0.02, 0.01, "Data: AFL Tables via fitzRoy, home-and-away games. "
             "Analysis: Stefan Karydis · github.com/stefank10/afl-team-analytics",
             fontsize=7.5, color=muted, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    res = run()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)
    plot(res)
    print(res.round(2).to_string(index=False))
