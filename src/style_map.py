"""How do AFL teams play, and how has the competition changed?

Style is measured with RATIOS, not raw counts, so a fast, high-possession game doesn't
make a team look different (the main caveat from the reliability analysis):
  kick share, uncontested-possession share, marks per disposal, contested-mark share,
  bounces per 100 disposals, tackles per opposition possession, marks inside 50 per
  inside 50, one percenters per opposition inside 50.

Two outputs:
  1. Style map: each club-season is compared with that season's league average
     (z-scores within season), then PCA finds the two main directions clubs differ in.
     Comparing within season keeps "how clubs differ" separate from "how the whole
     league has moved", which would otherwise dominate the first axis.
  2. Competition shift: the league-average value of each ratio by season,
     indexed to 2019 = 100.

Home-and-away games only. Run from the repo root:  python -m src.style_map
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from sklearn.decomposition import PCA

DATA = Path("data/processed/team_matches.csv")
OUT = Path("outputs")
FIG_MAP = OUT / "figures" / "style_map_2026.png"
FIG_SHIFT = OUT / "figures" / "competition_shift.png"
MAP_SEASON = 2026

LABELS = {
    "kick_share": "Kick share of disposals",
    "uncontested_share": "Uncontested share of possessions",
    "marks_per_disposal": "Marks per disposal",
    "contested_mark_share": "Contested share of marks",
    "bounces_per_100_disp": "Bounces per 100 disposals",
    "tackles_per_opp_poss": "Tackles per opposition possession",
    "mi50_per_i50": "Marks inside 50 per inside 50",
    "one_pct_per_opp_i50": "One percenters per opposition inside 50",
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def style_ratios() -> pd.DataFrame:
    d = pd.read_csv(DATA)
    g = d[d["is_final"] == 0].groupby(["season", "team"]).sum(numeric_only=True)
    opp_poss = g["opp_contested_possessions"] + g["opp_uncontested_possessions"]
    return pd.DataFrame({
        "kick_share": g["kicks"] / (g["kicks"] + g["handballs"]),
        "uncontested_share": g["uncontested_possessions"]
                             / (g["uncontested_possessions"] + g["contested_possessions"]),
        "marks_per_disposal": g["marks"] / g["disposals"],
        "contested_mark_share": g["contested_marks"] / g["marks"],
        "bounces_per_100_disp": 100 * g["bounces"] / g["disposals"],
        "tackles_per_opp_poss": g["tackles"] / opp_poss,
        "mi50_per_i50": g["marks_inside_50"] / g["inside_50s"],
        "one_pct_per_opp_i50": g["one_percenters"] / g["opp_inside_50s"],
    })


def fit_style_pca(ratios: pd.DataFrame):
    z = ratios.groupby(level="season").transform(lambda c: (c - c.mean()) / c.std())
    pca = PCA(n_components=2).fit(z)
    scores = pd.DataFrame(pca.transform(z), index=z.index, columns=["pc1", "pc2"])
    # Orient axes so the right = uncontested/running, up = kick-and-mark.
    load = pd.DataFrame(pca.components_.T, index=z.columns, columns=["pc1", "pc2"])
    if load.loc["uncontested_share", "pc1"] < 0:
        scores["pc1"] *= -1; load["pc1"] *= -1
    if load.loc["kick_share", "pc2"] < 0:
        scores["pc2"] *= -1; load["pc2"] *= -1
    return scores, load, pca.explained_variance_ratio_


def style_stability(scores: pd.DataFrame) -> dict:
    """Average correlation of each club's position from one season to the next."""
    out = {}
    for pc in ["pc1", "pc2"]:
        wide = scores[pc].unstack("season")
        seasons = sorted(wide.columns)
        out[pc] = float(np.mean([wide[a].corr(wide[b]) for a, b in zip(seasons, seasons[1:])]))
    return out


def plot_map(scores: pd.DataFrame, var: np.ndarray) -> None:
    s = scores.loc[MAP_SEASON]
    fig, ax = plt.subplots(figsize=(8, 7.4), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    lim = max(s.abs().max().max() * 1.15, 3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color=GRID, lw=1, zorder=0); ax.axvline(0, color=GRID, lw=1, zorder=0)
    ax.scatter(s["pc1"], s["pc2"], s=70, color="#2a78d6", edgecolor=SURFACE, linewidth=2, zorder=3)
    texts = [ax.text(x, y, t, fontsize=8.5, color=INK) for t, (x, y) in s.iterrows()]
    adjust_text(texts, x=s["pc1"].to_numpy(), y=s["pc2"].to_numpy(), ax=ax,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    corner = dict(fontsize=8.5, color=MUTED, style="italic")
    ax.text(-lim * 0.97, lim * 0.95, "Contest & pressure,\nkick-and-mark", va="top", **corner)
    ax.text(lim * 0.97, lim * 0.95, "Uncontested & running,\nkick-and-mark", va="top", ha="right", **corner)
    ax.text(-lim * 0.97, -lim * 0.95, "Contest & pressure,\nhandball & ground ball", va="bottom", **corner)
    ax.text(lim * 0.97, -lim * 0.95, "Uncontested & running,\nhandball & ground ball", va="bottom", ha="right", **corner)
    ax.set_xlabel("← Contest & pressure          Uncontested & running ball movement →",
                  fontsize=9, color=MUTED)
    ax.set_ylabel("← Handball & ground ball          Kick & mark →", fontsize=9, color=MUTED)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.text(0.02, 0.975, f"How the 18 AFL clubs played in {MAP_SEASON}", fontsize=13.5,
             weight="bold", color=INK, ha="left")
    fig.text(0.02, 0.945, "8 style ratios vs that season's league average, reduced to two axes "
             f"({100 * var.sum():.0f}% of the differences between clubs).",
             fontsize=9, color=MUTED, ha="left")
    fig.text(0.02, 0.01, "Home-and-away games. Data: AFL Tables via fitzRoy. Analysis: Stefan Karydis "
             "· github.com/stefank10/afl-team-analytics", fontsize=7.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    FIG_MAP.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_MAP, facecolor=SURFACE)


def league_shift(ratios: pd.DataFrame) -> pd.DataFrame:
    avg = ratios.groupby(level="season").mean()
    return 100 * avg / avg.iloc[0]


def plot_shift(idx: pd.DataFrame) -> None:
    # The five ratios that moved most (kick share and tackles barely changed).
    show = ["bounces_per_100_disp", "mi50_per_i50", "uncontested_share",
            "one_pct_per_opp_i50", "contested_mark_share"]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    fig, ax = plt.subplots(figsize=(8, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    seasons = idx.index.to_numpy()
    ends = []
    for col, c in zip(show, colors):
        ax.plot(seasons, idx[col], color=c, lw=2, label=LABELS[col])
        ax.scatter(seasons[-1], idx[col].iloc[-1], s=40, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ends.append((idx[col].iloc[-1], f"{LABELS[col]}  {idx[col].iloc[-1] - 100:+.0f}%"))
    # Direct labels at line ends, nudged apart so they don't collide.
    ends.sort()
    placed = []
    for v, t in ends:
        y = v if not placed else max(v, placed[-1] + 1.6)
        placed.append(y)
        ax.text(seasons[-1] + 0.15, y, t, va="center", fontsize=8, color=INK)
    ax.axhline(100, color=MUTED, lw=0.8)
    ax.set_xticks(seasons, [str(s) for s in seasons], fontsize=8.5, color=MUTED)
    ax.tick_params(axis="y", labelsize=8.5, colors=MUTED)
    ax.set_ylabel("League average, indexed (2019 = 100)", fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlim(seasons[0] - 0.2, seasons[-1] + 3.6)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, labelcolor=INK)
    fig.text(0.02, 0.965, "How the AFL's style has shifted since 2019", fontsize=13.5,
             weight="bold", color=INK, ha="left")
    fig.text(0.02, 0.93, "More running and bouncing, fewer contested marks and one percenters.",
             fontsize=9, color=MUTED, ha="left")
    fig.text(0.02, 0.01, "Home-and-away games, ratios so game pace doesn't distort them. Data: AFL Tables "
             "via fitzRoy. Analysis: Stefan Karydis", fontsize=7.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.91))
    fig.savefig(FIG_SHIFT, facecolor=SURFACE)


if __name__ == "__main__":
    ratios = style_ratios()
    scores, load, var = fit_style_pca(ratios)
    idx = league_shift(ratios)
    OUT.mkdir(exist_ok=True)
    ratios.round(4).to_csv(OUT / "style_ratios.csv")
    scores.round(3).to_csv(OUT / "style_scores.csv")
    load.round(3).to_csv(OUT / "style_loadings.csv")
    idx.round(1).to_csv(OUT / "competition_shift.csv")
    plot_map(scores, var)
    plot_shift(idx)
    print("Variance explained:", var.round(2), "| stability:", {k: round(v, 2) for k, v in style_stability(scores).items()})
    print(load.round(2))
    print(idx.round(1).T)
