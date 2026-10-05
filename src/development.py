"""How do AFL players develop, and which young players are ahead of the typical path?

1. Contribution score (points of margin per game). Each player's own stats are weighted by
   the margin model's points-per-stat (src/margin_model.py). Free kicks for and hit-outs are
   left out: at team level their weights reflect confounding (frees are already counted in
   possessions; the hit-out weight is slightly negative), not a player's value. It's a
   simple, transparent rating, not a complete measure of a player.
2. Development curve. A mixed-effects model of season-average contribution against age
   (quadratic), with each player getting their own baseline and their own age slope. Comparing
   a player with himself avoids survivorship bias: the raw averages look flat after 25 only
   because the players still playing at 30 are the good ones. 2020 is excluded (shortened
   quarters). Player-seasons need at least 5 games.
3. Backtest. Train on seasons before Y, predict Y, for Y = 2024, 2025, 2026. Compared with
   "same as last season" and "last season plus the typical age change".
4. Watchlist. Under-23s in 2026 (8+ games), projected one season ahead.

Run from the repo root:  python -m src.development
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.player_roles import load_player_games

warnings.filterwarnings("ignore")  # statsmodels convergence chatter; convergence is checked below

COEFS = Path("outputs/margin_model_coefficients.csv")
OUT = Path("outputs")
FIG = OUT / "figures" / "development_curve.png"
SCORE_STATS = {
    "diff_inside_50s": "Inside.50s", "diff_marks_inside_50": "Marks.Inside.50",
    "diff_contested_possessions": "Contested.Possessions", "diff_uncontested_possessions": "Uncontested.Possessions",
    "diff_clearances": "Clearances", "diff_tackles": "Tackles", "diff_contested_marks": "Contested.Marks",
    "diff_one_percenters": "One.Percenters", "diff_clangers": "Clangers",
}
MIN_GAMES = 5
SEASON = 2026
FORMULA = "score ~ a + I(a**2)"


def add_contribution(ps: pd.DataFrame) -> pd.DataFrame:
    w = pd.read_csv(COEFS).set_index("feature")["pts_per_unit"]
    ps = ps.copy()
    ps["score"] = sum(w[k] * ps[v] for k, v in SCORE_STATS.items())
    return ps


def player_seasons(ps: pd.DataFrame) -> pd.DataFrame:
    ps = ps[ps["Season"] != 2020]
    g = ps.groupby(["Season", "name"]).agg(
        games=("score", "size"), score=("score", "mean"), age=("Age", "mean"),
        team=("Playing.for", lambda s: s.mode().iat[0])).reset_index()
    g = g[g["games"] >= MIN_GAMES].copy()
    g["a"] = (g["age"] - 24) / 4  # centred and scaled age helps the model converge
    return g


def fit(train: pd.DataFrame):
    res = smf.mixedlm(FORMULA, train, groups=train["name"], re_formula="~a").fit(method="lbfgs")
    if not res.converged:
        raise RuntimeError("mixed model did not converge")
    return res


def predict(res, df: pd.DataFrame) -> np.ndarray:
    """Typical curve at each player's age, plus that player's own baseline and slope."""
    re = res.random_effects
    own = np.array([re[n].iloc[0] + re[n].iloc[1] * a if n in re else 0.0 for n, a in zip(df["name"], df["a"])])
    return np.asarray(res.predict(df)) + own


def backtest(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for test in [2024, 2025, 2026]:
        res = fit(g[g["Season"] < test])
        prev = g[g["Season"] == test - 1][["name", "score"]]
        te = g[g["Season"] == test].merge(prev, on="name", suffixes=("", "_prev"))
        typical_change = np.asarray(res.predict(te)) - np.asarray(res.predict(te.assign(a=te["a"] - 0.25)))
        preds = {"Same as last season": te["score_prev"].to_numpy(),
                 "Last season + typical age change": te["score_prev"].to_numpy() + typical_change,
                 "Development model": predict(res, te)}
        for group, mask in [("All players", te["age"] > 0), ("Under 23", te["age"] < 23)]:
            for model, p in preds.items():
                rows.append({"test_season": test, "group": group, "model": model, "n": int(mask.sum()),
                             "mae": float(np.abs(te["score"][mask] - p[mask]).mean())})
    return pd.DataFrame(rows)


def watchlist(g: pd.DataFrame, res, top: int = 15) -> pd.DataFrame:
    now = g[(g["Season"] == SEASON) & (g["age"] < 23) & (g["games"] >= 8)].copy()
    nxt = now.assign(a=now["a"] + 0.25)
    now["projected_next"] = predict(res, nxt)
    now["typical_next"] = np.asarray(res.predict(nxt))
    now["ahead_of_typical"] = now["projected_next"] - now["typical_next"]
    cols = ["name", "team", "age", "games", "score", "projected_next", "ahead_of_typical"]
    return now.sort_values("projected_next", ascending=False)[cols].head(top).round(2).reset_index(drop=True)


def plot(g: pd.DataFrame, res, wl: pd.DataFrame) -> None:
    ink, muted, grid, surf = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    ages = np.linspace(18.5, 33, 120)
    curve = np.asarray(res.predict(pd.DataFrame({"a": (ages - 24) / 4})))
    sd = np.sqrt(res.cov_re.iloc[0, 0])
    fig, ax = plt.subplots(figsize=(8, 5.6), dpi=200)
    fig.patch.set_facecolor(surf); ax.set_facecolor(surf)
    ax.fill_between(ages, curve - 0.674 * sd, curve + 0.674 * sd, color="#c3c2b7", alpha=0.35, lw=0,
                    label="Middle half of players")
    ax.plot(ages, curve, color="#1F2A44", lw=2.2, label="Typical player")
    for (_, r), c in zip(wl.head(5).iterrows(), colors):
        hist = g[g["name"] == r["name"]].sort_values("Season")
        ax.plot(hist["age"], hist["score"], color=c, lw=2, marker="o", ms=4, label=r["name"])
        ax.plot([hist["age"].iloc[-1], hist["age"].iloc[-1] + 1], [hist["score"].iloc[-1], r["projected_next"]],
                color=c, lw=1.6, ls=(0, (3, 2)))
        ax.scatter(hist["age"].iloc[-1] + 1, r["projected_next"], s=36, facecolor=surf, edgecolor=c, lw=1.6, zorder=4)
    ax.set_xlim(18, 33.5)
    ax.set_xlabel("Age", fontsize=9, color=muted)
    ax.set_ylabel("Contribution (points of margin per game)", fontsize=9, color=muted)
    ax.grid(axis="y", color=grid, lw=0.8); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8.5)
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=ink)
    fig.text(0.02, 0.965, "How AFL players develop, and five young players ahead of the curve", fontsize=13,
             weight="bold", color=ink, ha="left")
    fig.text(0.02, 0.93, "Dashed lines: next-season projection. It pulls back towards typical, because one big season is partly luck.",
             fontsize=9, color=muted, ha="left")
    fig.text(0.02, 0.01, "Seasons 2019–2026 excl. 2020, 5+ games. Data: AFL Tables via fitzRoy. "
             "Analysis: Stefan Karydis", fontsize=7.5, color=muted, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.91))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, facecolor=surf)


if __name__ == "__main__":
    g = player_seasons(add_contribution(load_player_games()))
    bt = backtest(g)
    res = fit(g)
    wl = watchlist(g, res)
    peak = 24 + 4 * (-res.params["a"] / (2 * res.params["I(a ** 2)"]))
    OUT.mkdir(exist_ok=True)
    bt.round(3).to_csv(OUT / "development_backtest.csv", index=False)
    wl.to_csv(OUT / "breakout_watchlist_2027.csv", index=False)
    plot(g, res, wl)
    print(f"{len(g)} player-seasons; typical peak age {peak:.1f}")
    print(bt.groupby(["group", "model"])["mae"].mean().round(3).to_string())
    print(wl.to_string())
