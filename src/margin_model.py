"""What explains the margin? A regularised regression on stat differentials.

Question: given how a game's team stats played out, how much of the final margin do
they explain, and which differentials matter most? This is a DESCRIPTIVE model of what
happened in a game, not a pre-game prediction model.

Design choices (each one is a deliberate rigour decision):
  - One row per game, from the home team's perspective, so no game is counted twice.
  - Features = home-minus-away differentials of process stats. Goals, behinds and goal
    assists are excluded: they ARE the score, so using them would be leakage.
    Disposals are dropped because they are exactly kicks + handballs.
  - Rebound 50s are excluded too. A first version included them and explained 95% of the
    2026 margin, which was too good to be true: an inside 50 either ends in a scoring shot
    or is rebounded by the opponent, so inside-50 differential + rebound-50 differential
    largely reconstructs the scoring-shot differential (r = 0.82). That is the scoreboard
    restated, not an explanation of it.
  - Train on 2019-2025, test on the unseen 2026 season (a time-based split, never random).
  - 2020 is excluded from training: shortened 16-minute quarters deflate every count
    and every margin, so it is not comparable with other seasons.
  - Ridge penalty chosen by leave-one-season-out cross-validation within training years.
  - Compared against two baselines: home advantage only, and inside-50 differential only
    (the classic rule of thumb).

Run from the repo root:  python -m src.margin_model
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data/processed/team_matches.csv")
OUT_DIR = Path("outputs")
FIG = OUT_DIR / "figures" / "margin_model.png"

# Main model: an interpretable set with no overlapping counts. Kicks, handballs and
# marks are left out because they largely duplicate possessions (kicks + handballs is
# roughly total possessions; most marks are uncontested possessions), which makes the
# coefficients trade off against each other and flip sign.
FEATURES = {
    "diff_inside_50s": "Inside 50s",
    "diff_marks_inside_50": "Marks inside 50",
    "diff_contested_possessions": "Contested possessions",
    "diff_uncontested_possessions": "Uncontested possessions",
    "diff_clearances": "Clearances",
    "diff_tackles": "Tackles",
    "diff_contested_marks": "Contested marks",
    "diff_one_percenters": "One percenters",
    "diff_hit_outs": "Hit-outs",
    "diff_clangers": "Clangers",
    "diff_frees_for": "Free kick differential",
}
# Comparison model: every process stat. Slightly more accurate, much harder to read.
FULL_FEATURES = list(FEATURES) + ["diff_kicks", "diff_handballs", "diff_marks", "diff_bounces"]
TRAIN_SEASONS = [2019, 2021, 2022, 2023, 2024, 2025]
TEST_SEASON = 2026


def load_games() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    return df[df["is_home"] == 1].reset_index(drop=True)  # one row per game


def fit_and_evaluate():
    games = load_games()
    train = games[games["season"].isin(TRAIN_SEASONS)]
    test = games[games["season"] == TEST_SEASON]
    X_cols = list(FEATURES)
    y_tr, y_te = train["margin"], test["margin"]

    results = []

    # Baseline 1: home advantage only (predict the average home margin).
    home_adv = y_tr.mean()
    pred = np.full(len(test), home_adv)
    results.append(("Home advantage only", mean_absolute_error(y_te, pred), r2_score(y_te, pred)))

    # Baseline 2: inside-50 differential only.
    lin = LinearRegression().fit(train[["diff_inside_50s"]], y_tr)
    pred = lin.predict(test[["diff_inside_50s"]])
    results.append(("Inside-50 differential only", mean_absolute_error(y_te, pred), r2_score(y_te, pred)))

    def ridge_fit(cols):
        cv = list(LeaveOneGroupOut().split(train[cols], y_tr, groups=train["season"]))
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 40), cv=cv))
        return model.fit(train[cols], y_tr)

    full = ridge_fit(FULL_FEATURES)
    pred = full.predict(test[FULL_FEATURES])
    results.append(("Ridge, all process stats (comparison)", mean_absolute_error(y_te, pred), r2_score(y_te, pred)))

    # Main model: the interpretable set.
    ridge = ridge_fit(X_cols)
    test_pred = ridge.predict(test[X_cols])
    results.append(("Ridge, interpretable set (main model)", mean_absolute_error(y_te, test_pred), r2_score(y_te, test_pred)))

    scores = pd.DataFrame(results, columns=["model", "mae_2026", "r2_2026"])

    # Coefficients: margin points per +1 standard deviation of each differential,
    # plus points per +1 raw unit, with the range from refitting each season separately.
    scaler, rcv = ridge.named_steps["standardscaler"], ridge.named_steps["ridgecv"]
    coef = pd.DataFrame({
        "feature": X_cols,
        "label": [FEATURES[c] for c in X_cols],
        "pts_per_sd": rcv.coef_,
        "pts_per_unit": rcv.coef_ / scaler.scale_,
        "sd": scaler.scale_,
    })
    per_season = []
    for s in TRAIN_SEASONS:
        d = train[train["season"] == s]
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=[rcv.alpha_])).fit(d[X_cols], d["margin"])
        per_season.append(pd.Series(m.named_steps["ridgecv"].coef_, index=X_cols, name=s))
    ps = pd.concat(per_season, axis=1)
    coef["season_min"] = ps.min(axis=1).to_numpy()
    coef["season_max"] = ps.max(axis=1).to_numpy()
    coef = coef.sort_values("pts_per_sd", ascending=False).reset_index(drop=True)

    preds = test[["date", "team", "opponent", "margin"]].assign(predicted=test_pred.round(1))
    return scores, coef, preds, rcv.alpha_, home_adv


def plot(coef: pd.DataFrame, scores: pd.DataFrame) -> None:
    ink, muted, grid, pos, neg = "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834"
    c = coef.iloc[::-1]
    y = np.arange(len(c))
    colors = np.where(c["pts_per_sd"] >= 0, pos, neg)
    fig, ax = plt.subplots(figsize=(8, 7.2), dpi=200)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    ax.hlines(y, c["season_min"], c["season_max"], color=colors, lw=2, alpha=0.35)
    ax.scatter(c["pts_per_sd"], y, s=60, color=colors, zorder=3, edgecolor="#fcfcfb", linewidth=2)
    for yi, (_, r) in zip(y, c.iterrows()):
        if r["pts_per_sd"] >= 0:
            ax.text(max(r["season_max"], r["pts_per_sd"]) + 0.6, yi, f"{r['pts_per_sd']:+.1f}",
                    va="center", ha="left", fontsize=8.5, color=muted)
        else:
            ax.text(min(r["season_min"], r["pts_per_sd"]) - 0.6, yi, f"{r['pts_per_sd']:+.1f}",
                    va="center", ha="right", fontsize=8.5, color=muted)
    ax.set_yticks(y, c["label"], fontsize=9.5, color=ink)
    ax.axvline(0, color=muted, lw=0.8)
    ax.grid(axis="x", color=grid, lw=0.8); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", colors=muted, labelsize=8.5)
    ax.set_xlabel("Change in final margin (points) for a +1 standard deviation edge in that stat, "
                  "holding the others fixed", fontsize=8.5, color=muted)

    ridge_row = scores.iloc[-1]; i50_row = scores.iloc[1]
    fig.text(0.02, 0.975, "What explains an AFL margin, once you know how the game played out?",
             fontsize=13, weight="bold", color=ink, ha="left")
    fig.text(0.02, 0.945,
             f"Ridge regression on stat differentials, trained 2019–2025, tested on 2026: "
             f"typical error {ridge_row.mae_2026:.1f} pts (inside 50s alone: {i50_row.mae_2026:.1f}).",
             fontsize=9, color=muted, ha="left")
    ax.set_xlim(min(c["season_min"].min(), 0) - 3, c["season_max"].max() + 3)
    fig.text(0.02, 0.03, "Dot = trained on 2019–2025; line = range when refitted season by season. "
             "Scoring stats and rebound 50s excluded (they restate the score).",
             fontsize=7.5, color=muted, ha="left")
    fig.text(0.02, 0.008, "Data: AFL Tables via fitzRoy. Analysis: Stefan Karydis · "
             "github.com/stefank10/afl-team-analytics", fontsize=7.5, color=muted, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    scores, coef, preds, alpha, home_adv = fit_and_evaluate()
    scores.to_csv(OUT_DIR / "margin_model_scores.csv", index=False)
    coef.to_csv(OUT_DIR / "margin_model_coefficients.csv", index=False)
    preds.to_csv(OUT_DIR / "margin_model_2026_predictions.csv", index=False)
    plot(coef, scores)
    print(f"Average home margin in training: {home_adv:.1f} pts; ridge alpha = {alpha:.2f}\n")
    print(scores.round(2).to_string(index=False), "\n")
    print(coef.round(2).to_string(index=False))
