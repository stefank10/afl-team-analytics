"""Who plays like whom? Player roles and a similarity finder from public box-score stats.

Method:
  1. Player-season profiles: each player's totals divided by "full games" (the sum of
     time on ground / 100), so a sub or a rotated midfielder isn't judged on raw counts.
     Players need at least 8 games in a season.
  2. Features: contested and uncontested possessions, marks, contested marks, marks inside
     50, inside 50s, rebound 50s, one percenters, tackles, clearances, goal assists,
     hit-outs, goals, bounces (square-rooted, as they're very skewed) and kick share.
     Each is standardised within season, so league-wide trends don't move players.
  3. Roles: k-means with 8 clusters on all player-seasons 2019-2026. Each cluster is named
     from its standout stats by a fixed rule, not by hand.
  4. Similarity: cosine similarity between 2026 profiles, with the shared strengths that
     drive each match listed as reasons.
  5. Validation: how often a player keeps the same role from one season to the next,
     compared with chance.

AFL Tables doesn't record positions, so roles come from what players actually do.
Run from the repo root:  python -m src.player_roles
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RAW = Path("data/raw")
OUT = Path("outputs")
SITE_JSON = OUT / "players_2026.json"
FIG = OUT / "figures" / "player_roles.png"
SEASON = 2026
MIN_GAMES = 8
N_ROLES = 8

RATE_STATS = ["Contested.Possessions", "Uncontested.Possessions", "Marks", "Contested.Marks",
              "Marks.Inside.50", "Inside.50s", "Rebounds", "One.Percenters", "Tackles",
              "Clearances", "Goal.Assists"]
SQRT_STATS = ["Hit.Outs", "Goals", "Bounces"]
LABELS = {
    "Contested.Possessions": "Contested possessions", "Uncontested.Possessions": "Uncontested possessions",
    "Marks": "Marks", "Contested.Marks": "Contested marks", "Marks.Inside.50": "Marks inside 50",
    "Inside.50s": "Inside 50s", "Rebounds": "Rebound 50s", "One.Percenters": "One percenters",
    "Tackles": "Tackles", "Clearances": "Clearances", "Goal.Assists": "Goal assists",
    "Hit.Outs": "Hit-outs", "Goals": "Goals", "Bounces": "Bounces", "kick_share": "Kick share",
}


def load_player_games(raw_dir: Path = RAW) -> pd.DataFrame:
    ps = pd.concat((pd.read_csv(f, low_memory=False) for f in sorted(raw_dir.glob("player_stats_*.csv"))),
                   ignore_index=True)
    ps["name"] = ps["First.name"].str.strip() + " " + ps["Surname"].str.strip()
    ps["full_games"] = ps["Time.on.Ground"] / 100
    return ps


def player_seasons(ps: pd.DataFrame) -> pd.DataFrame:
    """Per-full-game rates for every player-season with enough games."""
    stats = RATE_STATS + SQRT_STATS + ["Kicks", "Handballs"]
    g = ps.groupby(["Season", "name"]).agg(
        games=("full_games", "size"), full_games=("full_games", "sum"),
        team=("Playing.for", lambda s: s.mode().iat[0]), age=("Age", "max"),
        **{s: (s, "sum") for s in stats})
    g = g[g["games"] >= MIN_GAMES]
    rates = g[stats].div(g["full_games"], axis=0)
    out = g[["games", "team", "age"]].copy()
    for s in RATE_STATS + SQRT_STATS:
        out[s] = rates[s]
    out["kick_share"] = rates["Kicks"] / (rates["Kicks"] + rates["Handballs"])
    return out


def features(prof: pd.DataFrame) -> pd.DataFrame:
    f = prof[RATE_STATS].copy()
    for s in SQRT_STATS:
        f[s] = np.sqrt(prof[s])
    f["kick_share"] = prof["kick_share"]
    return f.groupby(level="Season").transform(lambda c: (c - c.mean()) / c.std())


def name_roles(centres: pd.DataFrame) -> dict:
    """Name clusters by a fixed rule on their centroids, taking the clearest role first."""
    rules = [("Ruck", "Hit.Outs"), ("Key forward", "Marks.Inside.50"),
             ("Key defender", "One.Percenters"), ("Rebounding defender", "Rebounds"),
             ("Inside midfielder", "Clearances"), ("Small forward", "Goal.Assists"),
             ("Outside midfielder", "Uncontested.Possessions")]
    names, left = {}, list(centres.index)
    for role, stat in rules:
        best = centres.loc[left, stat].idxmax()
        names[best] = role
        left.remove(best)
    for c in left:
        names[c] = "Pressure / role player"
    return names


def fit_roles(z: pd.DataFrame):
    km = KMeans(N_ROLES, n_init=30, random_state=1).fit(z)
    centres = pd.DataFrame(km.cluster_centers_, columns=z.columns)
    names = name_roles(centres)
    roles = pd.Series(km.labels_, index=z.index).map(names)
    centres.index = [names[i] for i in centres.index]
    sil = silhouette_score(z, km.labels_, sample_size=4000, random_state=1)
    return roles, centres, sil


def role_stability(roles: pd.Series) -> tuple[float, float]:
    """Share of players with the same role in consecutive seasons, vs chance."""
    wide = roles.unstack("Season")
    seasons = sorted(wide.columns)
    same, chance = [], []
    for a, b in zip(seasons, seasons[1:]):
        both = wide[[a, b]].dropna()
        same.append((both[a] == both[b]).mean())
        p = both[a].value_counts(normalize=True)
        q = both[b].value_counts(normalize=True)
        chance.append(float((p * q.reindex(p.index).fillna(0)).sum()))
    return float(np.mean(same)), float(np.mean(chance))


def similarity_table(z_season: pd.DataFrame, top: int = 10) -> dict:
    """Cosine similarity between players, with the shared strengths behind each match."""
    x = z_season.to_numpy()
    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    sim = unit @ unit.T
    names = z_season.index.to_list()
    out = {}
    for i, n in enumerate(names):
        order = np.argsort(-sim[i])
        matches = []
        for j in order[1:top + 1]:
            shared = x[i] * x[j]
            both_high = [(z_season.columns[k], shared[k]) for k in np.argsort(-shared)
                         if x[i][k] > 0.5 and x[j][k] > 0.5][:3]
            matches.append({"name": names[j], "similarity": round(float(sim[i][j]), 2),
                            "shared": [LABELS[c] for c, _ in both_high]})
        out[n] = matches
    return out


def plot_roles(centres: pd.DataFrame) -> None:
    order = ["Ruck", "Key forward", "Small forward", "Inside midfielder", "Outside midfielder",
             "Rebounding defender", "Key defender", "Pressure / role player"]
    cols = ["Hit.Outs", "Marks.Inside.50", "Goals", "Goal.Assists", "Clearances", "Contested.Possessions",
            "Tackles", "Inside.50s", "Uncontested.Possessions", "Bounces", "Marks", "Rebounds",
            "kick_share", "Contested.Marks", "One.Percenters"]
    c = centres.loc[order, cols]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f4f3ef", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.imshow(c.to_numpy(), cmap=cmap, vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(cols)), [LABELS[k] for k in cols], rotation=40, ha="right", fontsize=8.5, color="#0b0b0b")
    ax.set_yticks(range(len(order)), order, fontsize=9.5, color="#0b0b0b")
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            v = c.iat[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if abs(v) > 1.6 else "#0b0b0b")
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True); ax.set_yticks(np.arange(-.5, len(order)), minor=True)
    ax.grid(which="minor", color="#fcfcfb", lw=2); ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.01, 0.965, "Eight player roles, found from what players do", fontsize=13.5, weight="bold",
             color="#0b0b0b", ha="left")
    fig.text(0.01, 0.925, "Each cell: how far a role's typical player sits above (blue) or below (orange) the "
             "league average, in standard deviations, per full game.", fontsize=9, color="#52514e", ha="left")
    fig.text(0.01, 0.01, "Player-seasons 2019–2026, minimum 8 games. Data: AFL Tables via fitzRoy. "
             "Analysis: Stefan Karydis", fontsize=7.5, color="#52514e", ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, facecolor=fig.get_facecolor())


def site_data(prof: pd.DataFrame, z: pd.DataFrame, roles: pd.Series, sims: dict) -> dict:
    """Compact JSON for the web tool: players referenced by index to keep the file small."""
    p = prof.loc[SEASON].sort_index()
    cols = RATE_STATS + SQRT_STATS + ["kick_share"]
    pct = p[cols].rank(pct=True, method="min")  # ties (e.g. zero hit-outs) rank at the bottom
    names = p.index.to_list()
    idx = {n: i for i, n in enumerate(names)}
    feats = [LABELS[c] for c in cols]
    fidx = {f: i for i, f in enumerate(feats)}
    players = []
    for n in names:
        players.append([
            n, p.at[n, "team"], roles.loc[(SEASON, n)], int(p.at[n, "games"]),
            [int(round(100 * pct.at[n, c])) for c in cols],
            [[idx[m["name"]], m["similarity"], [fidx[f] for f in m["shared"]]] for m in sims[n]],
        ])
    return {"season": SEASON, "minGames": MIN_GAMES, "features": feats,
            "fields": ["name", "team", "role", "games", "percentiles", "similar[idx, similarity, shared features]"],
            "players": players}


if __name__ == "__main__":
    prof = player_seasons(load_player_games())
    z = features(prof)
    roles, centres, sil = fit_roles(z)
    same, chance = role_stability(roles)
    sims = similarity_table(z.loc[SEASON])
    OUT.mkdir(exist_ok=True)
    prof.assign(role=roles).round(3).to_csv(OUT / "player_roles.csv")
    centres.round(2).to_csv(OUT / "role_profiles.csv")
    SITE_JSON.write_text(json.dumps(site_data(prof, z, roles, sims), separators=(",", ":")))
    plot_roles(centres)
    print(f"{len(prof)} player-seasons; {len(prof.loc[SEASON])} in {SEASON}")
    print(f"Silhouette {sil:.2f}; same role next season {same:.0%} vs {chance:.0%} by chance")
    print(roles.loc[SEASON].value_counts().to_string())
