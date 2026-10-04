"""Build a tidy team-match table from raw AFL Tables player stats and results.

Output: data/processed/team_matches.csv
One row per team per game, with:
  - match info (season, round, date, venue, home/away, opponent)
  - the team's totals for each stat, the opponent's totals (opp_*), and the
    differential (diff_* = team minus opponent)
  - points for, points against and margin (from the official results file)

Run from the repo root:  python -m src.build_team_matches
"""
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed/team_matches.csv")

# Team stats summed from player rows. Goals and behinds are kept for checks only:
# they are part of the outcome, so they never go into a model of the margin.
STATS = [
    "Kicks", "Handballs", "Disposals", "Marks", "Contested.Marks", "Marks.Inside.50",
    "Contested.Possessions", "Uncontested.Possessions", "Clearances", "Inside.50s",
    "Rebounds", "Tackles", "One.Percenters", "Hit.Outs", "Clangers", "Frees.For",
    "Frees.Against", "Bounces", "Goal.Assists", "Goals", "Behinds",
]

# The results file uses older/short names for two clubs.
RESULTS_TEAM_NAMES = {"Footscray": "Western Bulldogs", "GWS": "Greater Western Sydney"}


def snake(name: str) -> str:
    return name.lower().replace(".", "_")


def load_player_stats(raw_dir: Path = RAW) -> pd.DataFrame:
    files = sorted(raw_dir.glob("player_stats_*.csv"))
    return pd.concat((pd.read_csv(f, low_memory=False) for f in files), ignore_index=True)


def load_results(raw_dir: Path = RAW) -> pd.DataFrame:
    files = sorted(raw_dir.glob("results_*.csv"))
    res = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    for col in ["Home.Team", "Away.Team"]:
        res[col] = res[col].replace(RESULTS_TEAM_NAMES)
    return res


def results_long(res: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game from the official results."""
    home = res.assign(team=res["Home.Team"], opponent=res["Away.Team"], is_home=1,
                      goals_official=res["Home.Goals"], behinds_official=res["Home.Behinds"],
                      points_for=res["Home.Points"], points_against=res["Away.Points"])
    away = res.assign(team=res["Away.Team"], opponent=res["Home.Team"], is_home=0,
                      goals_official=res["Away.Goals"], behinds_official=res["Away.Behinds"],
                      points_for=res["Away.Points"], points_against=res["Home.Points"])
    cols = ["Season", "Date", "Round", "Round.Type", "Round.Number", "Venue", "team",
            "opponent", "is_home", "goals_official", "behinds_official", "points_for",
            "points_against"]
    out = pd.concat([home[cols], away[cols]], ignore_index=True)
    out["margin"] = out["points_for"] - out["points_against"]
    return out


def build(raw_dir: Path = RAW) -> pd.DataFrame:
    ps = load_player_stats(raw_dir)

    team = (ps.groupby(["Season", "Date", "Playing.for"], as_index=False)
              .agg(n_players=("Player", "size"), **{snake(s): (s, "sum") for s in STATS}))
    team = team.rename(columns={"Playing.for": "team"})

    res = results_long(load_results(raw_dir))
    df = res.merge(team, on=["Season", "Date", "team"], how="inner", validate="one_to_one")

    # Attach the opponent's stats and differentials.
    stat_cols = [snake(s) for s in STATS]
    opp = df[["Season", "Date", "team"] + stat_cols].rename(
        columns={"team": "opponent", **{c: f"opp_{c}" for c in stat_cols}})
    df = df.merge(opp, on=["Season", "Date", "opponent"], how="left", validate="one_to_one")
    for c in stat_cols:
        df[f"diff_{c}"] = df[c] - df[f"opp_{c}"]

    df.columns = [snake(c) for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df["is_final"] = (df["round_type"] == "Finals").astype(int)
    return df.sort_values(["date", "team"]).reset_index(drop=True)


if __name__ == "__main__":
    out = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {len(out)} team-games, {out['season'].nunique()} seasons, "
          f"{len(out.columns)} columns")
