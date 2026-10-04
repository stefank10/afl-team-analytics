"""Integrity checks: the tidy table must agree with the official results before any analysis."""
import pandas as pd
import pytest

from src.build_team_matches import build, load_results


@pytest.fixture(scope="module")
def df():
    return build()


def test_every_official_game_is_present(df):
    games = load_results()
    assert len(df) == 2 * len(games), "each official game should give exactly two team rows"


def test_one_row_per_team_per_game(df):
    assert not df.duplicated(["date", "team"]).any()


def test_full_teams_named(df):
    # 22 players per side before the substitute rule (2021+), 23 after.
    assert df["n_players"].isin([22, 23]).all()


def test_player_goals_match_official_goals(df):
    bad = df[df["goals"] != df["goals_official"]]
    assert bad.empty, f"{len(bad)} team-games where player goals != official goals"


def test_rushed_behinds_are_never_negative(df):
    # Official behinds include rushed behinds, which aren't credited to any player,
    # so official behinds must be >= the sum of players' behinds.
    rushed = df["behinds_official"] - df["behinds"]
    assert (rushed >= 0).all()


def test_points_match_goals_and_behinds(df):
    assert (df["points_for"] == 6 * df["goals_official"] + df["behinds_official"]).all()


def test_margins_are_symmetric(df):
    pair = df.merge(df, left_on=["date", "team"], right_on=["date", "opponent"])
    assert (pair["margin_x"] == -pair["margin_y"]).all()


def test_differentials_are_symmetric(df):
    pair = df.merge(df, left_on=["date", "team"], right_on=["date", "opponent"])
    assert (pair["diff_inside_50s_x"] == -pair["diff_inside_50s_y"]).all()
