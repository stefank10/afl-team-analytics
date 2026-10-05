"""Checks for player roles: sensible profiles, named roles, stable roles, valid similarity."""
import pytest

from src.player_roles import (MIN_GAMES, SEASON, features, fit_roles, load_player_games, player_seasons,
                              role_stability, similarity_table)


@pytest.fixture(scope="module")
def model():
    prof = player_seasons(load_player_games())
    z = features(prof)
    roles, centres, _ = fit_roles(z)
    return prof, z, roles, centres


def test_minimum_games(model):
    prof, *_ = model
    assert (prof["games"] >= MIN_GAMES).all()


def test_every_role_named_once(model):
    _, _, roles, centres = model
    assert centres.index.is_unique and len(centres) == 8
    assert roles.isin(centres.index).all()


def test_ruck_role_is_about_hit_outs(model):
    _, _, _, centres = model
    assert centres["Hit.Outs"].idxmax() == "Ruck"


def test_roles_are_stable_well_above_chance(model):
    *_, roles, _ = model
    same, chance = role_stability(roles)
    assert same > 3 * chance


def test_similarity_excludes_self_and_is_bounded(model):
    _, z, _, _ = model
    sims = similarity_table(z.loc[SEASON].iloc[:50], top=5)
    for name, matches in sims.items():
        assert all(m["name"] != name for m in matches)
        assert all(-1 <= m["similarity"] <= 1 for m in matches)
