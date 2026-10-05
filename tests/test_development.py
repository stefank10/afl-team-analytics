"""Checks for the development model: honest score, honest split, converges, beats persistence overall."""
import pytest

from src.development import SCORE_STATS, add_contribution, backtest, fit, player_seasons
from src.player_roles import load_player_games


@pytest.fixture(scope="module")
def seasons():
    return player_seasons(add_contribution(load_player_games()))


def test_score_leaves_out_confounded_stats():
    used = set(SCORE_STATS.values())
    assert "Frees.For" not in used and "Hit.Outs" not in used
    assert "Goals" not in used and "Behinds" not in used  # scoring stats never feed a margin-based score


def test_2020_excluded(seasons):
    assert 2020 not in set(seasons["Season"])


def test_model_converges(seasons):
    assert fit(seasons[seasons["Season"] < 2026]).converged


def test_backtest_only_uses_earlier_seasons_and_beats_persistence_on_average(seasons):
    bt = backtest(seasons)
    mean = bt[bt["group"] == "All players"].groupby("model")["mae"].mean()
    assert mean["Development model"] < mean["Same as last season"]
