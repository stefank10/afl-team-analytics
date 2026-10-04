"""Guard rails for the margin model: no leakage, honest split, beats the baselines."""
from src.margin_model import FEATURES, FULL_FEATURES, TEST_SEASON, TRAIN_SEASONS, fit_and_evaluate

# Stats that are the score, or restate it, must never be features.
FORBIDDEN = ["goals", "behinds", "points", "margin", "goal_assists", "rebounds"]


def test_no_scoring_features():
    for col in set(FEATURES) | set(FULL_FEATURES):
        assert not any(f in col for f in FORBIDDEN), f"leaky feature: {col}"


def test_time_based_split():
    assert TEST_SEASON not in TRAIN_SEASONS
    assert max(TRAIN_SEASONS) < TEST_SEASON
    assert 2020 not in TRAIN_SEASONS  # shortened quarters


def test_main_model_beats_baselines():
    scores, *_ = fit_and_evaluate()
    mae = dict(zip(scores["model"], scores["mae_2026"]))
    main = mae["Ridge, interpretable set (main model)"]
    assert main < mae["Inside-50 differential only"]
    assert main < mae["Home advantage only"]
