"""Checks for the style map: complete seasons, sensible ratios, within-season comparison."""
from src.style_map import fit_style_pca, style_ratios


def test_eighteen_clubs_every_season():
    r = style_ratios()
    assert (r.groupby(level="season").size() == 18).all()


def test_shares_are_proportions():
    r = style_ratios()
    for col in ["kick_share", "uncontested_share", "contested_mark_share"]:
        assert r[col].between(0, 1).all(), col


def test_map_compares_within_season():
    # Each season is centred on its own league average, so league-wide shifts over
    # time can't masquerade as differences between clubs.
    scores, _, _ = fit_style_pca(style_ratios())
    season_means = scores.groupby(level="season").mean().abs()
    assert (season_means < 1e-9).all().all()
