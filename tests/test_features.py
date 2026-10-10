"""Checks that features only use information from before each match.

Run with:  python -m pytest
"""
import numpy as np
import pandas as pd

from src.features import build_feature_table
from src.match_predictor import score


def tiny_league() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    teams = ["A", "B", "C", "D"]
    rows, day = [], pd.Timestamp("2020-08-15")
    for rnd in range(6):
        for i in range(0, 4, 2):
            h, a = np.roll(teams, rnd)[i], np.roll(teams, rnd)[i + 1]
            hg, ag = rng.integers(0, 4, 2)
            rows.append({"date": day, "home_team": h, "away_team": a,
                         "home_goals": hg, "away_goals": ag,
                         "result": "H" if hg > ag else "D" if hg == ag else "A",
                         **{f"{s}_{x}": rng.integers(1, 20) for s in ("home", "away")
                            for x in ("shots", "sot", "corners", "fouls", "yellows", "reds")},
                         "season": "2020-21", "season_start": 2020})
        day += pd.Timedelta(days=7)
    df = pd.DataFrame(rows)
    df["match_id"] = range(len(df))
    return df


def test_first_match_has_no_form_and_neutral_elo():
    table, _, _ = build_feature_table(tiny_league())
    first = table.iloc[0]
    assert np.isnan(first["home_goals_for"])
    assert first["home_elo"] == first["away_elo"]


def test_changing_a_result_does_not_change_that_matchs_features():
    m = tiny_league()
    before, _, _ = build_feature_table(m)
    m.loc[5, ["home_goals", "away_goals", "home_shots"]] = [9, 0, 40]
    after, _, _ = build_feature_table(m)
    cols = [c for c in before.columns if c.startswith(("home_", "away_", "elo"))
            and c not in ("home_team", "away_team", "home_goals", "away_goals")]
    pd.testing.assert_frame_equal(before.loc[:5, cols], after.loc[:5, cols])
    assert not before.loc[6:, cols].equals(after.loc[6:, cols])


def test_perfect_prediction_scores():
    s = score(np.array([[1.0, 0, 0], [0, 0, 1.0]]), pd.Series(["H", "A"]))
    assert s["accuracy"] == 1.0 and s["rps"] < 1e-9
