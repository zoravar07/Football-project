"""Turn raw results into pre-match features.

Every feature for a match is built only from matches played *before* it, so the
model never sees the result it's trying to predict.

Two kinds of feature:
  - Elo ratings: a running strength score per team, updated after every game.
  - Recent form: each team's average stats over its last few matches.

Run on its own with:  python -m src.features
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

FORM_STATS = ["goals_for", "goals_against", "shots_for", "shots_against",
              "sot_for", "sot_against", "corners_for", "points"]

FEATURES = (["elo_diff", "home_elo", "away_elo"]
            + [f"home_{s}" for s in FORM_STATS]
            + [f"away_{s}" for s in FORM_STATS])

RESULT_LABELS = ["H", "D", "A"]   # home win, draw, away win


# ---------------------------------------------------------------- Elo ratings

def _goal_multiplier(goal_diff: int) -> float:
    """Bigger wins move ratings more (World Football Elo style)."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8


def elo_expected_home(home_elo: float, away_elo: float) -> float:
    return 1 / (1 + 10 ** ((away_elo - (home_elo + config.ELO_HOME_ADV)) / 400))


def compute_elo(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk through matches in date order, rating each team before every game.

    Returns (pre-match ratings per match, long table of every team's rating over time).
    """
    ratings: dict[str, float] = {}
    last_season_teams: set[str] = set()
    season_teams: set[str] = set()
    current_season = None
    pre_home, pre_away, history = [], [], []

    for row in matches.itertuples(index=False):
        if row.season_start != current_season:
            if current_season is not None:
                # Summer break: pull everyone a little back towards the league average.
                last_season_teams = season_teams
                mean = np.mean([ratings[t] for t in last_season_teams])
                for t in last_season_teams:
                    ratings[t] += config.ELO_SEASON_REGRESS * (mean - ratings[t])
            current_season = row.season_start
            season_teams = set()

        for team in (row.home_team, row.away_team):
            if team not in season_teams:
                season_teams.add(team)
                if team not in last_season_teams:
                    # First season in the data: everyone starts level.
                    # After that, anyone new to the league has just been promoted.
                    ratings[team] = config.ELO_START if not last_season_teams else config.ELO_PROMOTED

        h, a = row.home_team, row.away_team
        rh, ra = ratings[h], ratings[a]
        pre_home.append(rh)
        pre_away.append(ra)

        if row.home_goals > row.away_goals:
            actual = 1.0
        elif row.home_goals == row.away_goals:
            actual = 0.5
        else:
            actual = 0.0
        change = (config.ELO_K * _goal_multiplier(int(row.home_goals - row.away_goals))
                  * (actual - elo_expected_home(rh, ra)))
        ratings[h] = rh + change
        ratings[a] = ra - change
        history.append((row.date, row.season, h, ratings[h]))
        history.append((row.date, row.season, a, ratings[a]))

    pre = pd.DataFrame({"match_id": matches["match_id"].values,
                        "home_elo": pre_home, "away_elo": pre_away})
    hist = pd.DataFrame(history, columns=["date", "season", "team", "elo"])
    return pre, hist


# ---------------------------------------------------------------- recent form

def team_match_log(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match (each match appears twice: home view and away view)."""
    def side(home: bool) -> pd.DataFrame:
        us, them = ("home", "away") if home else ("away", "home")
        pts = np.where(matches[f"{us}_goals"] > matches[f"{them}_goals"], 3,
                       np.where(matches[f"{us}_goals"] == matches[f"{them}_goals"], 1, 0))
        return pd.DataFrame({
            "match_id": matches["match_id"],
            "date": matches["date"],
            "season": matches["season"],
            "team": matches[f"{us}_team"],
            "opponent": matches[f"{them}_team"],
            "is_home": home,
            "goals_for": matches[f"{us}_goals"],
            "goals_against": matches[f"{them}_goals"],
            "shots_for": matches[f"{us}_shots"],
            "shots_against": matches[f"{them}_shots"],
            "sot_for": matches[f"{us}_sot"],
            "sot_against": matches[f"{them}_sot"],
            "corners_for": matches[f"{us}_corners"],
            "fouls": matches[f"{us}_fouls"],
            "yellows": matches[f"{us}_yellows"],
            "reds": matches[f"{us}_reds"],
            "points": pts,
        })

    log = pd.concat([side(True), side(False)], ignore_index=True)
    return log.sort_values(["team", "date", "match_id"], kind="stable").reset_index(drop=True)


def add_rolling_form(log: pd.DataFrame) -> pd.DataFrame:
    """Average of each stat over the team's previous N matches (not including this one)."""
    log = log.copy()
    grouped = log.groupby("team", sort=False)[FORM_STATS]
    rolled = grouped.transform(
        lambda s: s.shift(1).rolling(config.ROLLING_WINDOW, min_periods=1).mean())
    for s in FORM_STATS:
        log[f"form_{s}"] = rolled[s]
    return log


# ---------------------------------------------------------------- bookmaker benchmark

def market_probabilities(matches: pd.DataFrame) -> pd.DataFrame:
    """Turn decimal odds into probabilities, removing the bookmaker's margin."""
    inv = pd.DataFrame({
        "mkt_H": 1 / matches["odds_home"],
        "mkt_D": 1 / matches["odds_draw"],
        "mkt_A": 1 / matches["odds_away"],
    })
    inv = inv.div(inv.sum(axis=1), axis=0)
    inv["match_id"] = matches["match_id"].values
    return inv


# ---------------------------------------------------------------- put it together

def build_feature_table(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (one row per match with features + target, Elo history, team match log)."""
    elo_pre, elo_hist = compute_elo(matches)
    log = add_rolling_form(team_match_log(matches))

    form_cols = [f"form_{s}" for s in FORM_STATS]
    home_form = log[log["is_home"]][["match_id"] + form_cols].rename(
        columns={f"form_{s}": f"home_{s}" for s in FORM_STATS})
    away_form = log[~log["is_home"]][["match_id"] + form_cols].rename(
        columns={f"form_{s}": f"away_{s}" for s in FORM_STATS})

    table = (matches[["match_id", "date", "season", "season_start",
                      "home_team", "away_team", "home_goals", "away_goals", "result"]]
             .merge(elo_pre, on="match_id")
             .merge(home_form, on="match_id")
             .merge(away_form, on="match_id"))
    if {"odds_home", "odds_draw", "odds_away"} <= set(matches.columns):
        table = table.merge(market_probabilities(matches), on="match_id", how="left")
    table["elo_diff"] = table["home_elo"] - table["away_elo"]
    return table.sort_values("match_id").reset_index(drop=True), elo_hist, log


def current_team_state(elo_hist: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    """Each team's latest Elo and form, for predicting games that haven't happened yet."""
    latest_elo = elo_hist.groupby("team").tail(1).set_index("team")[["elo", "date", "season"]]
    recent = log.groupby("team", sort=False).tail(config.ROLLING_WINDOW)
    form = recent.groupby("team")[FORM_STATS].mean()
    state = latest_elo.join(form)
    return state.reset_index().rename(columns={"date": "last_match"})


def features_for_fixture(state: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Feature row for a hypothetical match between two teams, using their current state."""
    s = state.set_index("team")
    row = {"home_elo": s.at[home, "elo"], "away_elo": s.at[away, "elo"]}
    row["elo_diff"] = row["home_elo"] - row["away_elo"]
    for stat in FORM_STATS:
        row[f"home_{stat}"] = s.at[home, stat]
        row[f"away_{stat}"] = s.at[away, stat]
    return pd.DataFrame([row])[FEATURES]


if __name__ == "__main__":
    from src.data_loader import load_matches
    table, _, _ = build_feature_table(load_matches(download=False))
    print(table.tail())
