"""Download the data, build features, train the models. Run this before the dashboard.

    python run_pipeline.py            # normal run (refreshes the current season)
    python run_pipeline.py --offline  # use already-downloaded files only
"""
from __future__ import annotations

import sys

import pandas as pd

from src import config
from src.data_loader import load_matches
from src.features import build_feature_table, current_team_state
from src.match_predictor import train
from src.team_styles import cluster_styles, team_season_table


def main(offline: bool = False) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    line = "=" * 60
    print(line + "\nPREMIER LEAGUE ML - PIPELINE\n" + line)

    print("\nStep 1/4: Loading match data")
    matches = load_matches(download=not offline)
    matches.to_csv(config.PROCESSED_DIR / "matches.csv", index=False)
    print(f"  {len(matches):,} matches across {matches['season'].nunique()} seasons "
          f"({matches['season'].iloc[0]} to {matches['season'].iloc[-1]})")

    print("\nStep 2/4: Building features (Elo ratings + recent form)")
    table, elo_hist, log = build_feature_table(matches)
    table.to_csv(config.PROCESSED_DIR / "features.csv", index=False)
    elo_hist.to_csv(config.PROCESSED_DIR / "elo_history.csv", index=False)
    log.to_csv(config.PROCESSED_DIR / "team_match_log.csv", index=False)
    current_team_state(elo_hist, log).to_csv(config.PROCESSED_DIR / "team_state.csv", index=False)
    print("  done")

    print("\nStep 3/4: Training match outcome models")
    summary = train(table)
    print(f"  test seasons: {', '.join(summary['seasons']['test'])}")
    print(pd.DataFrame(summary["test_metrics"])
          [["model", "accuracy", "log_loss", "rps"]].round(3).to_string(index=False))

    print("\nStep 4/4: Clustering team playing styles")
    teams, info = cluster_styles(team_season_table(log))
    teams.to_csv(config.PROCESSED_DIR / "team_seasons.csv", index=False)
    print(f"  {len(teams)} team-seasons in {teams['cluster'].nunique()} style groups "
          f"(silhouette {info['silhouette']:.2f})")

    print("\n" + line + "\nDONE. Start the dashboard with:\n  streamlit run dashboard/app.py\n" + line)


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
