"""Shared settings: paths, seasons, and model parameters."""
from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# football-data.co.uk Premier League CSVs, one file per season.
# Season code "2324" = 2023-24.
URL_TEMPLATE = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

FIRST_SEASON_START = 2005   # 2005-06 onwards: full shots/corners/odds coverage
ELO_WARMUP_SEASONS = 2      # Elo needs a couple of seasons to settle; not used for training
N_VALIDATION_SEASONS = 1    # used to pick the model
N_TEST_SEASONS = 2          # held out, only touched once for the final score

ROLLING_WINDOW = 5          # matches of form to average over

# Elo settings
ELO_START = 1500.0
ELO_PROMOTED = 1420.0       # promoted teams start below average
ELO_K = 20.0
ELO_HOME_ADV = 60.0
ELO_SEASON_REGRESS = 0.2    # pull ratings 20% back towards the mean each summer


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def current_season_start(today: date | None = None) -> int:
    """A season starts in August, so before August we're still in last year's season."""
    today = today or date.today()
    return today.year if today.month >= 8 else today.year - 1


def all_season_starts(today: date | None = None) -> list[int]:
    return list(range(FIRST_SEASON_START, current_season_start(today) + 1))
