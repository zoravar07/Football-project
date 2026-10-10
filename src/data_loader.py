"""Download Premier League results from football-data.co.uk and tidy them into one table.

Run on its own with:  python -m src.data_loader
"""
from __future__ import annotations

import io
import ssl
import urllib.request

import pandas as pd

from src import config

# Columns we keep, renamed to something readable.
COLUMNS = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_sot",
    "AST": "away_sot",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
}

# Bookmaker odds, in order of preference. Bet365 has the longest history;
# the market average columns change name over the years.
ODDS_SOURCES = [
    ("B365H", "B365D", "B365A"),
    ("AvgH", "AvgD", "AvgA"),
    ("BbAvH", "BbAvD", "BbAvA"),
]


def _ssl_context() -> ssl.SSLContext:
    # Python from python.org on a Mac often has no certificates installed;
    # certifi ships its own, so use them when available.
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (pl-ml-project)"})
    with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp:
        return resp.read()


def download_seasons(force: bool = False) -> list[int]:
    """Save one raw CSV per season into data/raw/. Returns the seasons available locally.

    Finished seasons are only downloaded once. The current season is refreshed every
    run so new results come through.
    """
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    current = config.current_season_start()
    available = []
    for start in config.all_season_starts():
        path = config.RAW_DIR / f"E0_{config.season_code(start)}.csv"
        needs_fetch = force or not path.exists() or start == current
        if needs_fetch:
            url = config.URL_TEMPLATE.format(code=config.season_code(start))
            try:
                path.write_bytes(_download(url))
                print(f"  downloaded {config.season_label(start)}")
            except Exception as exc:  # network down, season not published yet, etc.
                if path.exists():
                    print(f"  couldn't refresh {config.season_label(start)} ({exc}); using saved copy")
                else:
                    print(f"  skipped {config.season_label(start)}: {exc}")
                    continue
        available.append(start)
    return available


def _read_raw(path) -> pd.DataFrame:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Couldn't read {path}")


def _clean_season(df: pd.DataFrame, start: int) -> pd.DataFrame:
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    out = df[[c for c in COLUMNS if c in df.columns]].rename(columns=COLUMNS)

    for home_col, draw_col, away_col in ODDS_SOURCES:
        if {home_col, draw_col, away_col} <= set(df.columns):
            out["odds_home"] = pd.to_numeric(df[home_col], errors="coerce")
            out["odds_draw"] = pd.to_numeric(df[draw_col], errors="coerce")
            out["odds_away"] = pd.to_numeric(df[away_col], errors="coerce")
            break

    # Dates are day-first; older files use 2-digit years.
    d = out["date"].astype(str).str.strip()
    out["date"] = pd.to_datetime(d.where(d.str.len() != 8), format="%d/%m/%Y", errors="coerce") \
        .fillna(pd.to_datetime(d.where(d.str.len() == 8), format="%d/%m/%y", errors="coerce"))
    out["season"] = config.season_label(start)
    out["season_start"] = start
    return out


def load_matches(download: bool = True) -> pd.DataFrame:
    """Return every Premier League match since 2005-06 as one tidy DataFrame."""
    if download:
        print("Fetching data from football-data.co.uk ...")
        seasons = download_seasons()
    else:
        seasons = [s for s in config.all_season_starts()
                   if (config.RAW_DIR / f"E0_{config.season_code(s)}.csv").exists()]
    if not seasons:
        raise RuntimeError(
            "No data found. Check your internet connection and run `python run_pipeline.py` again."
        )

    frames = [_clean_season(_read_raw(config.RAW_DIR / f"E0_{config.season_code(s)}.csv"), s)
              for s in seasons]
    matches = pd.concat(frames, ignore_index=True)

    numeric = [c for c in matches.columns
               if c not in ("date", "home_team", "away_team", "result", "season")]
    matches[numeric] = matches[numeric].apply(pd.to_numeric, errors="coerce")
    matches["home_team"] = matches["home_team"].str.strip()
    matches["away_team"] = matches["away_team"].str.strip()
    matches = matches.dropna(subset=["date"]).sort_values("date", kind="stable").reset_index(drop=True)
    matches["match_id"] = range(len(matches))
    return matches


if __name__ == "__main__":
    m = load_matches()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    m.to_csv(config.PROCESSED_DIR / "matches.csv", index=False)
    print(f"{len(m):,} matches, {m['season'].nunique()} seasons -> data/processed/matches.csv")
