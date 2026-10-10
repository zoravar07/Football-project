# ⚽ Premier League ML

**Live app: https://premier-league-ml.streamlit.app**

Predicting Premier League results using 20 years of real match data, with the
bookmakers as the benchmark. There's also a team playing-style analysis and an
interactive Streamlit dashboard.

**Data:** every Premier League match since 2005-06 from [football-data.co.uk](https://www.football-data.co.uk)
(about 8,000 games): scores, shots, shots on target, corners, fouls, cards and Bet365 odds.
The pipeline downloads it automatically and refreshes the current season every time it runs.

## What's in it

### 1. Match outcome predictor
Predicts home win / draw / away win probabilities.

- **Features** come only from games played *before* kick-off, so nothing leaks from the future:
  - **Elo ratings** that I built from scratch. They update after every game, give more weight to
    bigger wins, start promoted sides lower, and pull every team back towards the average each summer.
  - **Recent form**: each team's average goals, shots, shots on target, corners and points over
    its last 5 games.
- **Models:** Elo-only logistic regression, full logistic regression, gradient boosting
  (scikit-learn `HistGradientBoostingClassifier`).
- **Evaluation:** the seasons are split in time order. The first 2 are Elo warm-up, then training
  seasons, then 1 validation season to pick the model, then 2 test seasons for a single final score.
  The data is never shuffled.
- **Metrics:** accuracy, log loss, Brier score and ranked probability score (RPS, the
  standard metric for football forecasts because it treats draw as "between" home and away).
- **Benchmark:** Bet365 odds converted to probabilities with the bookmaker margin removed.

### 2. Team playing styles
K-means clustering of team-seasons (e.g. "Brighton 2023-24") using per-game style stats:
shots, shots faced, shot accuracy, chances conceded, corners, fouls and cards. Points and goals
are deliberately left out, so the groups describe *how* a team played rather than how good it was.
PCA puts the teams on a 2D map. A similarity search finds a team-season's closest
statistical twins across 20 years.

### 3. Dashboard (Streamlit)
- **Match predictor:** pick any two teams and get the probabilities, current form, Elo history and recent head-to-heads
- **Model vs bookmakers:** test-season scores, a calibration plot, feature importance, and how the model is doing this season
- **Team styles:** an interactive style map, the cluster profiles, and similar-team search
- **Elo ratings:** the current table and rating history since 2005

## Results

Test seasons: 2024-25, 2025-26 (760 matches, never used for training or model selection).

| Model | Accuracy | Log loss ↓ | Brier ↓ | RPS ↓ |
|---|---|---|---|---|
| Baseline (league averages) | 41.7% | 1.083 | 0.656 | 0.232 |
| Elo only | 49.3% | 1.009 | 0.605 | 0.207 |
| Logistic regression **(chosen on validation)** | 50.8% | 1.013 | 0.609 | 0.207 |
| Gradient boosting | 50.0% | 1.040 | 0.625 | 0.212 |
| Bookmakers (Bet365, margin removed) | 51.4% | 0.995 | 0.595 | 0.201 |

The model gets about 9 points of accuracy above the baseline and comes within **0.018 log loss** of
the bookmakers, using only public match stats. It doesn't beat the market, and that's expected: the odds
also price in line-ups, injuries and money from sharp bettors.

## Run it

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py            # downloads data, builds features, trains models
streamlit run dashboard/app.py    # opens http://localhost:8501
python -m pytest                  # checks the features don't use future information
```

Use from code:

```python
from src.match_predictor import predict_match_outcome
predict_match_outcome("Arsenal", "Man City")
# {'prediction': 'Arsenal win', 'probabilities': {'home_win': ..., 'draw': ..., 'away_win': ...}}
```

Team names follow football-data.co.uk's spelling (`Man City`, `Man United`, `Nott'm Forest`, `Wolves`).

## Project structure

```
├── run_pipeline.py          # runs everything below in order
├── src/
│   ├── config.py            # seasons, paths, Elo settings
│   ├── data_loader.py       # download + clean football-data.co.uk CSVs
│   ├── features.py          # Elo ratings, rolling form, bookmaker probabilities
│   ├── match_predictor.py   # train / validate / test, metrics, predictions
│   └── team_styles.py       # K-means style clusters, PCA, similarity search
├── dashboard/app.py         # Streamlit app
├── tests/test_features.py   # no-leakage checks
├── data/                    # created by the pipeline (raw CSVs + processed tables)
└── models/                  # created by the pipeline
```

## Limitations and next steps

- No player data, line-ups, injuries or xG. Bookmakers price all of these in, which is the main
  reason they're hard to beat on log loss.
- Next steps: add xG (e.g. from Understat), tune the Elo settings on the validation season, add a
  draw-specific model (draws are the hardest outcome to predict), and run a betting backtest on
  the gap between the model and the market.
