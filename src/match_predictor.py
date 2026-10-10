"""Predict home win / draw / away win, and compare the model with the bookmakers.

How the data is split (by season, never shuffled, so the model can't learn from the future):
  - warm-up seasons: only used to get Elo ratings going
  - training seasons: fit every candidate model
  - validation season: pick the best candidate (lowest log loss)
  - test seasons: the winner is refit on train + validation and scored once here

Run on its own with:  python -m src.match_predictor
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import FEATURES, RESULT_LABELS, features_for_fixture

MODEL_PATH = config.MODELS_DIR / "match_model.joblib"
METRICS_PATH = config.PROCESSED_DIR / "match_metrics.json"


# ---------------------------------------------------------------- candidates

def candidate_models() -> dict:
    return {
        "Elo only (logistic)": (["elo_diff"], make_pipeline(
            SimpleImputer(), StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))),
        "Logistic regression": (FEATURES, make_pipeline(
            SimpleImputer(), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))),
        "Gradient boosting": (FEATURES, HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.03, max_iter=300, l2_regularization=1.0,
            min_samples_leaf=50, random_state=42)),
    }


# ---------------------------------------------------------------- scoring

def proba_hda(model, X: pd.DataFrame) -> np.ndarray:
    """Model probabilities in the order home, draw, away (sklearn sorts classes A, D, H)."""
    p = model.predict_proba(X)
    order = [list(model.classes_).index(c) for c in RESULT_LABELS]
    return p[:, order]


def score(probs: np.ndarray, results: pd.Series) -> dict:
    y = np.column_stack([(results == c).values for c in RESULT_LABELS]).astype(float)
    p = np.clip(probs, 1e-12, 1)
    p = p / p.sum(axis=1, keepdims=True)
    # Ranked probability score: rewards being "nearly right" (predicting a draw when
    # the home side won is less wrong than predicting an away win). Lower is better.
    rps = (((np.cumsum(p, axis=1) - np.cumsum(y, axis=1))[:, :2] ** 2).sum(axis=1) / 2).mean()
    return {
        "matches": int(len(results)),
        "accuracy": float((p.argmax(axis=1) == y.argmax(axis=1)).mean()),
        "log_loss": float(-(y * np.log(p)).sum(axis=1).mean()),
        "brier": float(((p - y) ** 2).sum(axis=1).mean()),
        "rps": float(rps),
    }


def split_seasons(table: pd.DataFrame) -> dict[str, list[int]]:
    current = config.current_season_start()
    counts = table.groupby("season_start").size()
    completed = [s for s in sorted(counts.index) if s < current and counts[s] >= 370]
    usable = completed[config.ELO_WARMUP_SEASONS:]
    n_hold = config.N_TEST_SEASONS + config.N_VALIDATION_SEASONS
    if len(usable) < n_hold + 3:
        raise RuntimeError(f"Need more seasons of data (found {len(usable)} usable).")
    return {
        "train": usable[:-n_hold],
        "val": usable[-n_hold:-config.N_TEST_SEASONS],
        "test": usable[-config.N_TEST_SEASONS:],
        "live": [current] if current in counts.index else [],
    }


# ---------------------------------------------------------------- training

def train(table: pd.DataFrame) -> dict:
    split = split_seasons(table)
    part = {k: table[table["season_start"].isin(v)] for k, v in split.items()}
    tr, va, te = part["train"], part["val"], part["test"]

    # 1. Fit each candidate on training seasons, compare on the validation season.
    val_scores = {}
    for name, (cols, model) in candidate_models().items():
        model.fit(tr[cols], tr["result"])
        val_scores[name] = score(proba_hda(model, va[cols]), va["result"])["log_loss"]
    best_name = min(val_scores, key=val_scores.get)
    print(f"  validation log loss: " + ", ".join(f"{k} {v:.4f}" for k, v in val_scores.items()))
    print(f"  picked: {best_name}")

    # 2. Refit every candidate on train + validation and score on the untouched test seasons.
    trva = pd.concat([tr, va])
    rows, fitted = [], {}
    home_rate = trva["result"].value_counts(normalize=True).reindex(RESULT_LABELS).values
    rows.append({"model": "Baseline: league averages",
                 **score(np.tile(home_rate, (len(te), 1)), te["result"])})
    for name, (cols, model) in candidate_models().items():
        model.fit(trva[cols], trva["result"])
        fitted[name] = (cols, model)
        rows.append({"model": name, **score(proba_hda(model, te[cols]), te["result"])})
    has_market = "mkt_H" in te.columns and te[["mkt_H", "mkt_D", "mkt_A"]].notna().all(axis=1).any()
    if has_market:
        te_m = te.dropna(subset=["mkt_H", "mkt_D", "mkt_A"])
        rows.append({"model": "Bookmakers (Bet365, margin removed)",
                     **score(te_m[["mkt_H", "mkt_D", "mkt_A"]].values, te_m["result"])})
    metrics = pd.DataFrame(rows)

    # 3. Save the chosen model's test-set and current-season predictions for the dashboard.
    #    Current-season games use a model refit on everything up to last season,
    #    so they're still genuine out-of-sample predictions.
    cols, chosen = fitted[best_name]
    _, upto_last = candidate_models()[best_name]
    upto_last.fit(pd.concat([trva, te])[cols], pd.concat([trva, te])["result"])
    keep = ["date", "season", "home_team", "away_team", "home_goals", "away_goals", "result",
            "mkt_H", "mkt_D", "mkt_A"]
    preds = []
    for which, model in (("test", chosen), ("live", upto_last)):
        df = part[which]
        if df.empty:
            continue
        out = df[[c for c in keep if c in df.columns]].copy()
        out[["model_H", "model_D", "model_A"]] = proba_hda(model, df[cols])
        out["split"] = which
        preds.append(out)
    pd.concat(preds).to_csv(config.PROCESSED_DIR / "predictions.csv", index=False)

    # 4. Which features matter? (shuffle each one and see how much worse the model gets)
    imp = permutation_importance(chosen, te[cols], te["result"], scoring="neg_log_loss",
                                 n_repeats=10, random_state=42)
    pd.DataFrame({"feature": cols, "importance": imp.importances_mean}) \
        .sort_values("importance", ascending=False) \
        .to_csv(config.PROCESSED_DIR / "feature_importance.csv", index=False)

    # 5. Production model: same recipe refit on every finished match, for future fixtures.
    _, production = candidate_models()[best_name]
    played = table[table["season_start"].isin(split["train"] + split["val"] + split["test"] + split["live"])]
    production.fit(played[cols], played["result"])
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"name": best_name, "features": cols, "model": production}, MODEL_PATH)

    label = lambda ss: [config.season_label(s) for s in ss]
    summary = {
        "chosen_model": best_name,
        "validation_log_loss": val_scores,
        "seasons": {k: label(v) for k, v in split.items()},
        "test_metrics": metrics.to_dict(orient="records"),
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------- using the model

def predict_match_outcome(home_team: str, away_team: str) -> dict:
    """Probabilities for a match between two teams, based on their current Elo and form."""
    bundle = joblib.load(MODEL_PATH)
    state = pd.read_csv(config.PROCESSED_DIR / "team_state.csv")
    X = features_for_fixture(state, home_team, away_team)[bundle["features"]]
    p = proba_hda(bundle["model"], X)[0]
    probs = {"home_win": float(p[0]), "draw": float(p[1]), "away_win": float(p[2])}
    names = {"home_win": f"{home_team} win", "draw": "Draw", "away_win": f"{away_team} win"}
    return {"prediction": names[max(probs, key=probs.get)], "probabilities": probs}


if __name__ == "__main__":
    table = pd.read_csv(config.PROCESSED_DIR / "features.csv", parse_dates=["date"])
    s = train(table)
    print(pd.DataFrame(s["test_metrics"]).round(4).to_string(index=False))
