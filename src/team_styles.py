"""Group team-seasons by playing style with K-means, and find statistically similar sides.

Each row is one team in one season (e.g. "Liverpool 2019-20"), described by per-game
averages. Clustering uses style stats only (shots, corners, fouls...), not points or
goals, so the groups describe *how* teams play rather than just how good they were.

Run on its own with:  python -m src.team_styles
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src import config

STYLE_FEATURES = ["shots_for", "shots_against", "shot_accuracy", "sot_against",
                  "corners_for", "fouls", "yellows"]

NICE_NAMES = {
    "shots_for": "Shots per game",
    "shots_against": "Shots faced per game",
    "shot_accuracy": "Shot accuracy (on target %)",
    "sot_against": "Shots on target faced",
    "corners_for": "Corners per game",
    "fouls": "Fouls per game",
    "yellows": "Yellow cards per game",
    "goals_for": "Goals per game",
    "goals_against": "Goals conceded per game",
    "points": "Points per game",
}

# How to describe a cluster that's well above (+) or below (-) average on a stat.
TRAIT_WORDS = {
    ("shots_for", +1): "high shot volume", ("shots_for", -1): "few shots",
    ("shots_against", +1): "open defence", ("shots_against", -1): "tight defence",
    ("shot_accuracy", +1): "accurate shooting", ("shot_accuracy", -1): "speculative shooting",
    ("sot_against", +1): "concede clear chances", ("sot_against", -1): "limit clear chances",
    ("corners_for", +1): "lots of corners", ("corners_for", -1): "few corners",
    ("fouls", +1): "physical", ("fouls", -1): "low-foul",
    ("yellows", +1): "card-prone", ("yellows", -1): "disciplined",
}


def team_season_table(log: pd.DataFrame, min_games: int = 8) -> pd.DataFrame:
    g = log.groupby(["season", "team"])
    df = g[["goals_for", "goals_against", "shots_for", "shots_against", "sot_for",
            "sot_against", "corners_for", "fouls", "yellows", "points"]].mean()
    df["games"] = g.size()
    df["shot_accuracy"] = df["sot_for"] / df["shots_for"]
    df = df[df["games"] >= min_games].dropna(subset=STYLE_FEATURES).reset_index()
    df["label"] = df["team"] + " " + df["season"]
    return df


def _name_cluster(z_centroid: pd.Series) -> str:
    top = z_centroid.abs().sort_values(ascending=False).index[:2]
    return ", ".join(TRAIT_WORDS[(f, 1 if z_centroid[f] > 0 else -1)] for f in top).capitalize()


def cluster_styles(teams: pd.DataFrame, k: int = 4, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Adds 'cluster', 'style', 'pc1', 'pc2' columns. Returns (table, info)."""
    teams = teams.copy()
    scaler = StandardScaler()
    Z = scaler.fit_transform(teams[STYLE_FEATURES])
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
    teams["cluster"] = km.labels_

    centroids = pd.DataFrame(km.cluster_centers_, columns=STYLE_FEATURES)
    names = {i: _name_cluster(centroids.loc[i]) for i in range(k)}
    # Make sure two clusters never end up with the same name.
    seen: dict[str, int] = {}
    for i, n in names.items():
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[i] = f"{n} ({seen[n]})"
    teams["style"] = teams["cluster"].map(names)

    pca = PCA(n_components=2, random_state=seed).fit(Z)
    teams[["pc1", "pc2"]] = pca.transform(Z)

    info = {
        "silhouette": float(silhouette_score(Z, km.labels_)),
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "centroids_z": centroids.rename(index=names),
    }
    return teams, info


def similar_teams(teams: pd.DataFrame, label: str, n: int = 10) -> pd.DataFrame:
    """The n team-seasons whose style is closest to `label` (standardised distance)."""
    Z = StandardScaler().fit_transform(teams[STYLE_FEATURES])
    i = teams.index[teams["label"] == label][0]
    d = np.sqrt(((Z - Z[teams.index.get_loc(i)]) ** 2).sum(axis=1))
    out = teams.assign(distance=d).drop(index=i).nsmallest(n, "distance")
    out["similarity"] = 1 / (1 + out["distance"])
    return out


def choose_k(teams: pd.DataFrame, ks=range(2, 9)) -> pd.DataFrame:
    """Silhouette score for each k, to justify the number of clusters."""
    Z = StandardScaler().fit_transform(teams[STYLE_FEATURES])
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Z)
        rows.append({"k": k, "silhouette": silhouette_score(Z, km.labels_), "inertia": km.inertia_})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    log = pd.read_csv(config.PROCESSED_DIR / "team_match_log.csv")
    t, info = cluster_styles(team_season_table(log))
    print(t.groupby("style").size())
    print(f"silhouette {info['silhouette']:.3f}")
