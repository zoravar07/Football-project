"""Streamlit dashboard.  Start with:  streamlit run dashboard/app.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.match_predictor import predict_match_outcome  # noqa: E402
from src.team_styles import (NICE_NAMES, STYLE_FEATURES, choose_k, cluster_styles,  # noqa: E402
                             similar_teams, team_season_table)

st.set_page_config(page_title="Premier League ML", page_icon="⚽", layout="wide")

# Newer Streamlit uses width="stretch"; older versions (e.g. on Python 3.9) use use_container_width.
from packaging.version import Version  # noqa: E402  (installed with streamlit)
STRETCH = {"width": "stretch"} if Version(st.__version__) >= Version("1.50") else {"use_container_width": True}

HOME, DRAW, AWAY = "#2563eb", "#9ca3af", "#dc2626"
MODEL, MARKET = "#2563eb", "#f59e0b"
P = config.PROCESSED_DIR


# ---------------------------------------------------------------- data

@st.cache_data
def load():
    needed = ["matches.csv", "features.csv", "elo_history.csv", "team_state.csv",
              "team_match_log.csv", "predictions.csv", "match_metrics.json"]
    missing = [f for f in needed if not (P / f).exists()]
    if missing or not (config.MODELS_DIR / "match_model.joblib").exists():
        return None
    return {
        "matches": pd.read_csv(P / "matches.csv", parse_dates=["date"]),
        "elo": pd.read_csv(P / "elo_history.csv", parse_dates=["date"]),
        "state": pd.read_csv(P / "team_state.csv", parse_dates=["last_match"]),
        "log": pd.read_csv(P / "team_match_log.csv", parse_dates=["date"]),
        "preds": pd.read_csv(P / "predictions.csv", parse_dates=["date"]),
        "importance": pd.read_csv(P / "feature_importance.csv"),
        "metrics": json.loads((P / "match_metrics.json").read_text()),
    }


@st.cache_data
def styles(k: int):
    d = load()
    return cluster_styles(team_season_table(d["log"]), k=k)


@st.cache_data
def k_scores():
    return choose_k(team_season_table(load()["log"]))


# On a fresh server (e.g. Streamlit Cloud) nothing has been trained yet, so build it all first.
if not (config.MODELS_DIR / "match_model.joblib").exists():
    with st.spinner("First launch: downloading 20 seasons of results and training the models. "
                    "This takes about a minute..."):
        from run_pipeline import main as run_pipeline
        run_pipeline()
    st.cache_data.clear()

data = load()
if data is None:
    st.error("No trained models found. In the terminal, run `python run_pipeline.py` first, "
             "then refresh this page.")
    st.stop()

matches, state = data["matches"], data["state"]
latest_season = matches["season"].iloc[-1]
current_teams = sorted(set(matches.loc[matches["season"] == latest_season, "home_team"])
                       | set(matches.loc[matches["season"] == latest_season, "away_team"]))
m = data["metrics"]
test_metrics = pd.DataFrame(m["test_metrics"])


# ---------------------------------------------------------------- header

st.title("⚽ Premier League ML")
st.caption(f"{len(matches):,} Premier League matches from {matches['season'].iloc[0]} to "
           f"{latest_season} · data from football-data.co.uk · last match in data: "
           f"{matches['date'].max():%d %b %Y}")

tab_pred, tab_eval, tab_style, tab_elo, tab_about = st.tabs(
    ["Match predictor", "Model vs bookmakers", "Team styles", "Elo ratings", "How it works"])


# ---------------------------------------------------------------- match predictor

with tab_pred:
    c1, c2 = st.columns(2)
    show_all = st.checkbox("Include teams not in the league this season", value=False)
    options = sorted(state["team"]) if show_all else current_teams
    home = c1.selectbox("Home team", options, index=0)
    away = c2.selectbox("Away team", options, index=min(1, len(options) - 1))

    if home == away:
        st.info("Pick two different teams.")
    else:
        res = predict_match_outcome(home, away)
        p = res["probabilities"]
        k1, k2, k3 = st.columns(3)
        k1.metric(f"{home} win", f"{p['home_win']:.0%}")
        k2.metric("Draw", f"{p['draw']:.0%}")
        k3.metric(f"{away} win", f"{p['away_win']:.0%}")

        fig = go.Figure()
        for label, val, col in ((f"{home} win", p["home_win"], HOME), ("Draw", p["draw"], DRAW),
                                (f"{away} win", p["away_win"], AWAY)):
            fig.add_bar(x=[val], y=[""], orientation="h", name=label, marker_color=col,
                        text=f"{label} {val:.0%}", textposition="inside", insidetextanchor="middle")
        fig.update_layout(barmode="stack", height=90, showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
                          xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False))
        st.plotly_chart(fig, **STRETCH)
        st.caption(f"Model: {m['chosen_model']}, using each team's current Elo rating and "
                   f"their average stats over the last {config.ROLLING_WINDOW} games.")

        s = state.set_index("team")
        form_cols = {"elo": "Elo rating", "points": "Points per game", "goals_for": "Goals scored",
                     "goals_against": "Goals conceded", "shots_for": "Shots", "sot_for": "Shots on target",
                     "shots_against": "Shots faced", "corners_for": "Corners"}
        comp = pd.DataFrame({home: s.loc[home, list(form_cols)], away: s.loc[away, list(form_cols)]})
        comp.index = [form_cols[c] for c in comp.index]
        left, right = st.columns([1, 1.4])
        left.markdown(f"**Current form** (last {config.ROLLING_WINDOW} matches)")
        left.dataframe(comp.astype(float).round(2), **STRETCH)

        hist = data["elo"][data["elo"]["team"].isin([home, away])]
        hist = hist[hist["date"] >= hist["date"].max() - pd.Timedelta(days=365 * 3)]
        fig = px.line(hist, x="date", y="elo", color="team", color_discrete_map={home: HOME, away: AWAY},
                      labels={"elo": "Elo rating", "date": ""}, title="Elo rating, last 3 years")
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0), legend_title_text="")
        right.plotly_chart(fig, **STRETCH)

        h2h = matches[((matches["home_team"] == home) & (matches["away_team"] == away))
                      | ((matches["home_team"] == away) & (matches["away_team"] == home))]
        if len(h2h):
            st.markdown(f"**Last meetings** ({len(h2h)} in the data)")
            st.dataframe(h2h.tail(6).iloc[::-1][["date", "season", "home_team", "home_goals",
                                                "away_goals", "away_team"]]
                         .assign(date=lambda d: d["date"].dt.strftime("%d %b %Y")),
                         hide_index=True, **STRETCH)


# ---------------------------------------------------------------- evaluation

with tab_eval:
    st.markdown(f"Trained on **{m['seasons']['train'][0]} to {m['seasons']['train'][-1]}**, "
                f"model chosen on **{', '.join(m['seasons']['val'])}**, scored once on "
                f"**{', '.join(m['seasons']['test'])}** (never seen during training). "
                "The bookmaker line is the toughest benchmark there is: their odds already "
                "contain team news, injuries and market money that this model doesn't see.")

    chosen = test_metrics[test_metrics["model"] == m["chosen_model"]].iloc[0]
    mkt = test_metrics[test_metrics["model"].str.startswith("Bookmakers")]
    base = test_metrics[test_metrics["model"].str.startswith("Baseline")].iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Model accuracy", f"{chosen['accuracy']:.1%}",
              f"{(chosen['accuracy'] - base['accuracy']) * 100:+.1f} pts vs baseline")
    if len(mkt):
        mk = mkt.iloc[0]
        k2.metric("Bookmaker accuracy", f"{mk['accuracy']:.1%}")
        k3.metric("Model log loss", f"{chosen['log_loss']:.3f}",
                  f"{chosen['log_loss'] - mk['log_loss']:+.3f} vs bookmakers", delta_color="inverse")
        k4.metric("Bookmaker log loss", f"{mk['log_loss']:.3f}")

    st.dataframe(
        test_metrics.rename(columns={"model": "Model", "matches": "Matches", "accuracy": "Accuracy",
                                     "log_loss": "Log loss ↓", "brier": "Brier ↓", "rps": "RPS ↓"})
        .style.format({"Accuracy": "{:.1%}", "Log loss ↓": "{:.3f}", "Brier ↓": "{:.3f}",
                       "RPS ↓": "{:.4f}"}),
        hide_index=True, **STRETCH)
    st.caption("Accuracy only checks the top pick. Log loss, Brier and RPS (ranked probability score) "
               "judge the probabilities themselves; lower is better. Baseline = always predict the "
               "league's average home/draw/away split.")

    test = data["preds"][data["preds"]["split"] == "test"]
    c1, c2 = st.columns(2)

    # Calibration: when the model says 60%, does it happen ~60% of the time?
    bins = np.linspace(0, 1, 11)
    cal = []
    for who, cols in (("Model", ["model_H", "model_D", "model_A"]), ("Bookmakers", ["mkt_H", "mkt_D", "mkt_A"])):
        t = test.dropna(subset=cols)
        if t.empty:
            continue
        probs = t[cols].values.ravel()
        hits = np.column_stack([(t["result"] == r).values for r in "HDA"]).ravel()
        b = pd.cut(probs, bins, include_lowest=True)
        g = pd.DataFrame({"p": probs, "hit": hits, "bin": b}).groupby("bin", observed=True)
        cal.append(pd.DataFrame({"predicted": g["p"].mean(), "actual": g["hit"].mean(),
                                 "n": g.size(), "source": who}))
    cal = pd.concat(cal)
    fig = px.line(cal[cal["n"] >= 15], x="predicted", y="actual", color="source", markers=True,
                  color_discrete_map={"Model": MODEL, "Bookmakers": MARKET},
                  labels={"predicted": "Predicted probability", "actual": "How often it happened"},
                  title="Calibration (test seasons)")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="grey"))
    fig.update_layout(height=380, legend_title_text="", xaxis_range=[0, 1], yaxis_range=[0, 1])
    c1.plotly_chart(fig, **STRETCH)
    c1.caption("Points on the dotted line = perfectly calibrated probabilities.")

    imp = data["importance"].head(12).iloc[::-1]
    imp["feature"] = imp["feature"].str.replace("_", " ")
    fig = px.bar(imp, x="importance", y="feature", orientation="h",
                 title="What the model relies on", labels={"importance": "Increase in log loss when shuffled",
                                                           "feature": ""})
    fig.update_traces(marker_color=MODEL)
    fig.update_layout(height=380)
    c2.plotly_chart(fig, **STRETCH)
    c2.caption("Permutation importance: shuffle one feature and see how much worse predictions get.")

    live = data["preds"][data["preds"]["split"] == "live"].copy()
    if len(live):
        st.subheader(f"This season so far ({live['season'].iloc[0]})")
        live["model_pick"] = live[["model_H", "model_D", "model_A"]].values.argmax(axis=1)
        live["model_pick"] = live["model_pick"].map({0: "H", 1: "D", 2: "A"})
        live["model_correct"] = live["model_pick"] == live["result"]
        has_mkt = live[["mkt_H", "mkt_D", "mkt_A"]].notna().all(axis=1)
        live["mkt_pick"] = np.where(has_mkt, pd.Series(
            live[["mkt_H", "mkt_D", "mkt_A"]].fillna(0).values.argmax(axis=1), index=live.index)
            .map({0: "H", 1: "D", 2: "A"}), None)
        a, b = st.columns(2)
        a.metric("Model accuracy this season", f"{live['model_correct'].mean():.1%}", f"{len(live)} matches",
                 delta_color="off")
        if has_mkt.any():
            b.metric("Bookmakers this season", f"{(live.loc[has_mkt, 'mkt_pick'] == live.loc[has_mkt, 'result']).mean():.1%}")
        show = live.sort_values("date", ascending=False).head(20)
        show = pd.DataFrame({
            "Date": show["date"].dt.strftime("%d %b"),
            "Match": show["home_team"] + " v " + show["away_team"],
            "Score": show["home_goals"].astype(int).astype(str) + "-" + show["away_goals"].astype(int).astype(str),
            "Home %": show["model_H"], "Draw %": show["model_D"], "Away %": show["model_A"],
            "Model right?": np.where(show["model_correct"], "✅", "❌"),
        })
        st.dataframe(show.style.format({"Home %": "{:.0%}", "Draw %": "{:.0%}", "Away %": "{:.0%}"}),
                     hide_index=True, **STRETCH)


# ---------------------------------------------------------------- team styles

with tab_style:
    st.markdown("Every team-season described by per-game style stats (shots, shots faced, accuracy, "
                "corners, fouls, cards), grouped with **K-means**. Points and goals are left out, so "
                "groups reflect *how* a team played, not just how well.")
    k = st.slider("Number of style groups (k)", 3, 7, 4)
    teams, info = styles(k)

    c1, c2 = st.columns([1.6, 1])
    seasons_avail = sorted(teams["season"].unique())
    season_filter = c2.multiselect("Show seasons", seasons_avail, default=seasons_avail[-3:])
    shown = teams[teams["season"].isin(season_filter)] if season_filter else teams
    fig = px.scatter(shown, x="pc1", y="pc2", color="style", hover_name="label",
                     hover_data={"pc1": False, "pc2": False, "points": ":.2f", "shots_for": ":.1f",
                                 "shots_against": ":.1f"},
                     labels={"pc1": "Style axis 1", "pc2": "Style axis 2", "style": "Style group"},
                     color_discrete_sequence=px.colors.qualitative.Safe,
                     category_orders={"style": sorted(teams["style"].unique())},
                     title="Team-seasons mapped by style (PCA of the style stats)")
    fig.update_traces(marker=dict(size=9, opacity=.8))
    fig.update_layout(height=480, legend=dict(orientation="h", y=-0.2))
    c1.plotly_chart(fig, **STRETCH)

    c2.markdown("**What defines each group** (vs league average)")
    cz = info["centroids_z"].rename(columns=NICE_NAMES)
    hm = px.imshow(cz.T.round(1), text_auto=True, color_continuous_scale="RdBu_r",
                   zmin=-1.5, zmax=1.5, aspect="auto")
    hm.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False,
                     xaxis_title="", yaxis_title="")
    hm.update_xaxes(tickangle=-30)
    c2.plotly_chart(hm, **STRETCH)
    c2.caption(f"Numbers are standard deviations from average. Silhouette score {info['silhouette']:.2f} "
               "(style overlaps a lot between teams, so groups aren't sharply separated).")

    summary = teams.groupby("style").agg(team_seasons=("label", "size"), avg_points=("points", "mean"),
                                         example=("label", lambda s: ", ".join(s.tail(3))))
    st.dataframe(summary.rename(columns={"team_seasons": "Team-seasons", "avg_points": "Avg points/game",
                                         "example": "Recent examples"}).round(2), **STRETCH)

    st.subheader("Find similar teams")
    default = teams["label"].iloc[-1]
    pick = st.selectbox("Team-season", sorted(teams["label"]), index=sorted(teams["label"]).index(default))
    sim = similar_teams(teams, pick, n=10)
    me = teams[teams["label"] == pick].iloc[0]
    st.caption(f"{pick}: style group **{me['style']}**, {me['points']:.2f} points per game.")
    st.dataframe(sim[["label", "style", "similarity", "points"] + STYLE_FEATURES]
                 .rename(columns={"label": "Team-season", "style": "Style", "similarity": "Similarity",
                                  "points": "Points/game", **NICE_NAMES}).round(2),
                 hide_index=True, **STRETCH)

    with st.expander("Why this many groups? (silhouette score by k)"):
        ks = k_scores()
        st.plotly_chart(px.line(ks, x="k", y="silhouette", markers=True, height=280),
                        **STRETCH)


# ---------------------------------------------------------------- Elo

with tab_elo:
    table = state[state["team"].isin(current_teams)].sort_values("elo", ascending=False)
    c1, c2 = st.columns([1, 2])
    c1.markdown(f"**Current ratings** ({latest_season} teams)")
    c1.dataframe(table[["team", "elo", "points"]].rename(
        columns={"team": "Team", "elo": "Elo", "points": f"Pts/game (last {config.ROLLING_WINDOW})"})
        .round(1), hide_index=True, **STRETCH, height=600)
    chosen_teams = c2.multiselect("Teams", sorted(state["team"]), default=list(table["team"].head(4)))
    hist = data["elo"][data["elo"]["team"].isin(chosen_teams)]
    fig = px.line(hist, x="date", y="elo", color="team", labels={"elo": "Elo rating", "date": ""})
    fig.update_layout(height=600, legend_title_text="")
    c2.plotly_chart(fig, **STRETCH)


# ---------------------------------------------------------------- about

with tab_about:
    st.markdown(f"""
**Data.** Every Premier League result since 2005-06 from [football-data.co.uk](https://www.football-data.co.uk):
score, shots, shots on target, corners, fouls, cards, and Bet365 closing odds.

**Features (all built from matches *before* kick-off).**
- *Elo rating*: a running strength score. Winners take points from losers, more for bigger wins
  and away wins. Promoted sides start at {config.ELO_PROMOTED:.0f}; ratings shrink 20% towards the
  average each summer.
- *Recent form*: each side's average goals, shots, shots on target, corners and points over
  their last {config.ROLLING_WINDOW} games.

**Models.** Elo-only logistic regression, full logistic regression, and gradient boosting.
Seasons are split in time order (train → validation → test), never shuffled. The model with the
best validation log loss is refit and scored once on the test seasons.

**Benchmark.** Bookmaker odds converted to probabilities with the margin removed.

**Team styles.** K-means clustering on per-game style stats for each team-season, PCA for the map.

**Limitations.** No player-level data, injuries, line-ups or xG, which is a big part of why
bookmakers are hard to beat.
""")
    st.json(m["validation_log_loss"])
