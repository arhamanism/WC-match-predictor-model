"""
Feature engineering for a single hypothetical match, mirroring the exact
logic used to build training features in World_Cup_Match_Predictor.ipynb.

This module turns (home_team, away_team, tournament, country, neutral, date)
into a one-row DataFrame with the same columns/order the CatBoost model was
trained on, using the lookup tables produced by export_artifacts.py.
"""

import json
import pickle
from pathlib import Path

import pandas as pd

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def load_artifacts():
    """Load model, schema, and all lookup tables. Call once and cache.

    Note: CatBoost is trained directly on raw string categoricals via its
    built-in cat_features handling (see cells 80-81 of the notebook, which
    use train_raw/test_raw straight from df, NOT the ordinal-encoded
    train_2/test_2). So no separate encoder is needed at inference time.
    """
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(ARTIFACTS_DIR / "cat_model.cbm"))

    with open(ARTIFACTS_DIR / "schema.json") as f:
        schema = json.load(f)

    with open(ARTIFACTS_DIR / "team_recent_form.json") as f:
        team_recent_form = json.load(f)

    with open(ARTIFACTS_DIR / "head_to_head.json") as f:
        head_to_head = json.load(f)

    with open(ARTIFACTS_DIR / "elo_ratings.json") as f:
        elo_ratings = json.load(f)

    with open(ARTIFACTS_DIR / "teams.json") as f:
        teams = json.load(f)

    with open(ARTIFACTS_DIR / "tournaments.json") as f:
        tournaments = json.load(f)

    with open(ARTIFACTS_DIR / "countries.json") as f:
        countries = json.load(f)

    return {
        "model": model,
        "schema": schema,
        "team_recent_form": team_recent_form,
        "head_to_head": head_to_head,
        "elo_ratings": elo_ratings,
        "teams": teams,
        "tournaments": tournaments,
        "countries": countries,
    }


def _recent_form(team_recent_form, team):
    points = team_recent_form.get(team, [])
    total = sum(points)
    avg = total / len(points) if points else 0
    return total, avg


def _head_to_head(head_to_head, home_team, away_team):
    pair = tuple(sorted([home_team, away_team]))
    key = "|||".join(pair)
    stats = head_to_head.get(key)

    if stats is None:
        return {
            "home_h2h_wins": 0,
            "away_h2h_wins": 0,
            "h2h_draws": 0,
            "home_h2h_goals_scored": 0,
            "away_h2h_goals_scored": 0,
            "home_h2h_goal_difference": 0,
            "last_h2h_result": "No History",
            "h2h_matches_played": 0,
            "home_h2h_avg_goals": 0,
            "away_h2h_avg_goals": 0,
        }

    if pair[0] == home_team:
        home_wins, away_wins = stats["team1_wins"], stats["team2_wins"]
        home_goals, away_goals = stats["team1_goals"], stats["team2_goals"]
        last_result = stats["last_result"] or "No History"
    else:
        home_wins, away_wins = stats["team2_wins"], stats["team1_wins"]
        home_goals, away_goals = stats["team2_goals"], stats["team1_goals"]
        if stats["last_result"] == "Win":
            last_result = "Loss"
        elif stats["last_result"] == "Loss":
            last_result = "Win"
        elif stats["last_result"] == "Draw":
            last_result = "Draw"
        else:
            last_result = "No History"

    matches = stats["matches"]
    return {
        "home_h2h_wins": home_wins,
        "away_h2h_wins": away_wins,
        "h2h_draws": stats["draws"],
        "home_h2h_goals_scored": home_goals,
        "away_h2h_goals_scored": away_goals,
        "home_h2h_goal_difference": home_goals - away_goals,
        "last_h2h_result": last_result,
        "h2h_matches_played": matches,
        "home_h2h_avg_goals": home_goals / matches if matches else 0,
        "away_h2h_avg_goals": away_goals / matches if matches else 0,
    }


def build_feature_row(artifacts, home_team, away_team, tournament, country, neutral, match_date):
    """Build a single-row DataFrame of raw (unencoded) features for one matchup."""
    schema = artifacts["schema"]

    home_recent_points, avg_home_recent_points = _recent_form(artifacts["team_recent_form"], home_team)
    away_recent_points, avg_away_recent_points = _recent_form(artifacts["team_recent_form"], away_team)

    h2h = _head_to_head(artifacts["head_to_head"], home_team, away_team)

    elo_ratings = artifacts["elo_ratings"]
    default_elo = sum(elo_ratings.values()) / len(elo_ratings) if elo_ratings else 1500
    home_elo = elo_ratings.get(home_team, default_elo)
    away_elo = elo_ratings.get(away_team, default_elo)

    row = {
        "home_team": home_team,
        "away_team": away_team,
        "tournament": tournament,
        "country": country,
        "neutral": int(neutral),
        "year": match_date.year,
        "month": match_date.month,
        "day": match_date.day,
        "home_recent_points": home_recent_points,
        "away_recent_points": away_recent_points,
        "avg_home_recent_points": avg_home_recent_points,
        "avg_away_recent_points": avg_away_recent_points,
        **h2h,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_difference": home_elo - away_elo,
        "home_higher_elo": int(home_elo > away_elo),
    }

    # Enforce exact training column order
    return pd.DataFrame([row])[schema["predictors"]]


def predict_match(artifacts, home_team, away_team, tournament, country, neutral, match_date):
    """Return (predicted_label, {label: probability}) for one matchup."""
    X = build_feature_row(artifacts, home_team, away_team, tournament, country, neutral, match_date)

    # CatBoost takes raw string categoricals directly - no encoder needed
    # (see load_artifacts docstring for why).
    model = artifacts["model"]
    proba = model.predict_proba(X)[0]
    class_names = artifacts["schema"]["class_names"]

    probs = dict(zip(class_names, proba))
    predicted_label = max(probs, key=probs.get)
    return predicted_label, probs


def _form_record(team_recent_form, team):
    """Convert a team's last-5 points list (3/1/0 per match) into a W/D/L record."""
    points = team_recent_form.get(team, [])
    wins = points.count(3)
    draws = points.count(1)
    losses = points.count(0)
    parts = []
    if wins:
        parts.append(f"{wins}W")
    if draws:
        parts.append(f"{draws}D")
    if losses:
        parts.append(f"{losses}L")
    return "/".join(parts) if parts else "No data"


def get_match_insights(artifacts, home_team, away_team):
    """Human-readable talking points for a matchup: ELO gap, form, head-to-head edge."""
    elo_ratings = artifacts["elo_ratings"]
    default_elo = sum(elo_ratings.values()) / len(elo_ratings) if elo_ratings else 1500
    home_elo = elo_ratings.get(home_team, default_elo)
    away_elo = elo_ratings.get(away_team, default_elo)
    elo_gap = abs(round(home_elo - away_elo))
    elo_favored = home_team if home_elo > away_elo else away_team if away_elo > home_elo else None

    home_form = _form_record(artifacts["team_recent_form"], home_team)
    away_form = _form_record(artifacts["team_recent_form"], away_team)

    h2h = _head_to_head(artifacts["head_to_head"], home_team, away_team)
    home_wins, away_wins = h2h["home_h2h_wins"], h2h["away_h2h_wins"]
    if h2h["h2h_matches_played"] == 0:
        h2h_summary = "No prior meetings"
    elif home_wins > away_wins:
        h2h_summary = f"{home_team} edge, {home_wins}-{away_wins}"
    elif away_wins > home_wins:
        h2h_summary = f"{away_team} edge, {away_wins}-{home_wins}"
    else:
        h2h_summary = f"Even, {home_wins}-{away_wins}"

    return {
        "elo_gap": elo_gap,
        "elo_favored": elo_favored,
        "home_form": home_form,
        "away_form": away_form,
        "h2h_summary": h2h_summary,
    }
