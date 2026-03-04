"""
Prediction service: loads trained model and exposes prediction logic for the API.
"""

import os
import pickle
import pandas as pd
import numpy as np

# Default to project root when running as script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_model_paths():
    """Return paths to model artifacts under project root."""
    models_dir = os.path.join(ROOT, "models")
    return {
        "model": os.path.join(models_dir, "mvp_model.pkl"),
        "scaler": os.path.join(models_dir, "scaler.pkl"),
        "features": os.path.join(models_dir, "feature_names.pkl"),
    }


def load_model():
    """Load trained model, scaler, and feature names. Returns (model, scaler, feature_names)."""
    paths = get_model_paths()
    for name, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Model artifact not found: {p}. Run train_sklearn.py first."
            )
    with open(paths["model"], "rb") as f:
        model = pickle.load(f)
    with open(paths["scaler"], "rb") as f:
        scaler = pickle.load(f)
    with open(paths["features"], "rb") as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names


def predict_from_dataframe(df, feature_cols, model, scaler, top_n=10):
    """
    Predict MVP probabilities for a dataframe that already has feature columns.
    Returns list of dicts with Player, Tm, PTS, TRB, AST, MVP_probability, rank.
    """
    X = df[feature_cols].fillna(0).values
    X = np.nan_to_num(X, nan=0.0)
    X_scaled = scaler.transform(X)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[:, 1]
    else:
        probs = model.predict(X_scaled)
    out = df[["Player", "Tm"]].copy()
    if "PTS" in df.columns:
        out["PTS"] = df["PTS"].values
    if "TRB" in df.columns:
        out["TRB"] = df["TRB"].values
    if "AST" in df.columns:
        out["AST"] = df["AST"].values
    out["MVP_probability"] = probs.tolist()
    out = out.sort_values("MVP_probability", ascending=False).head(top_n)
    out["rank"] = range(1, len(out) + 1)
    return out.to_dict(orient="records")


def predict_by_year(year, top_n=10):
    """
    Load player stats for a year, create features, and return top N MVP predictions.
    Uses preprocess_data from project root.
    """
    import sys
    sys.path.insert(0, ROOT)
    from preprocess_data import load_player_stats, create_features

    model, scaler, feature_names = load_model()
    player_stats = load_player_stats(year)
    if player_stats is None:
        return None
    player_stats = create_features(player_stats)
    player_stats["Year"] = year
    for col in feature_names:
        if col not in player_stats.columns:
            player_stats[col] = 0
    return predict_from_dataframe(
        player_stats, feature_names, model, scaler, top_n=top_n
    )
