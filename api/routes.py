"""
REST API routes for NBA MVP Predictor.
"""

from flask import Blueprint, request, jsonify
import pandas as pd

from services.predictor import load_model, predict_by_year, predict_from_dataframe

bp = Blueprint("api", __name__)


@bp.route("/health", methods=["GET"])
def health():
    """Health check for deployment."""
    return jsonify({"status": "ok", "service": "nba-mvp-predictor"})


@bp.route("/predict", methods=["GET"])
def predict_get():
    """
    Real-time prediction by year.
    GET /predict?year=2021&top_n=10
    """
    year = request.args.get("year", type=int)
    top_n = request.args.get("top_n", default=10, type=int)
    if year is None:
        return jsonify({"error": "Missing query parameter: year"}), 400
    try:
        results = predict_by_year(year, top_n=top_n)
        if results is None:
            return jsonify({"error": f"No data available for year {year}"}), 404
        return jsonify({"year": year, "predictions": results})
    except FileNotFoundError as e:
        return jsonify({"error": "Model not trained yet. Run train_sklearn.py first."}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/predict", methods=["POST"])
def predict_post():
    """
    Predict from JSON body: list of player stats (each with feature keys).
    Optional: "year", "top_n".
    """
    try:
        data = request.get_json() or {}
        year = data.get("year")
        top_n = int(data.get("top_n", 10))
        if year is not None:
            results = predict_by_year(year, top_n=top_n)
            if results is None:
                return jsonify({"error": f"No data for year {year}"}), 404
            return jsonify({"year": year, "predictions": results})
        # If payload has "players" array, use it (for custom input)
        players = data.get("players")
        if not players:
            return jsonify({"error": "Provide 'year' or 'players' array"}), 400
        df = pd.DataFrame(players)
        model, scaler, feature_names = load_model()
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        results = predict_from_dataframe(df, feature_names, model, scaler, top_n=top_n)
        return jsonify({"predictions": results})
    except FileNotFoundError:
        return jsonify({"error": "Model not trained. Run train_sklearn.py first."}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500
