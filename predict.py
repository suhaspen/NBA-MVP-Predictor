"""
Prediction script to predict MVP candidates for a given year.
Uses sklearn model (models/) if available, otherwise PyTorch model.
"""

import pandas as pd
import numpy as np
import pickle
import os

def _use_sklearn_model():
    """Use sklearn model when available (from train_sklearn.py)."""
    return os.path.exists("models") and os.path.exists(os.path.join("models", "mvp_model.pkl"))

def load_model(model_path="best_model.pth", use_share=True):
    """Load trained model (sklearn from models/ or PyTorch from root)."""
    if _use_sklearn_model():
        from services.predictor import load_model as load_sklearn
        return load_sklearn()
    # PyTorch path
    import torch
    from model import MVPRanker, MVPPredictor
    with open("feature_names.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    input_size = len(feature_cols)
    if use_share:
        model = MVPRanker(input_size=input_size, hidden_sizes=[128, 64, 32], dropout=0.3)
    else:
        model = MVPPredictor(input_size=input_size, hidden_sizes=[128, 64, 32], dropout=0.3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model, scaler, feature_cols

def predict_year(year, top_n=10, use_share=True):
    """
    Predict MVP candidates for a given year.
    
    Args:
        year: Year to predict for
        top_n: Number of top candidates to return
        use_share: Whether model uses share (regression) or binary (classification)
    
    Returns:
        DataFrame with predictions
    """
    print(f"Predicting MVP for {year}...")
    from preprocess_data import load_player_stats, create_features

    if _use_sklearn_model():
        results = __predict_sklearn(year, top_n)
        if results is not None:
            results = results.rename(columns={"MVP_probability": "Predicted_MVP_Share"})
            results = results[["Player", "Tm", "PTS", "TRB", "AST", "Predicted_MVP_Share"]]
        return results

    # PyTorch path
    model, scaler, feature_cols = load_model(use_share=use_share)
    player_stats = load_player_stats(year)
    if player_stats is None:
        print(f"Could not load player stats for {year}")
        return None
    player_stats = create_features(player_stats)
    player_stats["Year"] = year
    for col in feature_cols:
        if col not in player_stats.columns:
            player_stats[col] = 0
    X = player_stats[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    X_scaled = scaler.transform(X)
    import torch
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        predictions = model(X_tensor).squeeze().numpy()
    results = player_stats[["Player", "Tm", "PTS", "TRB", "AST", "WS"]].copy()
    results["Predicted_MVP_Share"] = predictions
    results = results.sort_values("Predicted_MVP_Share", ascending=False)
    return results.head(top_n)


def __predict_sklearn(year, top_n):
    """Use services.predictor for sklearn model."""
    from services.predictor import predict_by_year
    preds = predict_by_year(year, top_n=top_n)
    if preds is None:
        return None
    return pd.DataFrame(preds)

def predict_current_season(top_n=10):
    """Predict MVP for the current season (2024)."""
    return predict_year(2024, top_n=top_n)

def compare_with_actual(year, top_n=10):
    """
    Compare predictions with actual MVP results.
    
    Args:
        year: Year to compare
        top_n: Number of top candidates to show
    """
    # Load actual MVP data
    mvps = pd.read_csv("mvps.csv")
    mvps["Player"] = mvps["Player"].str.replace("*", "", regex=False).str.strip()
    mvp_year = mvps[mvps["Year"] == year].copy()
    
    # Get predictions
    predictions = predict_year(year, top_n=top_n)
    
    if predictions is None:
        return
    
    # Merge with actual results
    comparison = predictions.merge(
        mvp_year[["Player", "Rank", "Share"]],
        on="Player",
        how="left"
    )
    
    print(f"\n{'='*80}")
    print(f"MVP Predictions vs Actual Results for {year}")
    print(f"{'='*80}")
    print(f"\nTop {top_n} Predicted MVP Candidates:")
    print(comparison[["Player", "Tm", "PTS", "TRB", "AST", "Predicted_MVP_Share", "Rank", "Share"]].to_string(index=False))
    
    # Check if actual MVP is in top predictions
    actual_mvp = mvp_year[mvp_year["Rank"] == 1]["Player"].values
    if len(actual_mvp) > 0:
        actual_mvp_name = actual_mvp[0]
        predicted_rank = comparison[comparison["Player"] == actual_mvp_name].index
        if len(predicted_rank) > 0:
            rank = predicted_rank[0] + 1
            print(f"\n✓ Actual MVP ({actual_mvp_name}) predicted at rank #{rank}")
        else:
            print(f"\n✗ Actual MVP ({actual_mvp_name}) not in top {top_n} predictions")

def main():
    """Main prediction function."""
    import sys
    
    if len(sys.argv) > 1:
        year = int(sys.argv[1])
        top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        compare_with_actual(year, top_n=top_n)
    else:
        # Predict for 2024
        print("Predicting MVP for 2024 season...")
        results = predict_current_season(top_n=10)
        if results is not None:
            print("\nTop 10 MVP Candidates:")
            print(results.to_string(index=False))
        
        # Also show comparison for a recent year with known results
        print("\n" + "="*80)
        compare_with_actual(2023, top_n=10)

if __name__ == "__main__":
    main()
