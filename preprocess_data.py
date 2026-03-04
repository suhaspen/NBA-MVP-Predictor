"""
Data preprocessing script to merge MVP voting data with player statistics
and create features for the MVP prediction model.
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from io import StringIO
import os

def load_mvp_data():
    """Load MVP voting data from CSV."""
    mvps = pd.read_csv("mvps.csv")
    # Clean up player names - remove asterisks and extra spaces
    mvps["Player"] = mvps["Player"].str.replace("*", "", regex=False)
    mvps["Player"] = mvps["Player"].str.strip()
    return mvps

def load_player_stats(year):
    """Load player statistics for a given year."""
    file_path = f"player/{year}.html"
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found, skipping {year}")
        return None
    
    with open(file_path, encoding='utf-8') as file:
        page = file.read()
    
    soup = BeautifulSoup(page, "html.parser")
    
    # Remove header rows that interfere with parsing
    for tr in soup.find_all("tr", class_="thead"):
        tr.decompose()
    
    # Find the stats table
    table = soup.find(id="per_game_stats")
    if table is None:
        print(f"Warning: Could not find stats table for {year}")
        return None
    
    try:
        df = pd.read_html(StringIO(str(table)))[0]
        
        # Clean up column names if they have multi-level index
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel()
        
        # Remove header rows that might be in the data
        df = df[df["Player"] != "Player"]
        
        # Clean player names
        df["Player"] = df["Player"].str.replace("*", "", regex=False)
        df["Player"] = df["Player"].str.strip()
        
        # Add year column
        df["Year"] = year
        
        return df
    except Exception as e:
        print(f"Error processing {year}: {e}")
        return None

def merge_mvp_with_stats(mvps, years):
    """Merge MVP voting data with player statistics."""
    all_players = []
    
    for year in years:
        print(f"Processing {year}...")
        player_stats = load_player_stats(year)
        
        if player_stats is None:
            continue
        
        # Get MVP data for this year
        mvp_year = mvps[mvps["Year"] == year].copy()
        
        # Merge on Player and Year
        merged = player_stats.merge(
            mvp_year[["Player", "Year", "Share", "Rank"]],
            on=["Player", "Year"],
            how="left"
        )
        
        # Fill NaN values for players who didn't receive MVP votes
        merged["Share"] = merged["Share"].fillna(0)
        merged["Rank"] = merged["Rank"].fillna(999)  # High number for non-candidates
        
        # Create binary MVP winner label (Rank == 1)
        merged["MVP"] = (merged["Rank"] == 1).astype(int)
        
        all_players.append(merged)
    
    return pd.concat(all_players, ignore_index=True)

def create_features(df):
    """Create engineered features for the model."""
    # Convert percentage columns to numeric
    percentage_cols = ["FG%", "3P%", "FT%"]
    # eFG% might not exist in older data, so check first
    if "eFG%" in df.columns:
        percentage_cols.append("eFG%")
    
    for col in percentage_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert other numeric columns
    numeric_cols = ["G", "MP", "PTS", "TRB", "AST", "STL", "BLK", "PF", 
                   "WS", "WS/48", "Age"]
    # TOV might be named differently or not exist in older data
    for tov_name in ["TOV", "TO", "Turnovers"]:
        if tov_name in df.columns:
            numeric_cols.append(tov_name)
            break
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create efficiency metrics
    if "PTS" in df.columns and "MP" in df.columns:
        df["PTS_per_36"] = (df["PTS"] / df["MP"]) * 36
        df["TRB_per_36"] = (df["TRB"] / df["MP"]) * 36
        df["AST_per_36"] = (df["AST"] / df["MP"]) * 36
    
    # Create advanced metrics
    # Find TOV column (might have different names)
    tov_col = None
    for tov_name in ["TOV", "TO", "Turnovers"]:
        if tov_name in df.columns:
            tov_col = tov_name
            break
    
    if all(col in df.columns for col in ["PTS", "TRB", "AST", "STL", "BLK", "PF"]) and tov_col:
        df["PER_like"] = (df["PTS"] + df["TRB"] * 1.2 + df["AST"] * 1.5 + 
                         df["STL"] * 3 + df["BLK"] * 3 - df[tov_col] * 1 - df["PF"] * 0.5)
    elif all(col in df.columns for col in ["PTS", "TRB", "AST", "STL", "BLK", "PF"]):
        # PER-like without turnovers
        df["PER_like"] = (df["PTS"] + df["TRB"] * 1.2 + df["AST"] * 1.5 + 
                         df["STL"] * 3 + df["BLK"] * 3 - df["PF"] * 0.5)
    
    # Team wins (we'll need to scrape this separately or use a proxy)
    # For now, we'll use WS (Win Shares) as a proxy for team success contribution
    
    # Usage rate proxy (points + assists + rebounds)
    if all(col in df.columns for col in ["PTS", "AST", "TRB"]):
        df["Usage_proxy"] = df["PTS"] + df["AST"] * 2 + df["TRB"] * 1.5
    
    # Fill NaN values with 0 for numeric columns
    numeric_cols_all = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols_all] = df[numeric_cols_all].fillna(0)
    
    return df

def get_feature_columns():
    """Return list of feature columns to use in the model."""
    # Base features (most common)
    base_features = [
        "Age", "G", "MP", "PTS", "TRB", "AST", "STL", "BLK", "PF",
        "FG%", "3P%", "FT%", "WS", "WS/48",
        "PTS_per_36", "TRB_per_36", "AST_per_36",
        "PER_like", "Usage_proxy"
    ]
    # TOV might have different names, will be handled dynamically
    return base_features

def prepare_dataset():
    """Main function to prepare the complete dataset."""
    print("Loading MVP data...")
    mvps = load_mvp_data()
    
    print("Loading and merging player statistics...")
    years = sorted(mvps["Year"].unique())
    dataset = merge_mvp_with_stats(mvps, years)
    
    print("Creating features...")
    dataset = create_features(dataset)
    
    # Select relevant columns
    feature_cols = get_feature_columns()
    # Filter to only columns that exist
    feature_cols = [col for col in feature_cols if col in dataset.columns]
    
    # Keep identifier columns
    keep_cols = ["Player", "Year", "Tm", "MVP", "Share", "Rank"] + feature_cols
    keep_cols = [col for col in keep_cols if col in dataset.columns]
    
    dataset = dataset[keep_cols]
    
    # Save processed dataset
    dataset.to_csv("nba_dataset.csv", index=False)
    print(f"Saved dataset with {len(dataset)} rows and {len(dataset.columns)} columns")
    print(f"Features: {feature_cols}")
    
    return dataset

if __name__ == "__main__":
    dataset = prepare_dataset()
    print("\nDataset preview:")
    print(dataset.head())
    print(f"\nMVP winners: {dataset['MVP'].sum()}")
    print(f"Total players: {len(dataset)}")
