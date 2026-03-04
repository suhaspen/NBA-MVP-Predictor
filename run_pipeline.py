"""
Complete pipeline script to run the entire NBA MVP prediction workflow.
Run this script to scrape data, preprocess, train, and make predictions.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error: {description} failed!")
        return False
    return True

def main():
    """Run the complete pipeline."""
    print("NBA MVP Predictor - Complete Pipeline")
    print("="*80)
    
    # Check if data already exists
    skip_scraping = os.path.exists("player") and len(os.listdir("player")) > 1
    skip_preprocessing = os.path.exists("nba_dataset.csv")
    skip_training = os.path.exists("models") and os.path.exists(os.path.join("models", "mvp_model.pkl"))
    
    # Step 1: Scrape player stats (30 years of NBA data)
    if not skip_scraping:
        if not run_command("python scrape_player_stats.py", "Step 1: Scraping player statistics"):
            return
    else:
        print("\nSkipping scraping - player stats already exist")
    
    # Step 2: Preprocess data (pandas: clean, merge, 5000+ data points)
    if not skip_preprocessing:
        if not run_command("python preprocess_data.py", "Step 2: Preprocessing data"):
            return
    else:
        print("\nSkipping preprocessing - dataset already exists")
    
    # Step 3: Train Logistic Regression and XGBoost (cross-validation)
    if not skip_training:
        if not run_command("python train_sklearn.py", "Step 3: Training models (LogReg + XGBoost)"):
            return
    else:
        print("\nSkipping training - models already exist")
        response = input("Do you want to retrain? (y/n): ")
        if response.lower() == 'y':
            if not run_command("python train_sklearn.py", "Step 3: Training models"):
                return
    
    # Step 4: Predictions (CLI or start API)
    print("\n" + "="*80)
    print("Step 4: Predictions (run 'python run_app.py' for REST API)")
    print("="*80)
    
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        run_command("python run_app.py", "Starting REST API")
    elif len(sys.argv) > 1:
        year = sys.argv[1]
        top_n = sys.argv[2] if len(sys.argv) > 2 else "10"
        run_command(f"python predict.py {year} {top_n}", f"Predicting MVP for {year}")
    else:
        run_command("python predict.py", "Predicting MVP for recent season")
    
    print("\n" + "="*80)
    print("Pipeline complete!")
    print("="*80)

if __name__ == "__main__":
    main()
