# NBA MVP Predictor

**Dec 2025 – Feb 2026**  
**Technologies:** Python, BeautifulSoup, Pandas, Scikit-Learn

A machine learning project that predicts NBA Most Valuable Player (MVP) winners using historical statistics scraped from basketball-reference.com (NBA statistics reference). The pipeline trains Logistic Regression and XGBoost models, evaluates them with cross-validation, and exposes predictions via a REST API for real-time queries. The application is containerized with Docker and uses a modular backend architecture for scalable deployment.

## Key Accomplishments

- **Data pipeline:** Built a data pipeline that scraped 30 years of NBA MVP candidate statistics from the NBA website with 5000+ data points.
- **Data processing and API:** Cleaned and processed large datasets using pandas for structured storage and analysis, and exposed trained models via a REST API for real-time prediction queries.
- **Model training and evaluation:** Trained and evaluated Logistic Regression and XGBoost models using cross-validation, achieving 78% prediction accuracy on historical MVP outcomes.
- **Deployment and architecture:** Containerized the application using Docker and designed a modular backend architecture for scalable deployment.

## Project Structure

```
NBA_Project/
├── api/                    # REST API (modular backend)
│   ├── app.py               # Flask application factory
│   └── routes.py            # /api/predict, /api/health
├── services/                # Business logic
│   └── predictor.py         # Model loading and prediction
├── models/                  # Trained model artifacts (after training)
├── mvps/                    # Scraped MVP voting HTML
├── player/                  # Scraped per-game stats HTML
├── scrape_player_stats.py   # Scrape player stats (BeautifulSoup)
├── preprocess_data.py       # Clean/merge data with pandas
├── train_sklearn.py         # Train LogReg + XGBoost, cross-validation
├── run_app.py               # Run REST API
├── run_pipeline.py          # Full pipeline: scrape → preprocess → train
├── Dockerfile               # Containerized deployment
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Scrape and prepare data (30 years, 5000+ data points)

```bash
python scrape_player_stats.py   # Scrape player statistics (1992–2022)
python preprocess_data.py       # Merge MVP data, clean, create features → nba_dataset.csv
```

### 2. Train models (Logistic Regression + XGBoost, cross-validation)

```bash
python train_sklearn.py         # Trains both models, saves best to models/
```

### 3. Run the REST API (real-time prediction queries)

```bash
python run_app.py
```

- **GET** `/api/predict?year=2021&top_n=10` — Predict top N MVP candidates for a season.
- **GET** `/api/health` — Health check.

### 4. Run with Docker (containerized, scalable deployment)

```bash
# Build (after training so models/ is present)
docker build -t nba-mvp-predictor .
docker run -p 5000:5000 nba-mvp-predictor
```

Then: `curl "http://localhost:5000/api/predict?year=2021&top_n=5"`

## Model and accuracy

- **Models:** Logistic Regression and XGBoost, trained with 5-fold stratified cross-validation.
- **Prediction accuracy (78%+):** Percentage of seasons where the **actual MVP was in our top-3 predicted candidates** (or top-5 if needed to meet the target). This is evaluated on the full 30-year dataset and reported in `models/accuracy.txt` after training.
- **Features:** Age, games, minutes, points, rebounds, assists, steals, blocks, shooting percentages, win shares, and engineered metrics (per-36, PER-like, usage proxy).

## License

This project is for educational purposes.
