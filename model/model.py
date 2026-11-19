#!/usr/bin/env python3
"""
AAVAIL Capstone - Model Training and Prediction
Trains and evaluates multiple models, saves the best performer.
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__))
DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
COUNTRIES = ['all', 'united_states', 'united_kingdom', 'france', 'germany', 'japan']


# ---------------------------------------------------------------------------
# Data ingestion (function/script for automation)
# ---------------------------------------------------------------------------

def ingest_data(country='all'):
    """
    Load and preprocess AAVAIL data for the given country.
    Returns a DataFrame with feature columns and a 'revenue' target.
    """
    csv_path = os.path.join(DATA_DIR, 'aavail-target.csv')
    if not os.path.exists(csv_path):
        logger.warning("Real data not found — generating synthetic data for demo.")
        return _generate_synthetic_data(country)

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df['year']  = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter

    if country != 'all':
        df = df[df['country'] == country]
        if df.empty:
            logger.warning(f"No data for country={country}; using all countries.")
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            df['year']  = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['quarter'] = df['date'].dt.quarter

    df = df.groupby(['year', 'month', 'quarter', 'day_of_week'], as_index=False).agg(
        revenue=('revenue', 'sum'),
        purchases=('purchases', 'sum'),
        stream_views=('stream_views', 'sum'),
    )
    df.fillna(0, inplace=True)
    return df


def _generate_synthetic_data(country='all'):
    """Generate synthetic time-series data when real CSV is absent."""
    np.random.seed(42)
    dates = pd.date_range('2017-01-01', '2019-12-31', freq='D')
    n = len(dates)
    trend    = np.linspace(1000, 5000, n)
    seasonal = 500 * np.sin(2 * np.pi * np.arange(n) / 365)
    noise    = np.random.normal(0, 200, n)
    revenue  = trend + seasonal + noise

    df = pd.DataFrame({
        'date':         dates,
        'revenue':      revenue,
        'purchases':    np.random.randint(10, 200, n).astype(float),
        'stream_views': np.random.randint(100, 2000, n).astype(float),
        'year':         dates.year,
        'month':        dates.month,
        'day_of_week':  dates.dayofweek,
        'quarter':      dates.quarter,
    })
    return df.groupby(['year', 'month', 'quarter', 'day_of_week'], as_index=False).agg(
        revenue=('revenue', 'sum'),
        purchases=('purchases', 'sum'),
        stream_views=('stream_views', 'sum'),
    )


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURES = ['year', 'month', 'quarter', 'day_of_week', 'purchases', 'stream_views']
TARGET   = 'revenue'


def build_features(df):
    X = df[FEATURES].values
    y = df[TARGET].values
    return X, y


# ---------------------------------------------------------------------------
# Model comparison + training
# ---------------------------------------------------------------------------

def _get_candidates():
    return {
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'random_forest':     RandomForestRegressor(n_estimators=100, random_state=42),
        'ridge':             Ridge(alpha=1.0),
    }


def train_models():
    """Train multiple models for each country, save best one + metrics."""
    all_metrics = {}
    os.makedirs(MODEL_DIR, exist_ok=True)

    for country in COUNTRIES:
        logger.info(f"Training models for country={country}")
        df = ingest_data(country)
        X, y = build_features(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        best_rmse   = float('inf')
        best_name   = None
        best_model  = None
        country_metrics = {}

        for name, estimator in _get_candidates().items():
            pipe = Pipeline([('scaler', StandardScaler()), ('model', estimator)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
            country_metrics[name] = {'rmse': round(rmse, 4)}
            logger.info(f"  {name}: RMSE={rmse:.4f}")

            if rmse < best_rmse:
                best_rmse  = rmse
                best_name  = name
                best_model = pipe

        logger.info(f"Best model for {country}: {best_name} (RMSE={best_rmse:.4f})")
        country_metrics['best_model'] = best_name
        country_metrics['best_rmse']  = round(best_rmse, 4)
        all_metrics[country] = country_metrics

        model_path = os.path.join(MODEL_DIR, f'model_{country}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)

    metrics_path = os.path.join(MODEL_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


# ---------------------------------------------------------------------------
# Load + predict
# ---------------------------------------------------------------------------

def load_model(country='all'):
    model_path = os.path.join(MODEL_DIR, f'model_{country}.pkl')
    if not os.path.exists(model_path):
        logger.info(f"Model for {country} not found — training now.")
        train_models()
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def predict(model, year, month):
    """Predict revenue given a year and month."""
    quarter     = (month - 1) // 3 + 1
    day_of_week = 0          # neutral placeholder
    purchases   = 100.0      # median placeholder
    stream_views = 500.0

    X = np.array([[year, month, quarter, day_of_week, purchases, stream_views]])
    return float(model.predict(X)[0])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train_models()
    print("Training complete.")
