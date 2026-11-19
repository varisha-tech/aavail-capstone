#!/usr/bin/env python3
"""
AAVAIL Capstone - Production API
IBM AI Workflow Capstone Project
"""

import os
import re
import json
import logging
import numpy as np
from flask import Flask, request, jsonify
from model.model import load_model, predict

# Setup logging
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'api.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

COUNTRIES = ['all', 'united_states', 'united_kingdom', 'france', 'germany', 'japan']

@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint."""
    logger.info("PING endpoint called")
    return jsonify({'status': 'alive', 'message': 'AAVAIL API is running'})

@app.route('/predict', methods=['POST'])
def make_prediction():
    """
    Predict revenue for a given country and date range.
    Accepts JSON: { "country": "united_states", "year": 2018, "month": 1 }
    """
    if not request.is_json:
        logger.error("Request content type is not JSON")
        return jsonify({'error': 'Content-Type must be application/json'}), 400

    data = request.get_json()
    country = data.get('country', 'all')
    year = data.get('year', 2018)
    month = data.get('month', 1)

    if country not in COUNTRIES:
        logger.warning(f"Invalid country requested: {country}")
        return jsonify({'error': f'Country must be one of {COUNTRIES}'}), 400

    logger.info(f"Prediction requested for country={country}, year={year}, month={month}")

    try:
        model = load_model(country)
        prediction = predict(model, year, month)
        result = {
            'country': country,
            'year': year,
            'month': month,
            'predicted_revenue': round(float(prediction), 2)
        }
        logger.info(f"Prediction successful: {result}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    """Trigger model retraining."""
    from model.model import train_models
    logger.info("Training triggered via API")
    try:
        train_models()
        logger.info("Training completed successfully")
        return jsonify({'status': 'success', 'message': 'Models trained successfully'})
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Return model performance metrics."""
    metrics_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'metrics.json')
    if not os.path.exists(metrics_path):
        return jsonify({'error': 'No metrics found. Train the model first.'}), 404
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    return jsonify(data)

@app.route('/logs', methods=['GET'])
def get_logs():
    """Return recent API logs."""
    log_file = os.path.join(LOG_DIR, 'api.log')
    if not os.path.exists(log_file):
        return jsonify({'logs': []})
    with open(log_file, 'r') as f:
        lines = f.readlines()[-50:]  # last 50 lines
    return jsonify({'logs': [l.strip() for l in lines]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
