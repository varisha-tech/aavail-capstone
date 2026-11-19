#!/usr/bin/env python3
"""
Performance monitoring — tracks prediction drift and model metrics over time.
"""

import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

MONITOR_LOG = os.path.join(os.path.dirname(__file__), '..', 'logs', 'monitor.log')
METRICS_FILE = os.path.join(os.path.dirname(__file__), 'metrics.json')

os.makedirs(os.path.dirname(MONITOR_LOG), exist_ok=True)


def log_prediction(country, year, month, predicted_revenue):
    """Log each prediction for drift monitoring."""
    entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'country': country,
        'year': year,
        'month': month,
        'predicted_revenue': predicted_revenue
    }
    with open(MONITOR_LOG, 'a') as f:
        f.write(json.dumps(entry) + '\n')
    logger.info(f"Prediction logged: {entry}")


def get_performance_summary():
    """Return current model RMSE metrics from last training run."""
    if not os.path.exists(METRICS_FILE):
        return {'error': 'No metrics available. Run training first.'}
    with open(METRICS_FILE) as f:
        return json.load(f)


def check_for_drift(threshold_rmse=500.0):
    """
    Simple drift check: warn if any model RMSE exceeds threshold.
    Returns a list of warnings.
    """
    warnings = []
    summary = get_performance_summary()
    if 'error' in summary:
        return [summary['error']]

    for country, metrics in summary.items():
        best_rmse = metrics.get('best_rmse', 0)
        if best_rmse > threshold_rmse:
            msg = f"DRIFT WARNING: {country} best_rmse={best_rmse} exceeds threshold={threshold_rmse}"
            warnings.append(msg)
            logger.warning(msg)

    return warnings if warnings else ['No drift detected.']


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(get_performance_summary(), indent=2))
    print(check_for_drift())
