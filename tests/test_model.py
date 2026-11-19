#!/usr/bin/env python3
"""
Unit tests for the AAVAIL model — training, loading, prediction.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.model import (
    ingest_data,
    build_features,
    train_models,
    load_model,
    predict,
    COUNTRIES,
    FEATURES,
)


class TestDataIngestion(unittest.TestCase):

    def test_ingest_data_returns_dataframe(self):
        import pandas as pd
        df = ingest_data('all')
        self.assertIsInstance(df, pd.DataFrame)

    def test_ingest_data_has_required_columns(self):
        df = ingest_data('all')
        for col in FEATURES + ['revenue']:
            self.assertIn(col, df.columns, msg=f"Missing column: {col}")

    def test_ingest_data_no_nulls(self):
        df = ingest_data('all')
        self.assertFalse(df.isnull().values.any(), "Data contains null values")

    def test_ingest_data_all_countries(self):
        for country in COUNTRIES:
            df = ingest_data(country)
            self.assertGreater(len(df), 0, msg=f"Empty dataframe for {country}")


class TestFeatureEngineering(unittest.TestCase):

    def test_build_features_returns_arrays(self):
        df = ingest_data('all')
        X, y = build_features(df)
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

    def test_build_features_shapes_match(self):
        df = ingest_data('all')
        X, y = build_features(df)
        self.assertEqual(X.shape[0], y.shape[0])

    def test_build_features_correct_num_columns(self):
        df = ingest_data('all')
        X, y = build_features(df)
        self.assertEqual(X.shape[1], len(FEATURES))


class TestModelTraining(unittest.TestCase):

    def test_train_models_runs_without_error(self):
        try:
            train_models()
        except Exception as e:
            self.fail(f"train_models() raised an exception: {e}")

    def test_model_files_created_for_all_countries(self):
        import pickle
        train_models()
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
        for country in COUNTRIES:
            path = os.path.join(model_dir, f'model_{country}.pkl')
            self.assertTrue(os.path.exists(path), msg=f"Model file missing for {country}")

    def test_metrics_file_created(self):
        import json
        train_models()
        metrics_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'metrics.json')
        self.assertTrue(os.path.exists(metrics_path))
        with open(metrics_path) as f:
            metrics = json.load(f)
        self.assertIn('all', metrics)

    def test_multiple_models_were_compared(self):
        import json
        train_models()
        metrics_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'metrics.json')
        with open(metrics_path) as f:
            metrics = json.load(f)
        country_metrics = metrics['all']
        model_names = [k for k in country_metrics if k not in ('best_model', 'best_rmse')]
        self.assertGreaterEqual(len(model_names), 2, "At least 2 models must be compared")


class TestModelPrediction(unittest.TestCase):

    def test_load_model_returns_object(self):
        model = load_model('all')
        self.assertIsNotNone(model)

    def test_predict_returns_float(self):
        model = load_model('all')
        result = predict(model, 2018, 6)
        self.assertIsInstance(result, float)

    def test_predict_positive_revenue(self):
        model = load_model('all')
        result = predict(model, 2018, 6)
        self.assertGreater(result, 0)

    def test_predict_all_countries(self):
        for country in COUNTRIES:
            model = load_model(country)
            result = predict(model, 2018, 1)
            self.assertIsInstance(result, float, msg=f"Failed for {country}")


if __name__ == '__main__':
    unittest.main()
