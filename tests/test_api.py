#!/usr/bin/env python3
"""
Unit tests for the AAVAIL API.
Tests endpoints: /ping, /predict, /train, /metrics, /logs
"""

import sys
import os
import json
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.app import app


class TestAPIEndpoints(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    # ------------------------------------------------------------------
    # /ping
    # ------------------------------------------------------------------
    def test_ping_returns_200(self):
        response = self.client.get('/ping')
        self.assertEqual(response.status_code, 200)

    def test_ping_returns_alive_status(self):
        response = self.client.get('/ping')
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'alive')

    # ------------------------------------------------------------------
    # /predict
    # ------------------------------------------------------------------
    def test_predict_valid_country(self):
        payload = {'country': 'all', 'year': 2018, 'month': 6}
        response = self.client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)

    def test_predict_specific_country(self):
        payload = {'country': 'united_states', 'year': 2018, 'month': 3}
        response = self.client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('predicted_revenue', data)

    def test_predict_all_countries(self):
        """API must handle predictions for all countries and 'all' combined."""
        countries = ['all', 'united_states', 'united_kingdom', 'france', 'germany', 'japan']
        for country in countries:
            payload = {'country': country, 'year': 2018, 'month': 1}
            response = self.client.post(
                '/predict',
                data=json.dumps(payload),
                content_type='application/json'
            )
            self.assertEqual(response.status_code, 200,
                             msg=f"Failed for country={country}")

    def test_predict_invalid_country_returns_400(self):
        payload = {'country': 'mars', 'year': 2018, 'month': 1}
        response = self.client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_predict_non_json_returns_400(self):
        response = self.client.post('/predict', data='not json')
        self.assertEqual(response.status_code, 400)

    def test_predict_response_has_required_fields(self):
        payload = {'country': 'all', 'year': 2018, 'month': 6}
        response = self.client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        data = json.loads(response.data)
        for field in ['country', 'year', 'month', 'predicted_revenue']:
            self.assertIn(field, data)

    # ------------------------------------------------------------------
    # /logs
    # ------------------------------------------------------------------
    def test_logs_endpoint_returns_200(self):
        response = self.client.get('/logs')
        self.assertEqual(response.status_code, 200)

    def test_logs_endpoint_returns_list(self):
        response = self.client.get('/logs')
        data = json.loads(response.data)
        self.assertIn('logs', data)
        self.assertIsInstance(data['logs'], list)


if __name__ == '__main__':
    unittest.main()
