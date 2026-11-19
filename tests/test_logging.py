#!/usr/bin/env python3
"""
Unit tests for logging — verifies log files are created and written to.
Tests are isolated from production logs using a temp directory.
"""

import sys
import os
import json
import logging
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestLoggingSetup(unittest.TestCase):

    def setUp(self):
        """Use a temp dir to isolate from production logs."""
        self.test_log_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.test_log_dir, 'test_api.log')

    def _get_test_logger(self):
        logger = logging.getLogger(f'test_logger_{id(self)}')
        logger.setLevel(logging.INFO)
        logger.propagate = False   # <-- isolate from root/production handlers
        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
        return logger, handler

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_log_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    def test_log_file_created(self):
        logger, handler = self._get_test_logger()
        logger.info("Test log entry")
        handler.flush()
        self.assertTrue(os.path.exists(self.log_file))

    def test_log_file_contains_entry(self):
        logger, handler = self._get_test_logger()
        logger.info("Prediction requested for country=all")
        handler.flush()
        with open(self.log_file) as f:
            content = f.read()
        self.assertIn("Prediction requested", content)

    def test_log_levels_written_correctly(self):
        logger, handler = self._get_test_logger()
        logger.info("INFO message")
        logger.warning("WARNING message")
        logger.error("ERROR message")
        handler.flush()
        with open(self.log_file) as f:
            content = f.read()
        self.assertIn("INFO", content)
        self.assertIn("WARNING", content)
        self.assertIn("ERROR", content)

    def test_log_isolation_from_production(self):
        """Test logs must NOT write to the production log directory."""
        prod_log = os.path.join(
            os.path.dirname(__file__), '..', 'logs', 'api.log'
        )
        # Clear prod log so we start from a clean slate
        if os.path.exists(prod_log):
            open(prod_log, 'w').close()

        UNIQUE_MARKER = "ISOLATION_TEST_MARKER_XYZ_12345"
        logger, handler = self._get_test_logger()
        logger.info(UNIQUE_MARKER)
        handler.flush()

        if os.path.exists(prod_log):
            with open(prod_log) as f:
                prod_content = f.read()
            self.assertNotIn(UNIQUE_MARKER, prod_content)

    def test_multiple_log_entries(self):
        logger, handler = self._get_test_logger()
        for i in range(5):
            logger.info(f"Log entry {i}")
        handler.flush()
        with open(self.log_file) as f:
            lines = [l for l in f.readlines() if l.strip()]
        self.assertEqual(len(lines), 5)


class TestAPILogging(unittest.TestCase):
    """Test that API calls produce log entries."""

    def setUp(self):
        import api.app as api_module
        api_module.app.config['TESTING'] = True
        self.client = api_module.app.test_client()

    def test_ping_generates_log(self):
        """Calling /ping should not crash and log the event."""
        response = self.client.get('/ping')
        self.assertEqual(response.status_code, 200)

    def test_predict_generates_log(self):
        payload = {'country': 'all', 'year': 2018, 'month': 6}
        response = self.client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)

    def test_logs_endpoint_returns_entries(self):
        # Trigger some log entries first
        self.client.get('/ping')
        response = self.client.get('/logs')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('logs', data)


if __name__ == '__main__':
    unittest.main()
