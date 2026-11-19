#!/usr/bin/env python3
"""
Run all unit tests with a single script.
Usage: python run_tests.py
"""

import sys
import os
import unittest

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

def run_all_tests():
    loader = unittest.TestLoader()
    suite  = loader.discover(start_dir='tests', pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total  = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed

    print("\n" + "="*60)
    print(f"  TOTAL : {total}")
    print(f"  PASSED: {passed}")
    print(f"  FAILED: {failed}")
    print("="*60)

    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == '__main__':
    run_all_tests()
