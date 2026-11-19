#!/usr/bin/env python3
"""
Data ingestion script — can be run standalone for automation/scheduling.
Usage: python scripts/ingest.py
       python scripts/ingest.py --country united_states
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.model import ingest_data, COUNTRIES

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='AAVAIL Data Ingestion')
    parser.add_argument('--country', default='all', choices=COUNTRIES,
                        help='Country to ingest data for')
    args = parser.parse_args()

    logger.info(f"Starting ingestion for country={args.country}")
    df = ingest_data(args.country)
    logger.info(f"Ingested {len(df)} rows for country={args.country}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Revenue range: {df['revenue'].min():.2f} – {df['revenue'].max():.2f}")
    return df


if __name__ == '__main__':
    main()
