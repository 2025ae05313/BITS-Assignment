"""Download the UCI Heart Disease (Cleveland) data and write the cleaned CSV.

Usage:
    python scripts/download_data.py            # UCI first, GitHub mirror as fallback
    python scripts/download_data.py --mirror   # skip UCI, use the mirror directly

Outputs:
    data/raw/heart_cleveland_raw.csv     raw records in UCI column order ('?' -> NaN)
    data/processed/heart_clean.csv       cleaned, binary target, ready for training
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heartml import config  # noqa: E402
from heartml.data import clean, download_raw  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mirror", action="store_true", help="use the GitHub mirror only")
    args = parser.parse_args()

    raw = download_raw(use_mirror=args.mirror)
    df = clean(raw)
    config.CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_DATA_PATH, index=False)
    logging.info("Clean dataset: %d rows x %d cols -> %s", *df.shape, config.CLEAN_DATA_PATH)
    logging.info("Missing values kept for in-pipeline imputation: %s",
                 {k: int(v) for k, v in df.isna().sum().items() if v})
    logging.info("Class balance: %s", df[config.TARGET].value_counts().to_dict())


if __name__ == "__main__":
    main()
