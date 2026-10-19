"""Generate all EDA figures into reports/figures/eda/ (script version of notebook 01)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from heartml import config  # noqa: E402
from heartml.data import load_clean  # noqa: E402
from heartml.eda import run_eda  # noqa: E402


def main() -> None:
    raw = pd.read_csv(config.RAW_DATA_PATH)
    df = load_clean()
    summary = run_eda(raw, df)
    print(json.dumps({k: summary[k] for k in
                      ("raw_rows", "clean_rows", "missing_raw", "class_counts",
                       "top_target_correlations", "figures")}, indent=2))


if __name__ == "__main__":
    main()
