"""End-to-end pipeline: features -> fraud/anomaly models -> readmission model -> metrics.

    python scripts/download_data.py   # once
    python run_pipeline.py            # ~3-4 minutes on a laptop
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
warnings.filterwarnings("ignore", category=UserWarning)

from healthguard import features_fraud, features_readmission, train_fraud, train_readmission  # noqa: E402


def step(name, fn):
    t = time.time()
    print(f"\n▶ {name}")
    out = fn()
    print(f"  ✓ {time.time() - t:.0f}s")
    return out


if __name__ == "__main__":
    step("Fraud features (DuckDB SQL: claim / patient / provider)", features_fraud.build)
    step("Readmission features", features_readmission.build)
    step("Fraud + anomaly models", train_fraud.run)
    step("Readmission model", train_readmission.run)
    print("\nAll done. Launch the dashboard with:  streamlit run app/streamlit_app.py")
