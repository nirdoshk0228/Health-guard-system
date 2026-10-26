"""Central paths and constants for HealthGuard AI."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FRAUD_RAW = RAW_DIR / "fraud"
READMIT_RAW = RAW_DIR / "readmission"
PROCESSED_DIR = DATA_DIR / "processed"
SQL_DIR = ROOT / "sql"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Kaggle dataset slugs
KAGGLE_FRAUD = "rohitrox/healthcare-provider-fraud-detection-analysis"
KAGGLE_READMIT = "brandao/diabetes"

# Files we need from each dataset (matched by prefix so Kaggle's numeric suffixes don't matter)
FRAUD_FILES = {
    "labels": "Train-",
    "beneficiary": "Train_Beneficiarydata",
    "inpatient": "Train_Inpatientdata",
    "outpatient": "Train_Outpatientdata",
}
READMIT_FILES = {"encounters": "diabetic_data"}

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Risk tier cut-offs on the combined 0-1 risk score
TIER_ORDER = ["Critical", "High", "Medium", "Low"]

for _d in (PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def find_file(folder: Path, prefix: str) -> Path:
    """Locate a CSV in `folder` (searched recursively) whose name starts with `prefix`."""
    matches = sorted(p for p in folder.rglob("*.csv") if p.name.startswith(prefix))
    if not matches:
        raise FileNotFoundError(
            f"No file starting with '{prefix}' in {folder}. Run `python scripts/download_data.py` first."
        )
    return matches[0]
