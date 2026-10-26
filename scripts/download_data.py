"""Download the two Kaggle datasets used by HealthGuard AI into data/raw/.

Datasets
--------
1. Healthcare Provider Fraud Detection Analysis  (rohitrox/healthcare-provider-fraud-detection-analysis)
   Medicare-style inpatient + outpatient claims, beneficiary details and provider-level fraud labels.
2. Diabetes 130-US Hospitals 1999-2008  (brandao/diabetes)
   ~100k inpatient encounters with a readmission outcome (<30 days, >30 days, NO).

Authentication
--------------
Either works:
  * Put kaggle.json in ~/.kaggle/  (Kaggle -> Settings -> API -> Create New Token), or
  * export KAGGLE_USERNAME=... and KAGGLE_KEY=...

Usage
-----
    python scripts/download_data.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from healthguard.config import (  # noqa: E402
    FRAUD_FILES,
    FRAUD_RAW,
    KAGGLE_FRAUD,
    KAGGLE_READMIT,
    READMIT_FILES,
    READMIT_RAW,
)


def _download_with_kagglehub(slug: str) -> Path:
    import kagglehub

    return Path(kagglehub.dataset_download(slug))


def _download_with_cli(slug: str, dest: Path) -> Path:
    tmp = dest / "_download"
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(tmp)], check=True)
    for z in tmp.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(tmp)
    return tmp


def fetch(slug: str, dest: Path, wanted: dict[str, str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    have = all(any(p.name.startswith(pref) for p in dest.rglob("*.csv")) for pref in wanted.values())
    if have:
        print(f"[skip] {slug} already in {dest}")
        return

    print(f"[download] {slug}")
    try:
        src = _download_with_kagglehub(slug)
    except Exception as exc:  # kagglehub missing or failed -> try the kaggle CLI
        print(f"  kagglehub failed ({exc}); trying kaggle CLI")
        src = _download_with_cli(slug, dest)

    for pref in wanted.values():
        files = [p for p in src.rglob("*.csv") if p.name.startswith(pref)]
        if not files:
            raise FileNotFoundError(f"'{pref}*.csv' not found in downloaded dataset {slug}")
        for f in files:
            shutil.copy2(f, dest / f.name)
            print(f"  -> {dest / f.name}")

    shutil.rmtree(dest / "_download", ignore_errors=True)


if __name__ == "__main__":
    fetch(KAGGLE_FRAUD, FRAUD_RAW, FRAUD_FILES)
    fetch(KAGGLE_READMIT, READMIT_RAW, READMIT_FILES)
    print("Done. Next: python run_pipeline.py")
