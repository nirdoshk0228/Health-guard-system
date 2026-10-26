"""Build claim-, patient- and provider-level fraud features with DuckDB SQL."""
from __future__ import annotations

import duckdb
import pandas as pd

from .config import FRAUD_FILES, FRAUD_RAW, PROCESSED_DIR, SQL_DIR, find_file

SQL_STEPS = ["01_claims.sql", "02_patients_physicians.sql", "03_providers.sql"]

# ---- Feature sets consumed by the models -------------------------------------------------
# (claim_after_death is computed as an integrity rule in SQL but is constant in this dataset,
#  so it is reported in the dashboard rather than fed to the models)
CLAIM_FEATURES = [
    "is_inpatient", "amt_reimbursed", "amt_deductible", "n_diag", "n_proc",
    "claim_duration_days", "los_days", "n_physicians", "missing_attending",
    "attending_is_operating", "admit_matches_primary", "amt_peer_ratio", "amt_peer_z",
    "dup_same_day", "patient_age", "patient_cc_count", "patient_renal",
    "patient_annual_reimb", "patient_n_claims",
    "patient_n_providers", "phys_n_claims", "phys_n_providers",
]

PROVIDER_FEATURES = [
    "n_claims", "n_ip_claims", "ip_share", "n_patients", "claims_per_patient",
    "total_reimbursed", "mean_reimbursed", "max_reimbursed", "std_reimbursed",
    "mean_ip_reimbursed", "mean_deductible", "mean_los", "mean_claim_duration",
    "n_attending_phys", "n_operating_phys", "claims_per_attending", "mean_n_diag",
    "mean_n_proc", "mean_n_physicians", "share_attending_is_operating",
    "share_missing_attending", "mean_peer_ratio", "max_peer_ratio", "share_peer_ratio_gt2",
    "share_dup_same_day", "mean_patient_age", "mean_patient_cc",
    "share_renal_patients", "mean_patient_annual_reimb", "mean_patient_n_providers",
    "mean_phys_n_providers", "n_states", "n_counties", "n_distinct_drg",
    "n_distinct_primary_diag",
]

PATIENT_FEATURES = [
    "age", "cc_count", "renal_disease", "is_deceased", "n_claims", "n_ip_claims",
    "n_providers", "n_attending_phys", "total_reimbursed", "mean_peer_ratio",
    "dup_same_day_claims",
]


def build(verbose: bool = True) -> dict[str, pd.DataFrame]:
    con = duckdb.connect()
    for name, prefix in [("raw_bene", FRAUD_FILES["beneficiary"]),
                         ("raw_ip", FRAUD_FILES["inpatient"]),
                         ("raw_op", FRAUD_FILES["outpatient"]),
                         ("raw_labels", FRAUD_FILES["labels"])]:
        path = find_file(FRAUD_RAW, prefix).as_posix()
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_csv('{path}', all_varchar=true, header=true)")
        if verbose:
            n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            print(f"  loaded {name:<11} {n:>8,} rows")

    for step in SQL_STEPS:
        con.execute((SQL_DIR / step).read_text())
        if verbose:
            print(f"  ran {step}")

    out = {
        # ORDER BY makes row order (and therefore every downstream split) reproducible
        "claims": con.execute("SELECT c.*, p.is_fraud AS provider_is_fraud FROM claims c "
                              "JOIN providers p USING (provider) ORDER BY claim_id").df(),
        "patients": con.execute("SELECT * FROM patients ORDER BY bene_id").df(),
        "providers": con.execute("SELECT * FROM providers ORDER BY provider").df(),
    }
    for k, df in out.items():
        df.to_parquet(PROCESSED_DIR / f"fraud_{k}.parquet", index=False)
        if verbose:
            print(f"  wrote fraud_{k}.parquet  {df.shape}")
    con.close()
    return out


if __name__ == "__main__":
    build()
