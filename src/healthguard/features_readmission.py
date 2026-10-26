"""Feature engineering for 30-day readmission (Diabetes 130-US Hospitals dataset)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, READMIT_FILES, READMIT_RAW, find_file

# Discharge dispositions where a readmission is impossible / not meaningful
# (11 = expired, 13/14 = hospice, 19/20/21 = expired variants)
EXCLUDED_DISCHARGE = {11, 13, 14, 19, 20, 21}

MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

NUMERIC_FEATURES = [
    "age_years", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency", "number_inpatient",
    "total_prior_visits", "number_diagnoses", "n_meds_active", "n_meds_changed",
    "n_meds_up", "n_meds_down", "med_change", "on_diabetes_med", "gender_male",
    "a1c_tested", "a1c_high", "glu_tested", "glu_high", "prior_encounters_in_data",
    "payer_missing", "diag_1_is_diabetes", "any_diag_diabetes", "meds_per_day",
    "labs_per_day",
]
CATEGORICAL_FEATURES = [
    "race", "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "medical_specialty_grp", "diag_1_grp", "diag_2_grp", "diag_3_grp", "insulin_status",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "readmit_30d"


def icd9_group(code) -> str:
    """Map an ICD-9 code to a clinical chapter (Strack et al., 2014 grouping)."""
    if code is None or (isinstance(code, float) and np.isnan(code)) or code == "?":
        return "Missing"
    s = str(code)
    if s.startswith(("V", "E")):
        return "Other"
    try:
        v = float(s)
    except ValueError:
        return "Other"
    if 390 <= v <= 459 or int(v) == 785:
        return "Circulatory"
    if 460 <= v <= 519 or int(v) == 786:
        return "Respiratory"
    if 520 <= v <= 579 or int(v) == 787:
        return "Digestive"
    if int(v) == 250:
        return "Diabetes"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or int(v) == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


def _age_mid(bucket: str) -> float:
    lo, hi = bucket.strip("[)").split("-")
    return (int(lo) + int(hi)) / 2


def build(verbose: bool = True) -> pd.DataFrame:
    path = find_file(READMIT_RAW, READMIT_FILES["encounters"])
    df = pd.read_csv(path, na_values=["?"], low_memory=False, keep_default_na=False)
    raw_n = len(df)

    df = df[~df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE)].copy()
    df = df.sort_values("encounter_id")  # encounter_id increases with time

    out = pd.DataFrame(index=df.index)
    out["encounter_id"] = df["encounter_id"]
    out["patient_nbr"] = df["patient_nbr"]
    out[TARGET] = (df["readmitted"] == "<30").astype(int)

    out["age_years"] = df["age"].map(_age_mid)
    out["gender_male"] = (df["gender"] == "Male").astype(int)
    for c in ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
              "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]:
        out[c] = df[c].astype(float)
    out["total_prior_visits"] = out["number_outpatient"] + out["number_emergency"] + out["number_inpatient"]
    out["meds_per_day"] = out["num_medications"] / out["time_in_hospital"]
    out["labs_per_day"] = out["num_lab_procedures"] / out["time_in_hospital"]

    meds = df[MED_COLS]
    out["n_meds_active"] = meds.isin(["Steady", "Up", "Down"]).sum(axis=1)
    out["n_meds_up"] = (meds == "Up").sum(axis=1)
    out["n_meds_down"] = (meds == "Down").sum(axis=1)
    out["n_meds_changed"] = out["n_meds_up"] + out["n_meds_down"]
    out["med_change"] = (df["change"] == "Ch").astype(int)
    out["on_diabetes_med"] = (df["diabetesMed"] == "Yes").astype(int)
    out["insulin_status"] = df["insulin"].fillna("No")

    a1c = df["A1Cresult"].replace({"": "None"}).fillna("None")
    glu = df["max_glu_serum"].replace({"": "None"}).fillna("None")
    out["a1c_tested"] = (a1c != "None").astype(int)
    out["a1c_high"] = a1c.isin([">7", ">8"]).astype(int)
    out["glu_tested"] = (glu != "None").astype(int)
    out["glu_high"] = glu.isin([">200", ">300"]).astype(int)

    out["payer_missing"] = df["payer_code"].isna().astype(int)
    out["race"] = df["race"].fillna("Unknown")
    for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        out[c] = df[c].astype(str)

    spec = df["medical_specialty"].fillna("Missing")
    top = spec.value_counts().index[:12]
    out["medical_specialty_grp"] = spec.where(spec.isin(top), "Other")

    for i in (1, 2, 3):
        out[f"diag_{i}_grp"] = df[f"diag_{i}"].map(icd9_group)
    out["diag_1_is_diabetes"] = (out["diag_1_grp"] == "Diabetes").astype(int)
    out["any_diag_diabetes"] = (out[["diag_1_grp", "diag_2_grp", "diag_3_grp"]] == "Diabetes").any(axis=1).astype(int)

    # How many earlier encounters this patient already has in the data (history signal, no leakage:
    # only encounters with a smaller encounter_id are counted)
    out["prior_encounters_in_data"] = out.groupby("patient_nbr").cumcount()

    for c in CATEGORICAL_FEATURES:
        out[c] = out[c].astype("category")

    out = out.reset_index(drop=True)
    out.to_parquet(PROCESSED_DIR / "readmission_features.parquet", index=False)
    if verbose:
        print(f"  {raw_n:,} encounters -> {len(out):,} after excluding expired/hospice discharges; "
              f"30-day readmission rate {out[TARGET].mean():.1%}")
    return out


if __name__ == "__main__":
    build()
