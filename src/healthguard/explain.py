"""SHAP explainability helpers (exact TreeSHAP via XGBoost) + human-readable feature labels."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

FEATURE_LABELS: dict[str, str] = {
    # provider level
    "n_claims": "total claims filed", "n_ip_claims": "inpatient claims filed",
    "ip_share": "share of claims that are inpatient", "n_patients": "distinct patients billed",
    "claims_per_patient": "claims per patient", "total_reimbursed": "total amount reimbursed",
    "mean_reimbursed": "average reimbursement per claim", "max_reimbursed": "largest single reimbursement",
    "std_reimbursed": "variability of reimbursement", "mean_ip_reimbursed": "average inpatient reimbursement",
    "mean_deductible": "average deductible paid", "mean_los": "average inpatient length of stay",
    "mean_claim_duration": "average claim duration (days)", "n_attending_phys": "distinct attending physicians",
    "n_operating_phys": "distinct operating physicians", "claims_per_attending": "claims per attending physician",
    "mean_n_diag": "diagnosis codes per claim", "mean_n_proc": "procedure codes per claim",
    "mean_n_physicians": "physicians listed per claim",
    "share_attending_is_operating": "share of claims where attending = operating physician",
    "share_missing_attending": "share of claims missing an attending physician",
    "mean_peer_ratio": "billing vs clinical peers (avg ratio)", "max_peer_ratio": "billing vs clinical peers (max ratio)",
    "share_peer_ratio_gt2": "share of claims billed >2x peer average",
    "share_dup_same_day": "share of duplicate same-day claims", "mean_patient_age": "average patient age",
    "mean_patient_cc": "average chronic conditions per patient", "share_renal_patients": "share of patients with renal disease",
    "mean_patient_annual_reimb": "patients' average annual reimbursement",
    "mean_patient_n_providers": "providers seen per patient (doctor-shopping signal)",
    "mean_phys_n_providers": "providers each physician bills through (network signal)",
    "n_states": "states the patient base spans", "n_counties": "counties the patient base spans",
    "n_distinct_drg": "distinct DRG codes billed", "n_distinct_primary_diag": "distinct primary diagnoses billed",
    # claim level
    "is_inpatient": "inpatient claim", "amt_reimbursed": "amount reimbursed", "amt_deductible": "deductible paid",
    "n_diag": "number of diagnosis codes", "n_proc": "number of procedure codes",
    "claim_duration_days": "claim duration (days)", "los_days": "length of stay (days)",
    "n_physicians": "physicians on the claim", "missing_attending": "no attending physician",
    "attending_is_operating": "attending is also operating physician",
    "admit_matches_primary": "admit diagnosis matches primary diagnosis",
    "amt_peer_ratio": "amount vs peer-group average", "amt_peer_z": "amount z-score vs peer group",
    "dup_same_day": "duplicate same-day claim", "patient_age": "patient age",
    "patient_cc_count": "patient chronic conditions", "patient_renal": "patient has renal disease",
    "patient_annual_reimb": "patient annual reimbursement", "patient_n_claims": "patient's total claims",
    "patient_n_providers": "providers the patient visited", "phys_n_claims": "attending physician claim volume",
    "phys_n_providers": "providers the attending physician bills through",
    # readmission
    "age_years": "age", "time_in_hospital": "length of stay (days)", "num_lab_procedures": "lab tests performed",
    "num_procedures": "procedures performed", "num_medications": "medications administered",
    "number_outpatient": "outpatient visits in prior year", "number_emergency": "emergency visits in prior year",
    "number_inpatient": "inpatient admissions in prior year", "total_prior_visits": "total prior-year visits",
    "number_diagnoses": "number of diagnoses", "n_meds_active": "diabetes medications prescribed",
    "n_meds_changed": "diabetes medications changed", "n_meds_up": "medication doses increased",
    "n_meds_down": "medication doses decreased", "med_change": "any medication change",
    "on_diabetes_med": "on diabetes medication", "gender_male": "male", "a1c_tested": "HbA1c tested",
    "a1c_high": "HbA1c elevated (>7%)", "glu_tested": "max glucose tested", "glu_high": "glucose elevated (>200)",
    "prior_encounters_in_data": "earlier encounters at these hospitals", "payer_missing": "payer unknown",
    "diag_1_is_diabetes": "primary diagnosis is diabetes", "any_diag_diabetes": "diabetes among top-3 diagnoses",
    "meds_per_day": "medications per hospital day", "labs_per_day": "lab tests per hospital day",
    "race": "race", "admission_type_id": "admission type", "discharge_disposition_id": "discharge disposition",
    "admission_source_id": "admission source", "medical_specialty_grp": "admitting specialty",
    "diag_1_grp": "primary diagnosis group", "diag_2_grp": "secondary diagnosis group",
    "diag_3_grp": "tertiary diagnosis group", "insulin_status": "insulin regimen",
}


def label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def shap_values(model: xgb.XGBClassifier, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Exact TreeSHAP contributions (log-odds space). Returns (values[n, f], base_value)."""
    dm = xgb.DMatrix(X, enable_categorical=True)
    contrib = model.get_booster().predict(dm, pred_contribs=True)
    return contrib[:, :-1], float(contrib[0, -1])


def top_factors(row_shap: np.ndarray, row_x: pd.Series, k: int = 5) -> list[dict]:
    """The k features pushing this prediction most (by |SHAP|)."""
    idx = np.argsort(-np.abs(row_shap))[:k]
    out = []
    for i in idx:
        f = row_x.index[i]
        v = row_x.iloc[i]
        out.append({
            "feature": f,
            "label": label(f),
            "value": v if isinstance(v, str) else (None if pd.isna(v) else float(v)),
            "shap": float(row_shap[i]),
            "direction": "increases" if row_shap[i] > 0 else "decreases",
        })
    return out


def save_summary_plot(sv: np.ndarray, X: pd.DataFrame, path, title: str, max_display: int = 15) -> None:
    import shap

    Xp = X.copy()
    for c in Xp.columns:  # beeswarm needs numeric colouring
        if str(Xp[c].dtype) == "category":
            Xp[c] = Xp[c].cat.codes
    Xp.columns = [label(c) for c in Xp.columns]
    plt.figure()
    shap.summary_plot(sv, Xp, max_display=max_display, show=False, plot_size=(9, 6))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close("all")
