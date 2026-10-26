"""Load scored outputs / models and turn single rows into explainable RiskCases."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from . import explain
from .config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from .features_fraud import CLAIM_FEATURES, PROVIDER_FEATURES
from .features_readmission import CATEGORICAL_FEATURES, FEATURES as READMIT_FEATURES
from .llm_explainer import RiskCase

CLAIM_COLS = ["claim_id", "bene_id", "provider", "claim_start", "drg_code", "diag_1", "attending_phys",
              "provider_is_fraud", "claim_fraud_prob", "anomaly_score", "anomaly_pct",
              "provider_fraud_prob", "provider_risk_tier", "risk_score", "risk_tier"] + CLAIM_FEATURES


def load_metrics() -> dict:
    out = {}
    for name in ("fraud", "readmission"):
        p = REPORTS_DIR / f"metrics_{name}.json"
        out[name] = json.loads(p.read_text()) if p.exists() else {}
    return out


def load_tables() -> dict[str, pd.DataFrame]:
    claims_cols = list(dict.fromkeys(CLAIM_COLS))
    return {
        "providers": pd.read_parquet(PROCESSED_DIR / "scored_providers.parquet"),
        "provider_shap": pd.read_parquet(PROCESSED_DIR / "provider_shap.parquet"),
        "claims": pd.read_parquet(PROCESSED_DIR / "scored_claims.parquet", columns=claims_cols),
        "patients": pd.read_parquet(PROCESSED_DIR / "scored_patients.parquet"),
        "readmission": pd.read_parquet(PROCESSED_DIR / "scored_readmission_test.parquet"),
    }


def load_models() -> dict:
    return {"fraud": joblib.load(MODELS_DIR / "fraud_models.joblib"),
            "readmission": joblib.load(MODELS_DIR / "readmission_model.joblib")}


def _attach_peers(drivers: list[dict], peers: pd.DataFrame) -> list[dict]:
    """Add the peer-group median for each numeric driver (binary 0/1 flags get none)."""
    for d in drivers:
        f, v = d["feature"], None
        if f in peers.columns:
            col = peers[f].dropna()
            if not col.isin([0, 1]).all():
                v = col.median()
        d["peer"] = None if v is None or np.isnan(v) else float(v)
    return drivers


def provider_case(provider_id: str, providers: pd.DataFrame, provider_shap: pd.DataFrame, k: int = 6) -> RiskCase:
    row = providers.loc[providers["provider"] == provider_id].iloc[0]
    sv = provider_shap.loc[provider_shap["provider"] == provider_id, PROVIDER_FEATURES].to_numpy()[0]
    drivers = explain.top_factors(sv, row[PROVIDER_FEATURES], k)
    drivers = _attach_peers(drivers, providers[PROVIDER_FEATURES])
    return RiskCase("provider", provider_id, float(row["fraud_prob"]), row["risk_tier"], drivers,
                    extra={"anomaly_percentile": round(float(row["anomaly_pct"]), 3),
                           "total_reimbursed_usd": float(row["total_reimbursed"]),
                           "claims": int(row["n_claims"])})


def claim_case(claim_row: pd.Series, claims: pd.DataFrame, fraud_models: dict, k: int = 6) -> RiskCase:
    X = claim_row[CLAIM_FEATURES].to_frame().T.astype(float)
    sv, _ = explain.shap_values(fraud_models["claim"], X)
    drivers = explain.top_factors(sv[0], X.iloc[0], k)
    same_setting = claims.loc[claims["is_inpatient"] == claim_row["is_inpatient"], CLAIM_FEATURES]
    drivers = _attach_peers(drivers, same_setting)
    return RiskCase("claim", claim_row["claim_id"], float(claim_row["risk_score"]), claim_row["risk_tier"], drivers,
                    extra={"claim_model_prob": round(float(claim_row["claim_fraud_prob"]), 3),
                           "provider_fraud_prob": round(float(claim_row["provider_fraud_prob"]), 3),
                           "anomaly_percentile": round(float(claim_row["anomaly_pct"]), 3),
                           "provider": claim_row["provider"], "setting": "inpatient" if claim_row["is_inpatient"] else "outpatient"})


def readmission_frame(values: dict, levels: dict) -> pd.DataFrame:
    """Build a 1-row model frame from a dict of feature values, with the training category levels."""
    row = pd.DataFrame([{f: values.get(f) for f in READMIT_FEATURES}])
    for c in CATEGORICAL_FEATURES:
        row[c] = pd.Categorical(row[c].astype(str), categories=levels[c])
    for c in READMIT_FEATURES:
        if c not in CATEGORICAL_FEATURES:
            row[c] = row[c].astype(float)
    return row


def readmission_case(entity_id: str, X: pd.DataFrame, bundle: dict, cohort: pd.DataFrame, k: int = 6) -> RiskCase:
    from .train_readmission import risk_band

    p = float(bundle["model"].predict_proba(X)[:, 1][0])
    sv, _ = explain.shap_values(bundle["model"], X)
    drivers = explain.top_factors(sv[0], X.iloc[0], k)
    num_cols = [c for c in READMIT_FEATURES if c not in CATEGORICAL_FEATURES]
    drivers = _attach_peers(drivers, cohort[num_cols])
    return RiskCase("readmission", entity_id, p, str(risk_band([p], bundle["band_cuts"])[0]), drivers,
                    extra={"cohort_readmission_rate": round(float(cohort["readmit_30d"].mean()), 3)})
