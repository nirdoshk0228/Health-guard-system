"""Fraud + anomaly models.

* Provider-level XGBoost   -> fraud probability per provider (the label is provider-level)
* Claim-level XGBoost      -> how much a claim resembles claims from fraudulent providers
* Isolation Forests        -> unsupervised anomaly scores for claims, providers and patients
* Risk tiers               -> Critical / High / Medium / Low for review prioritisation

Evaluation is leakage-safe: one provider-stratified 80/20 split is shared by the provider and
claim models, so no provider's claims appear in both train and test. Scores shown in the
dashboard are 5-fold out-of-fold predictions, so every provider is scored by a model that never
saw it.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import explain
from .config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR, TEST_SIZE
from .features_fraud import CLAIM_FEATURES, PATIENT_FEATURES, PROVIDER_FEATURES
from .metrics import (
    best_f1_threshold,
    capture_at_top,
    classification_report,
    curves,
    threshold_for_precision,
)

# Chosen with a small 5-fold CV search on the training split only (see README)
PROVIDER_PARAMS = dict(
    n_estimators=800, learning_rate=0.02, max_depth=3, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0, eval_metric="aucpr",
    tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1,
)
CLAIM_PARAMS = dict(
    n_estimators=400, learning_rate=0.05, max_depth=6, min_child_weight=20,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, eval_metric="aucpr",
    tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1,
)


def _pct_rank(x) -> np.ndarray:
    return pd.Series(x).rank(pct=True).to_numpy()


def _isoforest(X: pd.DataFrame, fit_mask=None, contamination="auto") -> tuple[IsolationForest, np.ndarray]:
    Xf = np.log1p(X.fillna(0).clip(lower=0))  # tame heavy-tailed $ and count features
    model = IsolationForest(n_estimators=300, max_samples=512, contamination=contamination,
                            random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(Xf if fit_mask is None else Xf[fit_mask])
    score = -model.score_samples(Xf)  # higher = more anomalous
    return model, score


def _oof(model_fn, X, y, splitter, groups=None) -> np.ndarray:
    oof = np.zeros(len(X))
    for tr, va in splitter.split(X, y, groups):
        m = model_fn()
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def assign_provider_tier(p, t_high, t_critical, anomaly_pct) -> np.ndarray:
    p, a = np.asarray(p), np.asarray(anomaly_pct)
    return np.select(
        [p >= t_critical, p >= t_high, (p >= t_high / 2) | (a >= 0.97)],
        ["Critical", "High", "Medium"], default="Low",
    )


def quantile_tier(score) -> np.ndarray:
    q = _pct_rank(score)
    # top 2% Critical, next 8% High, next 20% Medium
    return np.select([q > 0.98, q > 0.90, q > 0.70], ["Critical", "High", "Medium"], default="Low")


def run(verbose: bool = True) -> dict:
    prov = pd.read_parquet(PROCESSED_DIR / "fraud_providers.parquet")
    claims = pd.read_parquet(PROCESSED_DIR / "fraud_claims.parquet")
    pats = pd.read_parquet(PROCESSED_DIR / "fraud_patients.parquet")
    metrics: dict = {}

    # ------------------------------------------------------------------ provider model
    Xp, yp = prov[PROVIDER_FEATURES], prov["is_fraud"]
    tr_idx, te_idx = train_test_split(np.arange(len(prov)), test_size=TEST_SIZE,
                                      stratify=yp, random_state=RANDOM_STATE)
    train_providers = set(prov.loc[tr_idx, "provider"])

    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    oof_tr = _oof(lambda: xgb.XGBClassifier(**PROVIDER_PARAMS), Xp.iloc[tr_idx], yp.iloc[tr_idx], skf)
    t_f1 = best_f1_threshold(yp.iloc[tr_idx], oof_tr)
    t_crit = threshold_for_precision(yp.iloc[tr_idx], oof_tr, 0.85)

    m_prov = xgb.XGBClassifier(**PROVIDER_PARAMS).fit(Xp.iloc[tr_idx], yp.iloc[tr_idx])
    p_te = m_prov.predict_proba(Xp.iloc[te_idx])[:, 1]

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    lr.fit(Xp.iloc[tr_idx].fillna(0), yp.iloc[tr_idx])
    p_te_lr = lr.predict_proba(Xp.iloc[te_idx].fillna(0))[:, 1]

    # The holdout has only ~100 fraudulent providers, so also report repeated CV on all providers
    rcv = cross_validate(xgb.XGBClassifier(**PROVIDER_PARAMS), Xp, yp, scoring=["roc_auc", "average_precision"],
                         cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE))
    rcv_lr = cross_validate(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
                            Xp.fillna(0), yp, scoring=["roc_auc", "average_precision"],
                            cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE))

    metrics["provider_fraud"] = {
        "model": "XGBoost",
        "repeated_cv": {
            "xgb_roc_auc_mean": float(rcv["test_roc_auc"].mean()), "xgb_roc_auc_std": float(rcv["test_roc_auc"].std()),
            "xgb_pr_auc_mean": float(rcv["test_average_precision"].mean()), "xgb_pr_auc_std": float(rcv["test_average_precision"].std()),
            "logreg_roc_auc_mean": float(rcv_lr["test_roc_auc"].mean()),
            "logreg_pr_auc_mean": float(rcv_lr["test_average_precision"].mean()),
        },
        "train_cv_roc_auc_oof": float(classification_report(yp.iloc[tr_idx], oof_tr, t_f1)["roc_auc"]),
        "test": classification_report(yp.iloc[te_idx], p_te, t_f1),
        "test_at_critical_threshold": classification_report(yp.iloc[te_idx], p_te, t_crit),
        "test_top10pct": capture_at_top(yp.iloc[te_idx].to_numpy(), p_te, 0.10),
        "baseline_logreg_test": classification_report(yp.iloc[te_idx], p_te_lr, best_f1_threshold(yp.iloc[tr_idx], lr.predict_proba(Xp.iloc[tr_idx].fillna(0))[:, 1])),
        "curves": curves(yp.iloc[te_idx], p_te),
        "thresholds": {"high": t_f1, "critical": t_crit},
    }

    # ------------------------------------------------------------------ claim model
    in_train = claims["provider"].isin(train_providers).to_numpy()
    Xc, yc = claims[CLAIM_FEATURES], claims["provider_is_fraud"]
    m_claim = xgb.XGBClassifier(**CLAIM_PARAMS).fit(Xc[in_train], yc[in_train])
    pc_te = m_claim.predict_proba(Xc[~in_train])[:, 1]
    t_claim = best_f1_threshold(yc[in_train][::5], m_claim.predict_proba(Xc[in_train][::5])[:, 1])
    metrics["claim_fraud"] = {
        "model": "XGBoost (claims inherit provider label; grouped by provider)",
        "test": classification_report(yc[~in_train], pc_te, t_claim),
        "test_top10pct": capture_at_top(yc[~in_train].to_numpy(), pc_te, 0.10),
        "curves": curves(yc[~in_train], pc_te),
    }

    # ------------------------------------------------------------------ anomaly detection
    if_claim, a_claim = _isoforest(Xc, fit_mask=in_train)
    prov_train_mask = prov["provider"].isin(train_providers).to_numpy()
    if_prov, a_prov = _isoforest(Xp, fit_mask=prov_train_mask)
    if_pat, a_pat = _isoforest(pats[PATIENT_FEATURES])
    a_prov_te = a_prov[~prov_train_mask]
    metrics["anomaly"] = {
        "provider_isoforest_test_roc_auc": float(classification_report(yp[~prov_train_mask], a_prov_te, np.median(a_prov_te))["roc_auc"]),
        "provider_isoforest_top5pct": capture_at_top(yp[~prov_train_mask].to_numpy(), a_prov_te, 0.05),
        "claim_isoforest_test_roc_auc": float(classification_report(yc[~in_train], a_claim[~in_train], np.median(a_claim))["roc_auc"]),
        "note": "Isolation Forests never see labels; AUC shows how far pure outlier-ness lines up with fraud.",
    }

    # ------------------------------------------------------------------ out-of-fold scores for every entity
    skf_all = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE + 1)
    prov["fraud_prob"] = _oof(lambda: xgb.XGBClassifier(**PROVIDER_PARAMS), Xp, yp, skf_all)
    prov["anomaly_score"] = a_prov
    prov["anomaly_pct"] = _pct_rank(a_prov)
    prov["risk_tier"] = assign_provider_tier(prov["fraud_prob"], t_f1, t_crit, prov["anomaly_pct"])
    prov["in_holdout"] = ~prov_train_mask

    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE)
    claims["claim_fraud_prob"] = _oof(lambda: xgb.XGBClassifier(**CLAIM_PARAMS), Xc, yc, sgkf, groups=claims["provider"])
    claims["anomaly_score"] = a_claim
    claims["anomaly_pct"] = _pct_rank(a_claim)
    claims = claims.merge(prov[["provider", "fraud_prob", "risk_tier"]].rename(
        columns={"fraud_prob": "provider_fraud_prob", "risk_tier": "provider_risk_tier"}), on="provider")
    claims["risk_score"] = (0.5 * claims["provider_fraud_prob"] + 0.3 * claims["claim_fraud_prob"]
                            + 0.2 * claims["anomaly_pct"])
    claims["risk_tier"] = quantile_tier(claims["risk_score"])

    pat_claim = claims.groupby("bene_id").agg(
        max_claim_risk=("risk_score", "max"), mean_claim_risk=("risk_score", "mean"),
        n_high_risk_providers=("provider_risk_tier", lambda s: s.isin(["Critical", "High"]).sum()),
    ).reset_index()
    pats = pats.merge(pat_claim, on="bene_id", how="left")
    pats["anomaly_score"] = a_pat
    pats["anomaly_pct"] = _pct_rank(a_pat)
    pats["risk_score"] = 0.7 * pats["max_claim_risk"].fillna(0) + 0.3 * pats["anomaly_pct"]
    pats["risk_tier"] = quantile_tier(pats["risk_score"])

    tier_tbl = (prov.groupby("risk_tier").agg(providers=("provider", "size"), fraud_rate=("is_fraud", "mean"),
                                                reimbursed=("total_reimbursed", "sum"))
                .reindex(["Critical", "High", "Medium", "Low"]).reset_index())
    metrics["provider_tiers_oof"] = tier_tbl.to_dict(orient="records")
    claim_tier = (claims.groupby("risk_tier").agg(claims=("claim_id", "size"), from_fraud_provider=("provider_is_fraud", "mean"),
                                                   reimbursed=("amt_reimbursed", "sum"))
                  .reindex(["Critical", "High", "Medium", "Low"]).reset_index())
    metrics["claim_tiers_oof"] = claim_tier.to_dict(orient="records")

    # ------------------------------------------------------------------ final models + SHAP
    final_prov = xgb.XGBClassifier(**PROVIDER_PARAMS).fit(Xp, yp)
    final_claim = xgb.XGBClassifier(**CLAIM_PARAMS).fit(Xc, yc)
    sv, base = explain.shap_values(final_prov, Xp)
    shap_df = pd.DataFrame(sv, columns=PROVIDER_FEATURES)
    shap_df.insert(0, "provider", prov["provider"].to_numpy())
    shap_df.to_parquet(PROCESSED_DIR / "provider_shap.parquet", index=False)
    explain.save_summary_plot(sv, Xp, FIGURES_DIR / "shap_provider_fraud.png", "Provider fraud model - SHAP")
    sample = np.random.default_rng(RANDOM_STATE).choice(len(Xc), 20000, replace=False)
    svc, _ = explain.shap_values(final_claim, Xc.iloc[sample])
    explain.save_summary_plot(svc, Xc.iloc[sample], FIGURES_DIR / "shap_claim_fraud.png", "Claim fraud model - SHAP")
    importance = pd.Series(np.abs(sv).mean(0), index=PROVIDER_FEATURES).sort_values(ascending=False)
    metrics["provider_fraud"]["top_shap_features"] = importance.head(10).round(4).to_dict()

    joblib.dump({"provider": final_prov, "claim": final_claim, "if_claim": if_claim, "if_provider": if_prov,
                 "if_patient": if_pat, "thresholds": {"high": t_f1, "critical": t_crit},
                 "provider_shap_base": base}, MODELS_DIR / "fraud_models.joblib")
    final_prov.save_model(MODELS_DIR / "provider_fraud_xgb.json")
    final_claim.save_model(MODELS_DIR / "claim_fraud_xgb.json")

    prov.to_parquet(PROCESSED_DIR / "scored_providers.parquet", index=False)
    claims.to_parquet(PROCESSED_DIR / "scored_claims.parquet", index=False)
    pats.to_parquet(PROCESSED_DIR / "scored_patients.parquet", index=False)

    (REPORTS_DIR / "metrics_fraud.json").write_text(json.dumps(metrics, indent=2, default=float))
    if verbose:
        t = metrics["provider_fraud"]["test"]
        print(f"  provider XGB  test ROC-AUC {t['roc_auc']:.3f} | PR-AUC {t['pr_auc']:.3f} | "
              f"P {t['precision']:.2f} R {t['recall']:.2f} F1 {t['f1']:.2f}")
        r = metrics["provider_fraud"]["repeated_cv"]
        print(f"  provider XGB  5x2 CV ROC-AUC {r['xgb_roc_auc_mean']:.3f} ± {r['xgb_roc_auc_std']:.3f} | "
              f"PR-AUC {r['xgb_pr_auc_mean']:.3f} ± {r['xgb_pr_auc_std']:.3f} (LogReg {r['logreg_roc_auc_mean']:.3f} / {r['logreg_pr_auc_mean']:.3f})")
        b = metrics["provider_fraud"]["baseline_logreg_test"]
        print(f"  logreg base   test ROC-AUC {b['roc_auc']:.3f} | PR-AUC {b['pr_auc']:.3f} | F1 {b['f1']:.2f}")
        c = metrics["claim_fraud"]["test"]
        print(f"  claim XGB     test ROC-AUC {c['roc_auc']:.3f} | PR-AUC {c['pr_auc']:.3f}")
        print(f"  isoforest     provider AUC {metrics['anomaly']['provider_isoforest_test_roc_auc']:.3f} | "
              f"claim AUC {metrics['anomaly']['claim_isoforest_test_roc_auc']:.3f}")
        print(tier_tbl.to_string(index=False))
    return metrics


if __name__ == "__main__":
    run()
