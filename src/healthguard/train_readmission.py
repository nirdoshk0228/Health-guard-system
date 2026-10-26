"""30-day hospital readmission model (XGBoost) with a logistic-regression baseline.

The split is grouped by patient (patient_nbr) so the same person never appears in both train and
test; patients have up to 40 encounters in this dataset, so a random row split would leak.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import explain
from .config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR, TEST_SIZE
from .features_readmission import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET
from .metrics import best_f1_threshold, capture_at_top, classification_report, curves

PARAMS = dict(
    n_estimators=2000, learning_rate=0.02, max_depth=5, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, max_cat_to_onehot=1,
    enable_categorical=True, tree_method="hist", eval_metric="auc",
    early_stopping_rounds=100, random_state=RANDOM_STATE, n_jobs=-1,
)


def risk_band(p, cuts) -> np.ndarray:
    p = np.asarray(p)
    return np.select([p >= cuts["high"], p >= cuts["medium"]], ["High", "Medium"], default="Low")


def run(verbose: bool = True) -> dict:
    df = pd.read_parquet(PROCESSED_DIR / "readmission_features.parquet")
    X, y, g = df[FEATURES], df[TARGET], df["patient_nbr"]

    outer = GroupShuffleSplit(1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr, te = next(outer.split(X, y, g))
    inner = GroupShuffleSplit(1, test_size=0.15, random_state=RANDOM_STATE)
    fit_i, val_i = next(inner.split(X.iloc[tr], y.iloc[tr], g.iloc[tr]))
    fit_i, val_i = tr[fit_i], tr[val_i]

    model = xgb.XGBClassifier(**PARAMS)
    model.fit(X.iloc[fit_i], y.iloc[fit_i], eval_set=[(X.iloc[val_i], y.iloc[val_i])], verbose=False)
    p_val = model.predict_proba(X.iloc[val_i])[:, 1]
    p_te = model.predict_proba(X.iloc[te])[:, 1]
    thr = best_f1_threshold(y.iloc[val_i], p_val)

    # Baseline: one-hot logistic regression
    pre = ColumnTransformer([("num", StandardScaler(), NUMERIC_FEATURES),
                             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CATEGORICAL_FEATURES)])
    lr = make_pipeline(pre, LogisticRegression(max_iter=3000, C=0.5))
    Xs = X.copy()
    for c in CATEGORICAL_FEATURES:
        Xs[c] = Xs[c].astype(str)
    lr.fit(Xs.iloc[tr], y.iloc[tr])
    p_te_lr = lr.predict_proba(Xs.iloc[te])[:, 1]

    # Risk bands: High = top decile of validation scores, Medium = next 20%
    cuts = {"high": float(np.quantile(p_val, 0.90)), "medium": float(np.quantile(p_val, 0.70))}
    band_te = risk_band(p_te, cuts)
    band_tbl = (pd.DataFrame({"band": band_te, "y": y.iloc[te].to_numpy()})
                .groupby("band").agg(encounters=("y", "size"), readmit_rate=("y", "mean"))
                .reindex(["High", "Medium", "Low"]).reset_index())

    frac_pos, mean_pred = calibration_curve(y.iloc[te], p_te, n_bins=10, strategy="quantile")
    metrics = {
        "model": "XGBoost (native categorical, early stopping)",
        "best_iteration": int(model.best_iteration),
        "test": classification_report(y.iloc[te], p_te, thr),
        "test_top10pct": capture_at_top(y.iloc[te].to_numpy(), p_te, 0.10),
        "test_top20pct": capture_at_top(y.iloc[te].to_numpy(), p_te, 0.20),
        "baseline_logreg_test": classification_report(y.iloc[te], p_te_lr, best_f1_threshold(y.iloc[tr], lr.predict_proba(Xs.iloc[tr])[:, 1])),
        "risk_bands_test": band_tbl.to_dict(orient="records"),
        "band_cuts": cuts,
        "calibration": {"mean_pred": mean_pred.round(4).tolist(), "frac_pos": frac_pos.round(4).tolist()},
        "curves": curves(y.iloc[te], p_te),
        "n_train": int(len(tr)), "n_test": int(len(te)),
    }

    sample = np.random.default_rng(RANDOM_STATE).choice(te, min(8000, len(te)), replace=False)
    sv, base = explain.shap_values(model, X.iloc[sample])
    explain.save_summary_plot(sv, X.iloc[sample], FIGURES_DIR / "shap_readmission.png", "30-day readmission model - SHAP")
    imp = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False)
    metrics["top_shap_features"] = imp.head(10).round(4).to_dict()

    scored = df.iloc[te].copy()
    scored["readmit_prob"] = p_te
    scored["risk_band"] = band_te
    scored.to_parquet(PROCESSED_DIR / "scored_readmission_test.parquet", index=False)

    levels = {c: list(X[c].cat.categories) for c in CATEGORICAL_FEATURES}
    joblib.dump({"model": model, "threshold": thr, "band_cuts": cuts, "category_levels": levels,
                 "feature_order": FEATURES}, MODELS_DIR / "readmission_model.joblib")
    model.save_model(MODELS_DIR / "readmission_xgb.json")
    (REPORTS_DIR / "metrics_readmission.json").write_text(json.dumps(metrics, indent=2, default=float))

    if verbose:
        t, b = metrics["test"], metrics["baseline_logreg_test"]
        print(f"  readmit XGB   test ROC-AUC {t['roc_auc']:.3f} | PR-AUC {t['pr_auc']:.3f} | "
              f"top-10% lift {metrics['test_top10pct']['lift']:.2f}x")
        print(f"  logreg base   test ROC-AUC {b['roc_auc']:.3f} | PR-AUC {b['pr_auc']:.3f}")
        print(band_tbl.to_string(index=False))
    return metrics


if __name__ == "__main__":
    run()
