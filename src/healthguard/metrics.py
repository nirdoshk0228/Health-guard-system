"""Evaluation helpers shared by all models."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.nanargmax(f1[:-1])])


def threshold_for_precision(y, p, target: float) -> float:
    """Lowest threshold whose precision reaches `target` (falls back to the best-F1 one)."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(prec[:-1] >= target)[0]
    return float(thr[ok[0]]) if len(ok) else best_f1_threshold(y, p)


def capture_at_top(y, p, frac: float) -> dict:
    """Share of all positives found when reviewing only the top `frac` highest-scored cases."""
    y = np.asarray(y)
    n = max(1, int(round(len(p) * frac)))
    top = np.argsort(-np.asarray(p))[:n]
    return {
        "review_fraction": frac,
        "recall": float(y[top].sum() / max(1, y.sum())),
        "precision": float(y[top].mean()),
        "lift": float(y[top].mean() / max(1e-12, y.mean())),
    }


def classification_report(y, p, threshold: float) -> dict:
    yhat = (np.asarray(p) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat)),
        "f1": float(f1_score(y, yhat)),
        "base_rate": float(np.mean(y)),
        "n": int(len(y)),
    }


def curves(y, p, points: int = 200) -> dict:
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    take = lambda a: a[np.linspace(0, len(a) - 1, min(points, len(a))).astype(int)].round(4).tolist()  # noqa: E731
    return {"roc": {"fpr": take(fpr), "tpr": take(tpr)}, "pr": {"recall": take(rec), "precision": take(prec)}}
