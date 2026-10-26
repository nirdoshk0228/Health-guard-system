"""Fast unit tests (no data download needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from healthguard.features_readmission import icd9_group  # noqa: E402
from healthguard.llm_explainer import RiskCase, explain_case, template_summary  # noqa: E402
from healthguard.metrics import best_f1_threshold, capture_at_top, classification_report  # noqa: E402
from healthguard.train_fraud import assign_provider_tier, quantile_tier  # noqa: E402


def test_icd9_grouping():
    assert icd9_group("428") == "Circulatory"
    assert icd9_group("250.83") == "Diabetes"
    assert icd9_group("486") == "Respiratory"
    assert icd9_group("V57") == "Other"
    assert icd9_group(np.nan) == "Missing"
    assert icd9_group("820") == "Injury"


def test_metrics_perfect_model():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    r = classification_report(y, p, best_f1_threshold(y, p))
    assert r["roc_auc"] == 1.0 and r["f1"] == 1.0
    top = capture_at_top(y, p, 0.4)
    assert top["recall"] == 1.0 and top["precision"] == 1.0


def test_tiers_are_monotonic():
    p = np.array([0.95, 0.6, 0.3, 0.05])
    tiers = assign_provider_tier(p, t_high=0.5, t_critical=0.9, anomaly_pct=np.zeros(4))
    assert list(tiers) == ["Critical", "High", "Medium", "Low"]
    q = quantile_tier(np.arange(100))
    assert (q == "Critical").sum() == 2 and (q == "High").sum() == 8


def test_template_explainer_offline(monkeypatch):
    for k in ("HEALTHGUARD_LLM", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OLLAMA_HOST"):
        monkeypatch.delenv(k, raising=False)
    case = RiskCase("provider", "PRV1", 0.91, "Critical", [
        {"feature": "claims_per_patient", "label": "claims per patient", "value": 9.2, "shap": 0.8,
         "direction": "increases", "peer": 2.1},
        {"feature": "ip_share", "label": "share of claims that are inpatient", "value": 0.02, "shap": -0.2,
         "direction": "decreases", "peer": 0.05},
    ])
    text, backend = explain_case(case)
    assert backend == "template"
    assert "PRV1" in text and "Critical" in text and "Next step" in text
    assert text == template_summary(case)
