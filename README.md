# HealthGuard AI

Healthcare Insurance Fraud & Readmission Risk Platform

This project looks at two problems in healthcare data:

1. Which providers (hospitals, clinics) are likely submitting fraudulent insurance claims.
2. Which diabetic patients are likely to be readmitted to hospital within 30 days of discharge.

It builds features from the raw data using SQL, trains machine-learning models, explains each prediction with SHAP, and shows everything in a Streamlit dashboard.

Tech: Python, SQL (DuckDB), XGBoost, Scikit-learn, SHAP, Streamlit, Plotly

![Dashboard overview](docs/screenshots/overview.png)

## Data

Two public Kaggle datasets:

| Dataset | Size | Used for |
|---|---|---|
| [Healthcare Provider Fraud Detection](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis) | 558,211 claims, 138,556 patients, 5,410 providers | Fraud detection |
| [Diabetes 130-US Hospitals (1999-2008)](https://www.kaggle.com/datasets/brandao/diabetes) | 101,766 hospital admissions | Readmission prediction |

In the fraud dataset, Kaggle marks each provider as potential fraud or not. About 9% of providers are marked as fraud. The labels are for providers, not individual claims.

For readmission, I removed admissions where the patient died or went to hospice, leaving 99,343 admissions. 11.4% of them were readmitted within 30 days.

## Architecture

```
Kaggle CSVs
    |
    v
SQL feature engineering (DuckDB)
    - claim level:    amount vs similar claims, length of stay, number of doctors, duplicates
    - patient level:  number of providers visited, chronic conditions, total spend
    - provider level: 35 features summarising each provider's claims
    |
    v
Models
    - XGBoost: fraud probability per provider and per claim
    - Isolation Forest: anomaly score (no labels used)
    - XGBoost: 30-day readmission probability
    |
    v
Risk tiers (Critical / High / Medium / Low) + SHAP explanation for each prediction
    |
    v
LLM summary (optional) + Streamlit dashboard
```

**Feature engineering.** Three SQL files in `sql/` turn the raw claims into features. For example, each claim's amount is compared with other claims that have the same diagnosis group, and each provider gets features such as claims per patient, average length of stay and how many different doctors bill through it.

**Fraud models.** The provider model is trained on the 35 provider features. The claim model is trained on claim features, with each claim taking its provider's label. An Isolation Forest scores how unusual each claim, provider and patient is without using any labels.

**Readmission model.** XGBoost trained on admission details: age, length of stay, prior visits, diagnoses grouped by ICD-9 category, medication changes and lab results.

**Explanations.** SHAP shows which features pushed each prediction up or down. The dashboard compares those features with the median provider. An optional LLM step (Groq) turns the SHAP output into a short written summary. Without an API key, a fixed template is used instead.

**Avoiding leakage.**
- A provider's claims are never split between training and test data.
- The scores shown in the dashboard come from models that did not see that provider during training (5-fold out-of-fold predictions).
- In the readmission data, all admissions of one patient stay in the same split.

## Results

All numbers are on data the models were not trained on.

**Provider fraud (XGBoost)**

| Metric | Result |
|---|---|
| ROC-AUC (5-fold cross-validation, repeated twice) | 0.941 ± 0.006 |
| PR-AUC | 0.69 (fraud rate is 9.4%) |
| Fraudulent providers found in the top 10% of scores | 71% |
| Logistic regression baseline, ROC-AUC | 0.936 |

**Risk tiers (provider level)**

| Tier | Providers | Share actually marked fraud |
|---|---|---|
| Critical | 131 | 89% |
| High | 536 | 47% |
| Medium | 394 | 15% |
| Low | 4,349 | 2% |

**Other models**

| Model | Result |
|---|---|
| Isolation Forest, provider level (no labels used) | ROC-AUC 0.82 |
| Claim-level fraud (XGBoost) | ROC-AUC 0.65 |
| 30-day readmission (XGBoost) | ROC-AUC 0.68 (logistic regression: 0.67) |
| Readmission rate by predicted risk band | High 28.5%, Medium 15.1%, Low 7.4% |

**What these results mean**
- XGBoost is only slightly better than logistic regression on provider fraud. Most of the signal comes from the engineered features.
- Fraud in this dataset is concentrated among large providers. Total amount reimbursed is the strongest single feature.
- The claim-level model is weak because the labels exist only for providers, so every claim from a fraudulent provider is labelled fraud even if that claim was normal.
- Readmission is hard to predict from hospital records alone. An ROC-AUC around 0.68 is in line with other published work on this dataset.

The full metrics are in `reports/metrics_fraud.json` and `reports/metrics_readmission.json`, and the SHAP plots are in `reports/figures/`.

## Dashboard

| Page | Shows |
|---|---|
| Overview | Totals, reimbursement by risk tier, highest-risk providers |
| Provider risk | Filter providers, see the SHAP drivers, compare with other providers, read the summary |
| Claim explorer | Filter claims, compare the model score with the anomaly score |
| Patient risk | Patients who visit many providers, and their claim history |
| Readmission risk | Score an admission and change the inputs to see how the risk moves |
| Model performance | Test metrics, ROC and precision-recall curves, SHAP plots |

![Provider page](docs/screenshots/provider.png)

## How to run

```bash
git clone https://github.com/nirdoshk0228/Health-guard-system.git
cd Health-guard-system
pip install -r requirements.txt

python scripts/download_data.py      # downloads the Kaggle data (needs a Kaggle API token)
python run_pipeline.py               # builds features and trains all models, about 3 minutes
streamlit run app/streamlit_app.py
```

To get LLM summaries, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add a Groq API key. Without a key, the template summaries are used.

Tests: `pytest -q`

## Project structure

```
app/streamlit_app.py          dashboard
scripts/download_data.py      downloads the Kaggle data
scripts/test_llm.py           checks the LLM summary works
sql/                          feature engineering in SQL
src/healthguard/
    features_fraud.py         runs the SQL and defines the feature lists
    features_readmission.py   readmission features
    train_fraud.py            fraud models, Isolation Forest, risk tiers
    train_readmission.py      readmission model
    explain.py                SHAP
    llm_explainer.py          LLM and template summaries
    scoring.py                prepares data for the dashboard
reports/                      saved metrics and SHAP plots
tests/                        unit tests
run_pipeline.py               runs everything in order
```

## Limitations

- The fraud labels are Kaggle's "potential fraud" flags, not confirmed cases.
- A high score means a provider should be reviewed, not that fraud has happened.
- The readmission model only uses data from the hospital stay. Information such as follow-up appointments would likely improve it.
