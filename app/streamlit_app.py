"""HealthGuard AI - Streamlit dashboard.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from healthguard import scoring  # noqa: E402
from healthguard.config import FIGURES_DIR, PROCESSED_DIR  # noqa: E402
from healthguard.explain import label  # noqa: E402
from healthguard.features_fraud import PATIENT_FEATURES  # noqa: E402
from healthguard.llm_explainer import active_backend, explain_case  # noqa: E402

st.set_page_config(page_title="HealthGuard AI", page_icon="🛡️", layout="wide")

TIER_COLORS = {"Critical": "#B42318", "High": "#E8590C", "Medium": "#D9A400", "Low": "#2F855A"}
TIERS = list(TIER_COLORS)
DISCHARGE = {"1": "Home", "2": "Short-term hospital", "3": "Skilled nursing facility", "4": "Intermediate care",
             "5": "Other inpatient care", "6": "Home with home health", "7": "Left AMA", "8": "Home IV provider",
             "18": "Unknown", "22": "Rehab facility", "23": "Long-term care hospital", "25": "Not mapped"}

st.markdown("""
<style>
  .block-container {padding-top: 1.6rem;}
  div[data-testid="stMetricValue"] {font-size: 1.6rem;}
  .tier {display:inline-block;padding:2px 10px;border-radius:999px;color:white;font-weight:600;font-size:0.85rem}
  .small {color:#6b7280;font-size:0.85rem}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading scored data…")
def get_tables():
    return scoring.load_tables()


@st.cache_resource(show_spinner="Loading models…")
def get_models():
    return scoring.load_models()


@st.cache_data
def get_metrics():
    return scoring.load_metrics()


if not (PROCESSED_DIR / "scored_providers.parquet").exists():
    st.error("No scored data found. Run `python scripts/download_data.py` then `python run_pipeline.py`.")
    st.stop()

T, M, MET = get_tables(), get_models(), get_metrics()
prov, pshap, claims, pats, readm = T["providers"], T["provider_shap"], T["claims"], T["patients"], T["readmission"]


def tier_badge(tier: str) -> str:
    return f'<span class="tier" style="background:{TIER_COLORS.get(tier, "#6b7280")}">{tier}</span>'


def money(x: float) -> str:
    return f"${x/1e6:,.1f}M" if abs(x) >= 1e6 else f"${x:,.0f}"


def shap_bar(case, title="What drives this score (SHAP)"):
    d = pd.DataFrame(case.drivers)
    def fmt(v):
        if isinstance(v, str) or v is None:
            return v
        return f"{v:,.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:.2f}"
    d["text"] = d.apply(lambda r: f"{r['label']} = {fmt(r['value'])}", axis=1)
    d = d.iloc[::-1]
    fig = go.Figure(go.Bar(x=d["shap"], y=d["text"], orientation="h",
                           marker_color=np.where(d["shap"] > 0, "#B42318", "#2F855A")))
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title="SHAP contribution (log-odds) — red raises risk, green lowers it")
    st.plotly_chart(fig, width="stretch")


def llm_block(case, key: str):
    backend = active_backend()
    st.markdown(f"**Analyst summary** <span class='small'>· backend: {backend}</span>", unsafe_allow_html=True)
    if st.button("Generate summary", key=key):
        with st.spinner("Writing summary…"):
            text, used = explain_case(case)
        st.session_state[key + "_txt"] = text
    if key + "_txt" in st.session_state:
        st.info(st.session_state[key + "_txt"])


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("🛡️ HealthGuard AI")
st.sidebar.caption("Healthcare insurance fraud & 30-day readmission risk")
page = st.sidebar.radio("View", ["Overview", "Provider risk", "Claim explorer", "Patient risk",
                                 "Readmission risk", "Model performance"])
st.sidebar.markdown("---")
st.sidebar.caption("Scores are out-of-fold model predictions: each provider was scored by a model that "
                   "never saw it during training. A high score means *review*, not *guilty*.")

# ============================================================================= OVERVIEW
if page == "Overview":
    st.title("Payment-integrity overview")
    flagged = prov[prov["risk_tier"].isin(["Critical", "High"])]
    c = st.columns(5)
    c[0].metric("Claims analysed", f"{len(claims):,}")
    c[1].metric("Providers", f"{len(prov):,}")
    c[2].metric("Providers flagged (Critical+High)", f"{len(flagged):,}", f"{len(flagged)/len(prov):.1%} of providers", delta_color="off")
    c[3].metric("Reimbursement under review", money(flagged["total_reimbursed"].sum()),
                f"{flagged['total_reimbursed'].sum()/prov['total_reimbursed'].sum():.0%} of total", delta_color="off")
    rcv = MET["fraud"]["provider_fraud"].get("repeated_cv", {})
    c[4].metric("Provider model ROC-AUC (CV)", f"{rcv.get('xgb_roc_auc_mean', MET['fraud']['provider_fraud']['test']['roc_auc']):.3f}")

    left, right = st.columns([1.1, 1])
    with left:
        tt = prov.groupby("risk_tier").agg(providers=("provider", "size"), fraud_rate=("is_fraud", "mean"),
                                           reimbursed=("total_reimbursed", "sum")).reindex(TIERS).reset_index()
        fig = px.bar(tt, x="risk_tier", y="reimbursed", color="risk_tier", color_discrete_map=TIER_COLORS,
                     text=tt["providers"].map(lambda n: f"{n:,} providers"),
                     labels={"risk_tier": "Risk tier", "reimbursed": "Total reimbursed ($)"},
                     title="Reimbursement by provider risk tier")
        fig.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.bar(tt, x="risk_tier", y="fraud_rate", color="risk_tier", color_discrete_map=TIER_COLORS,
                     text=tt["fraud_rate"].map(lambda r: f"{r:.0%}"),
                     labels={"risk_tier": "Risk tier", "fraud_rate": "Confirmed fraud rate"},
                     title="Tiers separate risk: labelled-fraud rate per tier")
        fig.update_layout(showlegend=False, height=360, yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Highest-risk providers")
    top = prov.sort_values("fraud_prob", ascending=False).head(15)
    st.dataframe(top[["provider", "risk_tier", "fraud_prob", "anomaly_pct", "n_claims", "n_patients",
                      "total_reimbursed", "ip_share"]].rename(columns={
                          "fraud_prob": "fraud prob", "anomaly_pct": "anomaly pctile", "ip_share": "inpatient share"}),
                 hide_index=True, width="stretch",
                 column_config={"fraud prob": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                "total_reimbursed": st.column_config.NumberColumn(format="$%d")})

# ============================================================================= PROVIDER
elif page == "Provider risk":
    st.title("Provider risk")
    f1, f2, f3 = st.columns([1, 1, 2])
    tiers = f1.multiselect("Tier", TIERS, default=["Critical", "High"])
    min_claims = f2.number_input("Min claims", 0, 10000, 10)
    view = prov[prov["risk_tier"].isin(tiers) & (prov["n_claims"] >= min_claims)].sort_values("fraud_prob", ascending=False)
    f3.caption(f"{len(view):,} providers match")
    st.dataframe(view[["provider", "risk_tier", "fraud_prob", "anomaly_pct", "n_claims", "total_reimbursed",
                       "claims_per_patient", "mean_patient_n_providers"]],
                 hide_index=True, width="stretch", height=260,
                 column_config={"fraud_prob": st.column_config.ProgressColumn("fraud prob", min_value=0, max_value=1, format="%.2f"),
                                "total_reimbursed": st.column_config.NumberColumn(format="$%d")})
    if view.empty:
        st.stop()
    pid = st.selectbox("Investigate provider", view["provider"].tolist())
    row = prov.loc[prov["provider"] == pid].iloc[0]
    case = scoring.provider_case(pid, prov, pshap)

    st.markdown(f"### {pid} &nbsp; {tier_badge(row['risk_tier'])}", unsafe_allow_html=True)
    c = st.columns(5)
    c[0].metric("Fraud probability", f"{row['fraud_prob']:.0%}")
    c[1].metric("Anomaly percentile", f"{row['anomaly_pct']:.0%}")
    c[2].metric("Claims", f"{int(row['n_claims']):,}")
    c[3].metric("Patients", f"{int(row['n_patients']):,}")
    c[4].metric("Reimbursed", money(row["total_reimbursed"]))

    a, b = st.columns([1.2, 1])
    with a:
        shap_bar(case)
    with b:
        peer = pd.DataFrame([{"driver": d["label"], "this provider": d["value"], "peer median": d["peer"]}
                             for d in case.drivers])
        st.markdown("**Against peers**")
        st.dataframe(peer, hide_index=True, width="stretch")
        llm_block(case, f"prov_{pid}")

    st.markdown("**This provider's riskiest claims**")
    pc = claims[claims["provider"] == pid].sort_values("risk_score", ascending=False).head(25)
    st.dataframe(pc[["claim_id", "bene_id", "claim_start", "is_inpatient", "amt_reimbursed", "amt_peer_ratio",
                     "claim_fraud_prob", "anomaly_pct", "risk_score", "risk_tier"]], hide_index=True, width="stretch")

# ============================================================================= CLAIMS
elif page == "Claim explorer":
    st.title("Claim explorer")
    f1, f2, f3, f4 = st.columns(4)
    tiers = f1.multiselect("Claim tier", TIERS, default=["Critical"])
    setting = f2.selectbox("Setting", ["All", "Inpatient", "Outpatient"])
    min_amt = f3.number_input("Min reimbursed ($)", 0, 200000, 0, step=500)
    anomalous = f4.checkbox("Top-5% anomalies only")
    view = claims[claims["risk_tier"].isin(tiers) & (claims["amt_reimbursed"] >= min_amt)]
    if setting != "All":
        view = view[view["is_inpatient"] == int(setting == "Inpatient")]
    if anomalous:
        view = view[view["anomaly_pct"] >= 0.95]
    view = view.sort_values("risk_score", ascending=False)
    st.caption(f"{len(view):,} claims · {money(view['amt_reimbursed'].sum())} reimbursed")

    fig = px.scatter(view.head(4000), x="claim_fraud_prob", y="anomaly_pct", color="risk_tier",
                     size=np.clip(view.head(4000)["amt_reimbursed"], 50, None), size_max=18,
                     color_discrete_map=TIER_COLORS, hover_data=["claim_id", "provider", "amt_reimbursed"],
                     labels={"claim_fraud_prob": "Claim model probability", "anomaly_pct": "Isolation-Forest anomaly percentile"},
                     title="Supervised vs unsupervised signal (bubble = $ reimbursed, top 4,000 shown)")
    fig.update_layout(height=380)
    st.plotly_chart(fig, width="stretch")

    st.dataframe(view.head(500)[["claim_id", "provider", "bene_id", "claim_start", "is_inpatient", "amt_reimbursed",
                                 "amt_peer_ratio", "claim_fraud_prob", "provider_fraud_prob", "anomaly_pct", "risk_score", "risk_tier"]],
                 hide_index=True, width="stretch", height=250)
    if view.empty:
        st.stop()
    cid = st.selectbox("Explain claim", view["claim_id"].head(500).tolist())
    crow = claims.loc[claims["claim_id"] == cid].iloc[0]
    case = scoring.claim_case(crow, claims, M["fraud"])
    st.markdown(f"### {cid} &nbsp; {tier_badge(crow['risk_tier'])}", unsafe_allow_html=True)
    c = st.columns(4)
    c[0].metric("Combined risk score", f"{crow['risk_score']:.2f}")
    c[1].metric("Claim model prob", f"{crow['claim_fraud_prob']:.0%}")
    c[2].metric("Provider fraud prob", f"{crow['provider_fraud_prob']:.0%}")
    c[3].metric("Anomaly percentile", f"{crow['anomaly_pct']:.0%}")
    st.caption("Combined score = 0.5 × provider probability + 0.3 × claim probability + 0.2 × anomaly percentile")
    a, b = st.columns([1.2, 1])
    with a:
        shap_bar(case, "Claim-model drivers (SHAP)")
    with b:
        llm_block(case, f"claim_{cid}")

# ============================================================================= PATIENTS
elif page == "Patient risk":
    st.title("Patient (beneficiary) risk")
    st.caption("Patient risk combines exposure to high-risk claims with how unusual the patient's own utilisation is "
               "(an Isolation Forest on patient-level features: providers visited, claims, chronic conditions, spend).")
    f1, f2 = st.columns(2)
    tiers = f1.multiselect("Tier", TIERS, default=["Critical"])
    min_prov = f2.slider("Min distinct providers visited", 1, 20, 1)
    view = pats[pats["risk_tier"].isin(tiers) & (pats["n_providers"] >= min_prov)].sort_values("risk_score", ascending=False)
    st.caption(f"{len(view):,} patients")
    a, b = st.columns(2)
    with a:
        fig = px.histogram(pats, x="n_providers", color="risk_tier", color_discrete_map=TIER_COLORS,
                           category_orders={"risk_tier": TIERS}, barmode="stack",
                           labels={"n_providers": "Distinct providers visited"}, title="Provider-hopping by patient tier")
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")
    with b:
        g = pats.groupby("risk_tier")[["n_claims", "n_providers", "cc_count", "total_reimbursed"]].mean().reindex(TIERS)
        g.columns = ["claims", "providers", "chronic conds", "reimbursed $"]
        st.markdown("**Average profile by tier**")
        st.dataframe(g.round(1), width="stretch")
    st.dataframe(view.head(500)[["bene_id", "risk_tier", "risk_score", "max_claim_risk", "anomaly_pct",
                                 "n_claims", "n_providers", "n_high_risk_providers", "cc_count", "total_reimbursed"]],
                 hide_index=True, width="stretch", height=260)
    if not view.empty:
        bid = st.selectbox("Patient claim history", view["bene_id"].head(500).tolist())
        prow = pats.loc[pats["bene_id"] == bid].iloc[0]
        st.markdown(f"### {bid} &nbsp; {tier_badge(prow['risk_tier'])}", unsafe_allow_html=True)
        z = ((pats[PATIENT_FEATURES] - pats[PATIENT_FEATURES].mean()) / pats[PATIENT_FEATURES].std()).loc[prow.name]
        unusual = z.abs().sort_values(ascending=False).head(4)
        st.write("Most unusual attributes: " + "; ".join(
            f"**{label(f)}** = {prow[f]:,.0f} ({'+' if z[f] > 0 else ''}{z[f]:.1f} SD)" for f in unusual.index))
        st.dataframe(claims[claims["bene_id"] == bid].sort_values("claim_start")[
            ["claim_id", "claim_start", "provider", "provider_risk_tier", "is_inpatient", "amt_reimbursed", "risk_score", "risk_tier"]],
            hide_index=True, width="stretch")

# ============================================================================= READMISSION
elif page == "Readmission risk":
    st.title("30-day readmission risk")
    rm = MET["readmission"]
    c = st.columns(4)
    c[0].metric("Holdout encounters", f"{len(readm):,}")
    c[1].metric("Observed 30-day readmission", f"{readm['readmit_30d'].mean():.1%}")
    c[2].metric("ROC-AUC (holdout)", f"{rm['test']['roc_auc']:.3f}")
    c[3].metric("Top-10% lift", f"{rm['test_top10pct']['lift']:.1f}×")

    a, b = st.columns(2)
    with a:
        bt = pd.DataFrame(rm["risk_bands_test"])
        fig = px.bar(bt, x="band", y="readmit_rate", color="band",
                     color_discrete_map={"High": "#B42318", "Medium": "#D9A400", "Low": "#2F855A"},
                     text=bt["readmit_rate"].map(lambda r: f"{r:.0%}"), title="Observed readmission rate by predicted band",
                     labels={"band": "Predicted risk band", "readmit_rate": "Readmitted ≤30 days"})
        fig.update_layout(showlegend=False, height=320, yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    with b:
        cal = rm["calibration"]
        fig = go.Figure([go.Scatter(x=[0, 0.5], y=[0, 0.5], mode="lines", line=dict(dash="dot", color="#9ca3af"), name="perfect"),
                         go.Scatter(x=cal["mean_pred"], y=cal["frac_pos"], mode="lines+markers", name="model", line=dict(color="#1f4e79"))])
        fig.update_layout(title="Calibration (holdout, deciles)", xaxis_title="Predicted probability",
                          yaxis_title="Observed rate", height=320)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Score an encounter")
    st.caption("Pick a holdout encounter to pre-fill, then change any field to see how the risk moves.")
    bundle = M["readmission"]
    levels = bundle["category_levels"]
    pick = st.selectbox("Start from encounter", readm.sort_values("readmit_prob", ascending=False)["encounter_id"].head(300).tolist())
    base = readm.loc[readm["encounter_id"] == pick].iloc[0].to_dict()
    for k, v in list(base.items()):
        if hasattr(v, "item"):
            base[k] = v.item()

    with st.form("readmit"):
        r1 = st.columns(4)
        base["age_years"] = r1[0].slider("Age", 5, 95, int(base["age_years"]), step=10)
        base["time_in_hospital"] = r1[1].slider("Length of stay (days)", 1, 14, int(base["time_in_hospital"]))
        base["number_inpatient"] = r1[2].number_input("Inpatient admissions, prior year", 0, 25, int(base["number_inpatient"]))
        base["number_emergency"] = r1[3].number_input("ER visits, prior year", 0, 80, int(base["number_emergency"]))
        r2 = st.columns(4)
        base["num_medications"] = r2[0].slider("Medications", 1, 80, int(base["num_medications"]))
        base["number_diagnoses"] = r2[1].slider("Diagnoses", 1, 16, int(base["number_diagnoses"]))
        dd_opts = levels["discharge_disposition_id"]
        base["discharge_disposition_id"] = r2[2].selectbox(
            "Discharge to", dd_opts, index=dd_opts.index(str(base["discharge_disposition_id"])),
            format_func=lambda x: f"{x} · {DISCHARGE.get(x, 'other')}")
        ins = levels["insulin_status"]
        base["insulin_status"] = r2[3].selectbox("Insulin", ins, index=ins.index(str(base["insulin_status"])))
        r3 = st.columns(3)
        dg = levels["diag_1_grp"]
        base["diag_1_grp"] = r3[0].selectbox("Primary diagnosis group", dg, index=dg.index(str(base["diag_1_grp"])))
        base["a1c_high"] = int(r3[1].checkbox("HbA1c > 7%", bool(base["a1c_high"])))
        base["med_change"] = int(r3[2].checkbox("Diabetes medication changed", bool(base["med_change"])))
        submitted = st.form_submit_button("Score")

    base["total_prior_visits"] = base["number_outpatient"] + base["number_emergency"] + base["number_inpatient"]
    base["meds_per_day"] = base["num_medications"] / base["time_in_hospital"]
    base["labs_per_day"] = base["num_lab_procedures"] / base["time_in_hospital"]
    base["a1c_tested"] = max(base["a1c_tested"], base["a1c_high"])
    base["diag_1_is_diabetes"] = int(base["diag_1_grp"] == "Diabetes")
    X = scoring.readmission_frame(base, levels)
    case = scoring.readmission_case(str(pick), X, bundle, readm)
    st.markdown(f"### Predicted 30-day readmission risk: {case.score:.0%} &nbsp; {tier_badge(case.tier)}"
                f"<span class='small'> &nbsp; cohort average {readm['readmit_30d'].mean():.0%}"
                f" · actual outcome for the original encounter: {'readmitted' if base['readmit_30d'] else 'not readmitted'}</span>",
                unsafe_allow_html=True)
    a, b = st.columns([1.2, 1])
    with a:
        shap_bar(case)
    with b:
        llm_block(case, f"readmit_{pick}_{hash(tuple(sorted((k, str(v)) for k, v in base.items())))}")

# ============================================================================= PERFORMANCE
else:
    st.title("Model performance")
    st.caption("All numbers are on held-out data the models never trained on. Provider and claim models share one "
               "provider-level split; the readmission split is grouped by patient.")
    fm, rm = MET["fraud"], MET["readmission"]
    rows = []
    for name, m, base in [("Provider fraud · XGBoost", fm["provider_fraud"]["test"], fm["provider_fraud"]["baseline_logreg_test"]),
                          ("Claim fraud · XGBoost", fm["claim_fraud"]["test"], None),
                          ("30-day readmission · XGBoost", rm["test"], rm["baseline_logreg_test"])]:
        rows.append({"model": name, "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"], "precision": m["precision"],
                     "recall": m["recall"], "F1": m["f1"], "base rate": m["base_rate"],
                     "LogReg ROC-AUC": base["roc_auc"] if base else np.nan})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    rcv = fm["provider_fraud"].get("repeated_cv")
    if rcv:
        st.caption(f"Provider model, 5-fold × 2 repeated CV on all 5,410 providers: ROC-AUC "
                   f"{rcv['xgb_roc_auc_mean']:.3f} ± {rcv['xgb_roc_auc_std']:.3f}, PR-AUC {rcv['xgb_pr_auc_mean']:.3f} ± "
                   f"{rcv['xgb_pr_auc_std']:.3f} (logistic-regression baseline {rcv['logreg_roc_auc_mean']:.3f} / "
                   f"{rcv['logreg_pr_auc_mean']:.3f}). The single holdout has ~100 fraudulent providers, so CV is the steadier estimate.")
    a = fm["anomaly"]
    st.markdown(f"**Isolation Forest (unsupervised).** Provider-level anomaly score reaches ROC-AUC "
                f"**{a['provider_isoforest_test_roc_auc']:.3f}** against fraud labels it never saw; the top-5% most anomalous "
                f"providers are **{a['provider_isoforest_top5pct']['lift']:.1f}×** more likely to be fraudulent than average. "
                f"At claim level the anomaly score is close to random (AUC {a['claim_isoforest_test_roc_auc']:.2f}): "
                "fraud here shows up in providers' billing *patterns*, not in individually odd claims.")

    c1, c2 = st.columns(2)
    for col, title, cv in [(c1, "ROC", "roc"), (c2, "Precision-recall", "pr")]:
        fig = go.Figure()
        for name, curv in [("Provider fraud", fm["provider_fraud"]["curves"]), ("Claim fraud", fm["claim_fraud"]["curves"]),
                           ("Readmission", rm["curves"])]:
            x, y = (curv["roc"]["fpr"], curv["roc"]["tpr"]) if cv == "roc" else (curv["pr"]["recall"], curv["pr"]["precision"])
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
        fig.update_layout(title=title, height=360, xaxis_title="FPR" if cv == "roc" else "Recall",
                          yaxis_title="TPR" if cv == "roc" else "Precision")
        col.plotly_chart(fig, width="stretch")

    st.subheader("Global SHAP importance")
    imgs = [("shap_provider_fraud.png", "Provider fraud"), ("shap_claim_fraud.png", "Claim fraud"),
            ("shap_readmission.png", "Readmission")]
    cols = st.columns(3)
    for col, (f, cap) in zip(cols, imgs):
        if (FIGURES_DIR / f).exists():
            col.image(str(FIGURES_DIR / f), caption=cap, width="stretch")
