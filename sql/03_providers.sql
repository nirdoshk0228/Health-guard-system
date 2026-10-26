-- =====================================================================
-- 03_providers.sql  -- provider-level feature table + fraud label
-- Inputs : claims, raw_labels
-- Output : providers
-- =====================================================================

CREATE OR REPLACE TABLE providers AS
WITH agg AS (
    SELECT
        provider,
        COUNT(*)                                       AS n_claims,
        SUM(is_inpatient)                              AS n_ip_claims,
        AVG(is_inpatient)                              AS ip_share,
        COUNT(DISTINCT bene_id)                        AS n_patients,
        COUNT(*) / COUNT(DISTINCT bene_id)             AS claims_per_patient,
        SUM(amt_reimbursed)                            AS total_reimbursed,
        AVG(amt_reimbursed)                            AS mean_reimbursed,
        MAX(amt_reimbursed)                            AS max_reimbursed,
        COALESCE(STDDEV_POP(amt_reimbursed), 0)        AS std_reimbursed,
        SUM(CASE WHEN is_inpatient = 1 THEN amt_reimbursed ELSE 0 END)
            / NULLIF(SUM(is_inpatient), 0)             AS mean_ip_reimbursed,
        AVG(amt_deductible)                            AS mean_deductible,
        AVG(CASE WHEN is_inpatient = 1 THEN los_days END) AS mean_los,
        AVG(claim_duration_days)                       AS mean_claim_duration,
        COUNT(DISTINCT attending_phys)                 AS n_attending_phys,
        COUNT(DISTINCT operating_phys)                 AS n_operating_phys,
        COUNT(*) / NULLIF(COUNT(DISTINCT attending_phys), 0) AS claims_per_attending,
        AVG(n_diag)                                    AS mean_n_diag,
        AVG(n_proc)                                    AS mean_n_proc,
        AVG(n_physicians)                              AS mean_n_physicians,
        AVG(attending_is_operating)                    AS share_attending_is_operating,
        AVG(missing_attending)                         AS share_missing_attending,
        AVG(amt_peer_ratio)                            AS mean_peer_ratio,
        MAX(amt_peer_ratio)                            AS max_peer_ratio,
        AVG((amt_peer_ratio > 2)::INT)                 AS share_peer_ratio_gt2,
        AVG(dup_same_day)                              AS share_dup_same_day,
        SUM(claim_after_death)                         AS claims_after_death,
        AVG(patient_age)                               AS mean_patient_age,
        AVG(patient_cc_count)                          AS mean_patient_cc,
        AVG(patient_renal)                             AS share_renal_patients,
        AVG(patient_annual_reimb)                      AS mean_patient_annual_reimb,
        AVG(patient_n_providers)                       AS mean_patient_n_providers,
        AVG(phys_n_providers)                          AS mean_phys_n_providers,
        COUNT(DISTINCT state)                          AS n_states,
        COUNT(DISTINCT county)                         AS n_counties,
        COUNT(DISTINCT drg_code)                       AS n_distinct_drg,
        COUNT(DISTINCT diag_1)                         AS n_distinct_primary_diag
    FROM claims
    GROUP BY provider
)
SELECT
    a.*,
    CASE WHEN l.PotentialFraud = 'Yes' THEN 1 ELSE 0 END AS is_fraud
FROM agg a
JOIN raw_labels l ON l.Provider = a.provider;
