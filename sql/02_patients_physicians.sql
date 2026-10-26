-- =====================================================================
-- 02_patients_physicians.sql  -- patient- and physician-level behaviour
-- Inputs : bene, claims
-- Outputs: patients, physicians, claims (augmented with patient/physician context)
-- =====================================================================

CREATE OR REPLACE TABLE patients AS
SELECT
    b.bene_id,
    DATE_DIFF('year', b.dob, DATE '2009-12-31')               AS age,
    b.gender, b.race, b.state, b.renal_disease, b.cc_count,
    (b.dod IS NOT NULL)::INT                                  AS is_deceased,
    b.ip_annual_reimb, b.op_annual_reimb,
    COUNT(c.claim_id)                                         AS n_claims,
    COALESCE(SUM(c.is_inpatient), 0)                          AS n_ip_claims,
    COUNT(DISTINCT c.provider)                                AS n_providers,
    COUNT(DISTINCT c.attending_phys)                          AS n_attending_phys,
    COALESCE(SUM(c.amt_reimbursed), 0)                        AS total_reimbursed,
    COALESCE(AVG(c.amt_peer_ratio), 1)                        AS mean_peer_ratio,
    COALESCE(SUM(c.claim_after_death), 0)                     AS claims_after_death,
    COALESCE(SUM(c.dup_same_day), 0)                          AS dup_same_day_claims
FROM bene b
LEFT JOIN claims c USING (bene_id)
GROUP BY ALL;

-- Physicians who bill through many different providers are a classic ring signal
CREATE OR REPLACE TABLE physicians AS
SELECT
    attending_phys,
    COUNT(*)                     AS phys_n_claims,
    COUNT(DISTINCT provider)     AS phys_n_providers,
    COUNT(DISTINCT bene_id)      AS phys_n_patients
FROM claims
WHERE attending_phys IS NOT NULL
GROUP BY attending_phys;

CREATE OR REPLACE TABLE claims AS
SELECT
    c.*,
    p.n_claims          AS patient_n_claims,
    p.n_providers       AS patient_n_providers,
    COALESCE(ph.phys_n_claims, 0)    AS phys_n_claims,
    COALESCE(ph.phys_n_providers, 0) AS phys_n_providers
FROM claims c
LEFT JOIN patients p    USING (bene_id)
LEFT JOIN physicians ph USING (attending_phys);
