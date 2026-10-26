-- =====================================================================
-- 01_claims.sql  -- clean beneficiaries + unified claim-level features
-- Inputs : raw_bene, raw_ip, raw_op  (all VARCHAR, loaded by features_fraud.py)
-- Outputs: bene, claims
-- =====================================================================

-- ---------- Beneficiaries (patients) ----------
CREATE OR REPLACE TABLE bene AS
SELECT
    BeneID                                                         AS bene_id,
    TRY_STRPTIME(DOB, '%m/%d/%Y')::DATE                           AS dob,
    TRY_STRPTIME(DOD, '%m/%d/%Y')::DATE                           AS dod,
    TRY_CAST(Gender AS INTEGER)                                    AS gender,
    TRY_CAST(Race AS INTEGER)                                      AS race,
    CASE WHEN RenalDiseaseIndicator = 'Y' THEN 1 ELSE 0 END        AS renal_disease,
    TRY_CAST(State AS INTEGER)                                     AS state,
    TRY_CAST(County AS INTEGER)                                    AS county,
    TRY_CAST(NoOfMonths_PartACov AS INTEGER)                       AS months_part_a,
    TRY_CAST(NoOfMonths_PartBCov AS INTEGER)                       AS months_part_b,
    -- Chronic condition flags are coded 1 = yes, 2 = no
    (ChronicCond_Alzheimer = '1')::INT           AS cc_alzheimer,
    (ChronicCond_Heartfailure = '1')::INT        AS cc_heartfailure,
    (ChronicCond_KidneyDisease = '1')::INT       AS cc_kidney,
    (ChronicCond_Cancer = '1')::INT              AS cc_cancer,
    (ChronicCond_ObstrPulmonary = '1')::INT      AS cc_copd,
    (ChronicCond_Depression = '1')::INT          AS cc_depression,
    (ChronicCond_Diabetes = '1')::INT            AS cc_diabetes,
    (ChronicCond_IschemicHeart = '1')::INT       AS cc_ischemicheart,
    (ChronicCond_Osteoporasis = '1')::INT        AS cc_osteoporosis,
    (ChronicCond_rheumatoidarthritis = '1')::INT AS cc_rheumatoid,
    (ChronicCond_stroke = '1')::INT              AS cc_stroke,
    TRY_CAST(IPAnnualReimbursementAmt AS DOUBLE) AS ip_annual_reimb,
    TRY_CAST(IPAnnualDeductibleAmt AS DOUBLE)    AS ip_annual_deduct,
    TRY_CAST(OPAnnualReimbursementAmt AS DOUBLE) AS op_annual_reimb,
    TRY_CAST(OPAnnualDeductibleAmt AS DOUBLE)    AS op_annual_deduct
FROM raw_bene;

ALTER TABLE bene ADD COLUMN cc_count INTEGER;
UPDATE bene SET cc_count = cc_alzheimer + cc_heartfailure + cc_kidney + cc_cancer + cc_copd
                         + cc_depression + cc_diabetes + cc_ischemicheart + cc_osteoporosis
                         + cc_rheumatoid + cc_stroke;

-- ---------- Unified claims (inpatient + outpatient) ----------
CREATE OR REPLACE TABLE claims AS
WITH unioned AS (
    SELECT 1 AS is_inpatient, * FROM raw_ip
    UNION ALL BY NAME
    SELECT 0 AS is_inpatient, * FROM raw_op
),
typed AS (
    SELECT
        ClaimID                                            AS claim_id,
        BeneID                                             AS bene_id,
        Provider                                           AS provider,
        is_inpatient,
        TRY_STRPTIME(ClaimStartDt, '%m/%d/%Y')::DATE       AS claim_start,
        TRY_STRPTIME(ClaimEndDt, '%m/%d/%Y')::DATE         AS claim_end,
        TRY_STRPTIME(AdmissionDt, '%m/%d/%Y')::DATE        AS admission_dt,
        TRY_STRPTIME(DischargeDt, '%m/%d/%Y')::DATE        AS discharge_dt,
        COALESCE(TRY_CAST(InscClaimAmtReimbursed AS DOUBLE), 0) AS amt_reimbursed,
        COALESCE(TRY_CAST(DeductibleAmtPaid AS DOUBLE), 0)      AS amt_deductible,
        AttendingPhysician                                 AS attending_phys,
        OperatingPhysician                                 AS operating_phys,
        OtherPhysician                                     AS other_phys,
        ClmAdmitDiagnosisCode                              AS admit_diag,
        DiagnosisGroupCode                                 AS drg_code,
        ClmDiagnosisCode_1                                 AS diag_1,
        (ClmDiagnosisCode_1 IS NOT NULL)::INT + (ClmDiagnosisCode_2 IS NOT NULL)::INT
          + (ClmDiagnosisCode_3 IS NOT NULL)::INT + (ClmDiagnosisCode_4 IS NOT NULL)::INT
          + (ClmDiagnosisCode_5 IS NOT NULL)::INT + (ClmDiagnosisCode_6 IS NOT NULL)::INT
          + (ClmDiagnosisCode_7 IS NOT NULL)::INT + (ClmDiagnosisCode_8 IS NOT NULL)::INT
          + (ClmDiagnosisCode_9 IS NOT NULL)::INT + (ClmDiagnosisCode_10 IS NOT NULL)::INT AS n_diag,
        (ClmProcedureCode_1 IS NOT NULL)::INT + (ClmProcedureCode_2 IS NOT NULL)::INT
          + (ClmProcedureCode_3 IS NOT NULL)::INT + (ClmProcedureCode_4 IS NOT NULL)::INT
          + (ClmProcedureCode_5 IS NOT NULL)::INT + (ClmProcedureCode_6 IS NOT NULL)::INT  AS n_proc
    FROM unioned
),
enriched AS (
    SELECT
        t.*,
        DATE_DIFF('day', claim_start, claim_end)                                  AS claim_duration_days,
        COALESCE(DATE_DIFF('day', admission_dt, discharge_dt), 0)                 AS los_days,
        (attending_phys IS NOT NULL)::INT + (operating_phys IS NOT NULL)::INT
          + (other_phys IS NOT NULL)::INT                                         AS n_physicians,
        (attending_phys IS NULL)::INT                                             AS missing_attending,
        COALESCE((attending_phys = operating_phys)::INT, 0)                       AS attending_is_operating,
        COALESCE((admit_diag = diag_1)::INT, 0)                                   AS admit_matches_primary,
        -- Peer group = same setting + same DRG (inpatient) or same primary diagnosis (outpatient)
        COALESCE(drg_code, diag_1, 'NA')                                          AS peer_group
    FROM typed t
),
peer AS (
    SELECT
        e.*,
        AVG(amt_reimbursed)    OVER (PARTITION BY is_inpatient, peer_group) AS peer_mean_amt,
        STDDEV_POP(amt_reimbursed) OVER (PARTITION BY is_inpatient, peer_group) AS peer_std_amt,
        COUNT(*)               OVER (PARTITION BY is_inpatient, peer_group) AS peer_n,
        COUNT(*) OVER (PARTITION BY bene_id, provider, claim_start)        AS same_day_claims
    FROM enriched e
)
SELECT
    p.claim_id, p.bene_id, p.provider, p.is_inpatient, p.claim_start, p.claim_end,
    p.attending_phys, p.operating_phys, p.drg_code, p.diag_1,
    p.amt_reimbursed, p.amt_deductible, p.n_diag, p.n_proc,
    p.claim_duration_days, p.los_days, p.n_physicians, p.missing_attending,
    p.attending_is_operating, p.admit_matches_primary,
    -- amount relative to clinical peers (shrink small peer groups toward 1.0)
    CASE WHEN p.peer_mean_amt > 0
         THEN (p.amt_reimbursed / p.peer_mean_amt) * (p.peer_n / (p.peer_n + 5.0)) + 1.0 * (5.0 / (p.peer_n + 5.0))
         ELSE 1.0 END                                                          AS amt_peer_ratio,
    CASE WHEN p.peer_std_amt > 0 THEN (p.amt_reimbursed - p.peer_mean_amt) / p.peer_std_amt ELSE 0 END AS amt_peer_z,
    (p.same_day_claims > 1)::INT                                               AS dup_same_day,
    -- patient context at claim time
    DATE_DIFF('year', b.dob, p.claim_start)                                    AS patient_age,
    b.cc_count                                                                 AS patient_cc_count,
    b.renal_disease                                                            AS patient_renal,
    b.ip_annual_reimb + b.op_annual_reimb                                      AS patient_annual_reimb,
    COALESCE((b.dod IS NOT NULL AND p.claim_start > b.dod)::INT, 0)            AS claim_after_death,
    b.state, b.county
FROM peer p
LEFT JOIN bene b USING (bene_id);
