# Paper 2: Automated Model Selection for the Radial Acceleration Relation with Rigorous Galaxy-Level Generalization

## A Methodological Framework for Physics Discovery Pipelines

**Date:** January 15, 2026  
**Dataset:** SPARC (Spitzer Photometry and Accurate Rotation Curves) - 171 Galaxies  
**Methodology:** Model Selection with Galaxy-Level Holdout and Kill Tests

---

## Abstract

We present a rigorous methodology for automated model selection in galactic dynamics, demonstrating that strict galaxy-level holdout and appropriate kill tests can reliably distinguish between candidate acceleration laws. Applying this framework to SPARC rotation curve data, we show that the pipeline correctly selects the Radial Acceleration Relation (RAR) from a menu of candidates and confirms it generalizes to 35 held-out galaxies never seen during selection. This work does **not** claim to discover new physics; rather, it establishes a **calibrated benchmark** that future discovery claims must meet.

**Key Contributions:**
1. Galaxy-level train/validation/test splits preventing data leakage
2. Galaxy-weighted log-space metrics appropriate for acceleration data
3. Three kill tests confirming selection is not spurious
4. Demonstration that the pipeline recovers known physics (RAR)

**What This Paper Does NOT Claim:**
- Discovery of new physical laws
- Evidence for or against dark matter vs. modified gravity
- Novel functional forms beyond existing literature

---

## 1. Introduction

### 1.1 The Problem: Overfitting in "AI Discovers Physics"

Many claims of automated physics discovery fail due to:
- **Data leakage**: Points from the same system appear in train and test
- **Objective mismatch**: Fitting one metric, reporting another
- **Missing controls**: No verification that discovery is not spurious
- **Overclaiming**: Framing model selection as "discovery"

We address each of these concerns explicitly.

### 1.2 Scope and Claims

**What we demonstrate:**
- A pipeline that selects between candidate functional forms
- Rigorous generalization testing at the galaxy level
- Recovery of the known RAR relation as validation

**What we do not demonstrate:**
- Symbolic discovery from scratch (candidates are hand-specified)
- New physics beyond existing models
- Preference between dark matter and modified gravity explanations

### 1.3 The Radial Acceleration Relation

The RAR (McGaugh et al. 2016) relates observed and baryonic accelerations:

$$g_{obs} = \frac{g_{bar}}{1 - e^{-\sqrt{g_{bar}/g^\dagger}}}$$

With characteristic acceleration g† ≈ 1.2 × 10⁻¹⁰ m/s². This empirical relation is well-established in the literature and serves as our **ground truth** for pipeline validation.

---

## 2. Methodology

### 2.1 Data Representation

Following standard practice in galactic dynamics:

| Variable | Definition | Source |
|----------|------------|--------|
| g_obs | V_obs² / R | Observed centripetal acceleration |
| g_bar | V_bar² / R | Baryonic (Newtonian) acceleration |
| V_bar | √(V_gas² + V_disk² + V_bul²) | SPARC decomposition |

Accelerations span ~4 orders of magnitude (10⁻¹³ to 10⁻⁹ m/s²), requiring log-space metrics.

### 2.2 Galaxy-Level Split

**Critical design choice:** We split by galaxy, not by data point.

| Set | N Galaxies | N Points | Purpose |
|-----|------------|----------|---------|
| Train | 102 (60%) | ~2,100 | Fit candidate parameters |
| Validation | 34 (20%) | ~700 | Select best candidate |
| Test | 35 (20%) | ~700 | Final evaluation (LOCKED) |

Splitting by galaxy prevents leakage: a galaxy's inner and outer points share correlated structure.

### 2.3 Candidate Functional Forms

We evaluate a menu of physically-motivated forms:

| Form | Formula | Parameters | Complexity |
|------|---------|------------|------------|
| Linear | g_obs = α × g_bar + β | α, β | 2 |
| Power | g_obs = α × g_bar^β | α, β | 3 |
| RAR | g_bar / (1 - exp(-√(g_bar/g†))) | g† | 2 |
| MOND-simple | g_bar × √(1 + a₀/g_bar) | a₀ | 2 |

**Note:** All candidates are hand-specified. This is model selection, not symbolic discovery.

### 2.4 Fitting Procedure

**Key improvement over v1:** Fit and evaluation use the same objective.

For each candidate form:
1. Fit parameters on **train galaxies only**
2. Use galaxy-weighted log-MAE as objective:

$$\mathcal{L} = \frac{1}{N_{gal}} \sum_{g=1}^{N_{gal}} \left[ \frac{1}{n_g} \sum_{i \in g} \left| \log_{10}(g_{pred,i}) - \log_{10}(g_{obs,i}) \right| \right]$$

This ensures:
- Large galaxies don't dominate (galaxy weighting)
- Metric appropriate for multi-decade acceleration range (log-space)
- Fit objective = evaluation objective (no mismatch)

### 2.5 Kill Tests (Controls)

Three tests verify selection is not spurious:

**Control 1: Permutation Test (Re-run Discovery)**
- Shuffle g_obs within each galaxy
- **Re-fit all candidates** on shuffled data
- Re-run selection
- Pass criterion: No candidate generalizes well on shuffled data

**Control 2: Feature Ablation**
- Remove g_bar, attempt to predict g_obs from R alone
- Pass criterion: Performance craters without g_bar

**Control 3: Random Feature Importance**
- Add 5 random noise features
- Verify g_bar dominates in linear probe
- Pass criterion: g_bar coefficient >> random coefficients

---

## 3. Results

### 3.1 Model Selection (Train → Validation)

All candidates fitted on 102 train galaxies, evaluated on 34 validation galaxies:

| Form | Train MAE | Val MAE | Δ | Selected |
|------|-----------|---------|---|----------|
| **RAR** | **0.256** | **0.206** | -0.050 | **✓** |
| MOND-simple | 0.279 | 0.233 | -0.046 | |
| Power | 0.287 | 0.241 | -0.046 | |
| Linear | 0.289 | 0.238 | -0.051 | |

**Result:** RAR is selected with best validation performance.

**Fitted parameter:** g† = 1.20 × 10⁻¹⁰ m/s² (matches McGaugh et al. 2016)

### 3.2 Kill Test Results

| Control | Result | Criterion | Status |
|---------|--------|-----------|--------|
| Permutation (re-fit) | No candidate generalizes | Val MAE > 0.33 for all | **PASS** |
| Feature Ablation | 75× worse without g_bar | Ratio > 10 | **PASS** |
| Random Features | g_bar 99× more important | Ratio > 5 | **PASS** |

**Interpretation:** Selection reflects genuine structure in the data.

### 3.3 Final Test (LOCKED, Run Once)

| Model | Test MAE | Notes |
|-------|----------|-------|
| **RAR (selected)** | **0.247** | g† = 1.20e-10 fitted on train |
| Linear (fitted) | 0.289 | α, β fitted on train |
| MOND-simple (fitted) | 0.306 | a₀ fitted on train |

**Result:** Selected model (RAR) generalizes to locked test galaxies.

---

## 4. Discussion

### 4.1 What This Demonstrates

1. **Pipeline validity:** Galaxy-level holdout prevents leakage
2. **Metric appropriateness:** Log-space MAE handles acceleration range
3. **Control effectiveness:** Kill tests distinguish real vs. spurious
4. **Calibration:** Pipeline correctly recovers known physics (RAR)

### 4.2 What This Does NOT Demonstrate

1. **Not discovery:** RAR was in the candidate set
2. **Not new physics:** RAR is well-established (McGaugh et al. 2016)
3. **Not theory discrimination:** Cannot distinguish dark matter vs. MOND explanations

### 4.3 Value of This Work

This work provides a **calibrated benchmark**. Any future claim of discovering new physics via automated methods should:

1. Use galaxy-level (not point-level) holdout
2. Pass equivalent kill tests
3. Compare against fitted (not default) baselines
4. Demonstrate the pipeline can recover known physics

---

## 5. Conclusions

We have demonstrated a rigorous model selection pipeline for galactic acceleration laws. The pipeline:

1. Correctly selects RAR from a menu of candidates
2. Confirms generalization to 35 held-out galaxies
3. Passes three kill tests verifying non-spurious selection
4. Recovers the known physical constant g† = 1.2 × 10⁻¹⁰ m/s²

This establishes a methodological baseline for future discovery claims.

---

## References

1. McGaugh, S. S., Lelli, F., & Schombert, J. M. (2016). Radial Acceleration Relation in Rotationally Supported Galaxies. *Physical Review Letters*, 117(20), 201101.

2. Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016). SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves. *The Astronomical Journal*, 152(6), 157.

---

## Appendix: Technical Corrections from v1

| Issue | v1 (incorrect) | v2 (corrected) |
|-------|----------------|----------------|
| Baselines | Hardcoded `linear_form(g_bar, 1.5, 0)` | Fitted on train |
| Fit objective | Point-wise curve_fit | Galaxy-weighted log-MAE |
| Permutation test | Fixed model, shuffle data | Re-fit all candidates |
| MOND form | μ(g_bar/a₀) | μ(g_obs/a₀) implicit form |

---

**Word Count:** ~1,200
