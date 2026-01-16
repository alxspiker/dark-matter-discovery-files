# Independent Verification and Comprehensive Analysis of the Gravitational Overflow Hypothesis

## A Data-Driven Investigation into Nonlinear Relationships in Galactic Rotation Curves

**Date:** January 15, 2026  
**Dataset:** SPARC (Spitzer Photometry and Accurate Rotation Curves) - 175 Galaxies  
**Methodology:** Statistical Analysis, Predictive Modeling, and Independent Verification

---

## Abstract (Corrected)

This paper presents an independent verification and critical analysis of the "Gravitational Overflow Hypothesis" - an exploratory hypothesis proposing that a simple composite coordinate organizes rotation-curve behavior in spiral galaxies. Through rigorous statistical analysis of the SPARC galaxy database (143 galaxies with sufficient data points), we find:

**What IS supported:**
1. Within individual galaxies, a sigmoid-like transition in x = V̂_bar + R̂ fits rotation curves well (median R² = 0.97, in-sample; not cross-validated)
2. Sigmoid significantly outperforms linear within galaxies (97.5% of cases, p < 0.0001)
3. A galaxy-level *low-x vs high-x velocity separation* is present (Wilcoxon p ≈ 1.77×10⁻²⁷), consistent with a smooth nonlinear transition

**What is NOT supported:**
1. The theoretical threshold x = 1.0 is NOT within the 95% CI for transition centers (mean x₀ = 0.895, CI [0.834, 0.959])
2. A single "universal sigmoid" does NOT generalize across galaxies — held-out galaxy R² = -35 (raw), and even with scale-adjustment, linear marginally outperforms sigmoid
3. The "digital" interpretation is not supported — smooth curvature (quadratic) beats discrete kinks (piecewise) in 61% of galaxies
4. Flatness correlation (r = 0.37) has no out-of-sample predictive power (CV R² ≈ 0)

**Honest conclusion:** The Gravitational Overflow model is a **useful per-galaxy approximation** that captures real nonlinear behavior in rotation curves, but it is NOT a universal law. Sigmoid parameters vary substantially across galaxies, and no single function f(x) predicts unseen galaxies better than a simple linear model.

---

## 1. Introduction

### 1.1 The Dark Matter Problem

One of the most profound mysteries in modern astrophysics is the behavior of rotation curves in spiral galaxies. According to Newtonian mechanics, orbital velocity should decline with distance from the galactic center following:

$$V(r) \propto \frac{1}{\sqrt{r}}$$

However, observations consistently reveal that rotation curves remain remarkably flat at large radii, with stars orbiting at constant velocities far exceeding Newtonian predictions. This discrepancy has traditionally been explained by two competing hypotheses:

1. **Dark Matter Theory**: Invisible, non-baryonic matter comprising ~85% of total matter creates an extended halo that maintains gravitational pull at large radii.

2. **Modified Newtonian Dynamics (MOND)**: At low accelerations, gravitational physics fundamentally deviates from Newton's law.

### 1.2 The Gravitational Overflow Hypothesis

This investigation tests the "Gravitational Overflow Hypothesis," which proposes that the relationship between mass, radius, and velocity might operate according to discrete digital logic rather than continuous physical laws. Specifically, we evaluate the claim that:

> When the normalized sum of baryonic velocity and radius exceeds a critical threshold (1.0), the galactic system "latches" into a high-velocity state, creating the flat rotation curve.

**Note:** Subsequent sections evaluate whether the data actually favor such a discrete latch versus a smooth nonlinear transition, and whether x=1.0 is a special threshold.

This can be expressed mathematically as:

$$V_{obs} = \begin{cases} V_{bar} & \text{if } \frac{V_{bar}}{V_{max}} + \frac{R}{R_{max}} < 1.0 \\ V_{flat} & \text{if } \frac{V_{bar}}{V_{max}} + \frac{R}{R_{max}} \geq 1.0 \end{cases}$$

---

## 2. Methodology

### 2.1 Data Source

We utilized the SPARC (Spitzer Photometry and Accurate Rotation Curves) database, which provides high-quality rotation curve measurements for 175 galaxies. The dataset includes:

- **Radius (R)**: Distance from galactic center in kiloparsecs (kpc)
- **V_obs**: Observed rotational velocity (km/s)
- **V_gas**: Contribution from gas component (km/s)
- **V_disk**: Contribution from stellar disk (km/s)
- **V_bul**: Contribution from bulge component (km/s)

The baryonic velocity was computed as:
$$V_{bar} = \sqrt{V_{gas}^2 + V_{disk}^2 + V_{bul}^2}$$

### 2.2 Analysis Framework

Our independent verification employed multiple statistical and predictive approaches:

1. **Velocity Separation Analysis**: Testing whether the threshold creates statistically significant separation in observed velocities
2. **Predictive Model Comparison**: Comparing RMSE of multiple models (Newtonian, Overflow, Mass-only, Radius-only, MOND)
3. **Threshold Sweep Analysis**: Finding the optimal threshold value
4. **Cross-Galaxy Generalization**: Testing whether models trained on some galaxies predict held-out galaxies

### 2.3 Statistical Methods

- Welch's t-test for independent samples
- Wilcoxon signed-rank test for paired comparisons
- RMSE (Root Mean Square Error) for predictive accuracy
- Binned mean analysis for trend visualization

---

## 3. Results

### 3.1 Dark Matter Discrepancy Confirmation

Analysis of the outer regions (last 5 data points) of each galaxy confirms the fundamental anomaly:

| Metric | Value |
|--------|-------|
| Galaxies Analyzed | 171 |
| Mean V_obs/V_bar (outer regions) | 1.64x |
| Median V_obs/V_bar (outer regions) | 1.60x |
| Galaxies where V_obs > V_bar at edge | 151/171 (88.3%) |

This confirms that observed velocities systematically exceed Newtonian predictions, demonstrating the robust presence of the "dark matter" anomaly across the sample.

### 3.2 Velocity Separation by Threshold Region

The central test examines whether the normalized sum creates a distinct separation in observed velocities:

| Region | Count | Mean Norm V_obs | Std Dev |
|--------|-------|-----------------|---------|
| Below Threshold (sum < 1.0) | 1,287 | 0.632 | 0.23 |
| Above Threshold (sum ≥ 1.0) | 2,088 | 0.860 | 0.14 |
| **Velocity Jump Factor** | - | **1.36x** | - |

**Statistical Significance:**
- T-test p-value: **9.29 × 10⁻²⁷³**
- Effect size (Cohen's d): **1.23** (large effect)

This extremely low p-value confirms that the velocity separation is statistically significant — points above the threshold exhibit systematically higher velocities. However, this does NOT distinguish between a discrete "phase transition" and a smooth nonlinear relationship.

**Caveat:** With ~3,300 pooled data points, even small effects produce extreme p-values. For a test that treats each galaxy as an independent unit (avoiding pseudo-replication), see **Section 3.7** (galaxy-level p = 1.77×10⁻²⁷, still highly significant).

### 3.3 Threshold Optimization

Sweeping threshold values from 0.5 to 1.5 revealed:

- **Optimal threshold for separation**: 0.75-1.0
- **Maximum separation achieved**: 0.326 (normalized velocity units)
- The theoretical value of 1.0 (carry-out point) falls within the optimal range

### 3.4 Predictive Model Comparison

We compared five predictive models across all 171 galaxies (note: later sigmoid analysis in Section 3.10 uses 143 galaxies with ≥8 data points for reliable curve fitting):

| Model | Mean RMSE (km/s) | Median RMSE (km/s) | Best Model Count |
|-------|------------------|---------------------|------------------|
| Newtonian | 33.96 | 33.12 | 2 |
| **Overflow Latch** | **16.86** | **14.43** | **76** |
| Mass-only Latch | 22.93 | 20.10 | 42 |
| Radius-only Latch | 20.86 | 16.47 | 42 |
| MOND-like heuristic* | 42.30 | 28.34 | 38 |

*Note: "MOND-like heuristic" is a simple acceleration-based scaling, NOT a full RAR/MOND fit with proper interpolating function. This baseline should not be interpreted as a test of MOND itself.

**Key Finding:** The Overflow model outperforms Newtonian predictions in **168/171 galaxies (98.2%)**, achieving a **50.3% reduction** in average prediction error.

**Model fitting details:** For each model, V_flat (the asymptotic velocity) is estimated per galaxy as the **mean of V_obs in the overflow region** (points where x ≥ 1.0). If no points exceed the threshold, the mean of the outer 30% of radii is used. The "Overflow Latch" model fits a threshold-based piecewise function; the "Newtonian" baseline uses V_bar directly. All models have at least one fitted parameter per galaxy.

### 3.5 Bit-Level Pattern Analysis (Summary)

The original hypothesis proposed "digital" behavior at the bit level. When digitizing velocities to 8-bit representation and comparing with carry-out logic, the MSB shows 70.2% match rate (exceeding 50% chance). **However**, subsequent analysis (Section 3.9) showed that smooth curvature fits better than discrete kinks, so this pattern is better explained as an artifact of the smooth nonlinear relationship. **Conclusion:** The bit-level correlation is not meaningful evidence for discrete physics. See **Appendix B** for full details.

### 3.6 Model Robustness

Testing the Overflow model against alternative latching mechanisms:

| Comparison | Overflow Wins | Mean Improvement | Wilcoxon p-value |
|-----------|---------------|------------------|------------------|
| vs Radius Latch | 122/165 (73.9%) | 6.98 km/s | < 0.0001 |
| vs Mass Latch | 117/165 (70.9%) | 3.57 km/s | < 0.0001 |

The Overflow model's superiority is statistically significant against both baseline alternatives.

### 3.7 Galaxy-Level Independence Test (Pseudo-Replication Control)

A critical concern in aggregated analyses is pseudo-replication — where pooling thousands of data points from multiple galaxies may create artificial statistical significance due to non-independence. To address this, we conducted a **galaxy-level velocity separation test** treating each galaxy as a single independent observation.

**Methodology:**
1. For each galaxy, normalize inputs using blind per-galaxy scaling
2. Split data points into "Low Regime" (sum < 1.0) and "High Regime" (sum ≥ 1.0)
3. Calculate the mean normalized V_obs for both regimes within each galaxy
4. Compute the "Velocity Jump" (Mean_High - Mean_Low) per galaxy
5. Apply statistical tests to the collection of independent galaxy jumps

**Results:**

| Metric | Value |
|--------|-------|
| Galaxies with Data in Both Regimes | 160 |
| Mean Velocity Jump | +0.580 (normalized units) |
| Median Velocity Jump | +0.590 (normalized units) |
| **Wilcoxon p-value** | **1.77 × 10⁻²⁷** |
| **T-Test p-value** | **1.01 × 10⁻⁴⁵** |

**Interpretation:** Even when treating each galaxy as a single independent unit (removing the concern of massive galaxies dominating statistics), the velocity separation remains **statistically significant** (Wilcoxon p = 1.77 × 10⁻²⁷). This addresses the pseudo-replication critique — the effect is real at the galaxy population level, not an artifact of pooled statistics.

**Important caveat:** This shows that high-x regions have higher velocities than low-x regions, but does NOT establish a discrete "phase transition" — a smooth nonlinear relationship also produces this pattern.

### 3.8 Rigorous Breakpoint Analysis (Kink vs Smooth Curve)

A deeper critique argues that the velocity separation may simply reflect that outer regions have higher velocities than inner regions (a smooth monotonic relationship), rather than a true discontinuity/kink at x=1.0. To address this, we conducted five rigorous tests specifically designed to detect a **genuine breakpoint** rather than just "outer > inner."

**Initial Test Results (Mean-Jump Based):**

| Test | Description | Result | Verdict |
|------|-------------|--------|---------|
| 1. Breakpoint Model | Piecewise linear vs smooth linear (AIC/BIC) | Piecewise wins 75.8% | ✅ PASS |
| 2. Local Jump | Narrow band (±0.15) around threshold only | 93.7% positive, p < 10⁻¹⁷ | ✅ PASS |
| 3. Random Controls | Compare x=1.0 vs random thresholds | z = 1.63 (91st percentile) | ❌ FAIL |
| 4. Threshold Sweep | Find where effect peaks | Peak at 0.50, not 1.0 | ❌ FAIL |
| 5. Sign Robustness | Fraction positive + bootstrap CI | 94.7% positive, CI excludes 0 | ✅ PASS |

### 3.9 Upgraded Analysis: ΔAIC-Based Controls and Kink vs Curvature

A key criticism of the initial rigorous tests was that Tests 3-4 measured "outer > inner" using mean-jump scores, which are not specific to breakpoint detection. We upgraded the analysis to use **ΔAIC-based controls** and added a critical test: **piecewise (kink) vs quadratic (smooth curvature)**.

**Upgraded Test Results:**

| Test | Description | Result | Verdict |
|------|-------------|--------|---------|
| 1. Piecewise vs Linear @ x=1.0 | ΔAIC comparison | Piecewise wins 75.8% | ✅ PASS |
| 2. Local Jump (±0.15 band) | Narrow band test | 93.7% positive, p < 10⁻¹⁷ | ✅ PASS |
| 3. ΔAIC Rank of x=1.0 | Is x=1.0 special among random breakpoints? | 49th percentile | ❌ FAIL |
| 4. Best Breakpoint Distribution | Where does ΔAIC peak per galaxy? | Mean = 1.17 | ⚠️ PARTIAL |
| 5. Piecewise vs Quadratic | Kink vs smooth curvature (both 3 params) | Quadratic wins 61.2% | ❌ FAIL |
| 6. Sign Robustness | Fraction positive + bootstrap CI | 94.7% positive, CI excludes 0 | ✅ PASS |

**Critical New Findings:**

**Test 3 (Upgraded) - ΔAIC Rank of x=1.0:**
- For each galaxy, compared ΔAIC at x=1.0 vs 50 random breakpoints
- Mean percentile rank of x=1.0: **49%** (no better than random)
- Only 23.8% of galaxies have x=1.0 in their top 20% of breakpoints
- Wilcoxon p-value: 0.635 (not significant)
- **Implication**: x=1.0 is NOT a uniquely special threshold location

**Test 4 (Upgraded) - Best Breakpoint Distribution:**
- Found optimal breakpoint per galaxy via ΔAIC maximization
- Mean best breakpoint: **x = 1.17** (not 1.0)
- Median: 1.175, Std Dev: 0.231
- Only 40.6% of galaxies have optimal breakpoint within ±0.15 of 1.0
- ΔAIC at x=1.0 achieves only **56.3%** of optimal ΔAIC
- **Implication**: The theoretical threshold of 1.0 is close but not optimal

**Test 5 (NEW) - Piecewise vs Quadratic:**
- Both models have 3 parameters (fair comparison)
- **Quadratic wins in 61.2% of galaxies**
- Mean ΔAIC (Quadratic - Piecewise): -2.87 (favoring quadratic)
- Wilcoxon p-value: **3.4 × 10⁻⁴**
- **Implication**: The relationship is better described by SMOOTH CURVATURE than a discrete KINK

**Revised Interpretation (3.5/6 tests passed):**

The upgraded analysis reveals a more nuanced picture:

- ✅ **A real descriptive relationship exists** — piecewise at x=1.0 beats linear, local jump is robust
- ✅ **Effect is highly consistent** — 95% of galaxies show positive velocity increase
- ⚠️ **Optimal threshold is ~1.17, not 1.0** — best breakpoints cluster slightly above the theoretical value
- ❌ **x=1.0 is NOT uniquely special** — it ranks at the 49th percentile among random breakpoints
- ❌ **Smooth curvature beats discrete kink** — quadratic (3 params) outperforms piecewise (3 params) in 61% of galaxies

**Revised Conclusion:**

The data strongly supports that there is a **real, descriptive within-galaxy relationship** between the normalized baryonic input (V_bar + R) and observed velocity. However, this relationship appears to be **smoothly nonlinear (curved)** rather than a **discrete phase transition (kink)**. The "overflow" threshold of x=1.0 captures this relationship reasonably well but is not a special discontinuity point — a smooth function with similar complexity fits the data better.

This suggests the "Gravitational Overflow Hypothesis" may be better understood as an **effective approximation** to an underlying smooth acceleration law (similar to MOND's interpolating function) rather than evidence of discrete digital physics.

### 3.10 Final Validation with Robust Statistics (CORRECTED)

Addressing reviewer-grade concerns, we conducted final validation with robust inference methods, circularity checks, and **rigorous cross-galaxy generalization testing**.

**Part 1: Within-Galaxy Sigmoid Fits (In-Sample Goodness-of-Fit)**

| Metric | Value |
|--------|-------|
| Galaxies with successful sigmoid fits | 118/143 |
| Mean sigmoid R² | 0.863 |
| Median sigmoid R² | **0.976** |
| Sigmoid beats linear within-galaxy | **97.5%** (115/118) |
| Paired t-test (sigmoid vs linear) | **p < 0.0001** |

**Note:** These R² values are in-sample goodness-of-fit metrics (all data points used for fitting), not cross-validated predictive accuracy. Within-galaxy cross-validation over radii was not the primary target of this work.

✅ **SUPPORTED**: Sigmoid fits significantly better than linear within individual galaxies (in-sample).

**Part 2: Transition Midpoint x₀ (Corrected Results)**

| Test | Statistic | P-value | Interpretation |
|------|-----------|---------|----------------|
| Mean x₀ | **0.895** | - | Below 1.0 |
| Median x₀ | **0.873** | - | Below 1.0 |
| Std Dev | 0.319 | - | High variability |
| IQR | [0.684, 1.056] | - | Wide range |
| T-test (mean = 1.0) | t = -3.24 | **0.002** | **SIGNIFICANT** |
| Wilcoxon (median = 1.0) | - | **< 0.001** | **SIGNIFICANT** |
| Bootstrap 95% CI | **[0.834, 0.959]** | - | **Does NOT contain 1.0** |

❌ **NOT SUPPORTED**: x = 1.0 is **outside** the 95% CI. The mean transition point is significantly below 1.0.

**Part 3: Cross-Galaxy Generalization (Critical Failure)**

**Definition:** "Scale-adjusted R²" allows a per-galaxy affine transform (y' = αy + β) chosen to best match held-out galaxy levels. This tests whether the *shape* of the universal curve transfers across galaxies, not whether absolute predictions work. It is a "shape transfer" test, not a universal law test.

| Model | Raw R² (held-out) | Scale-adjusted R² | Interpretation |
|-------|-------------------|-------------------|----------------|
| Universal sigmoid | **-35.1** | **0.677** | Terrible raw, decent shape |
| Universal linear | **-34.9** | **0.685** | Marginally better than sigmoid |
| Sigmoid wins (raw) | 50/118 (42%) | - | Worse than coin flip |
| Sigmoid wins (adjusted) | 47/118 (40%) | - | Linear is better |

❌ **NOT SUPPORTED**: A single universal sigmoid does NOT generalize across galaxies. Even with scale-adjustment (allowing per-galaxy shift/scale), linear performs marginally better (R² = 0.685 vs 0.677).

**Part 4: Why Universal Sigmoid Fails — Parameter Heterogeneity**

| Parameter | Mean | Std Dev | CV (std/mean) |
|-----------|------|---------|---------------|
| a (lower asymptote) | 0.13 | 0.44 | **3.35** |
| b (upper asymptote) | 1.35 | 0.47 | 0.35 |
| x₀ (transition center) | 0.90 | 0.31 | **0.35** |
| s (steepness) | 16.5 | 27.4 | **1.66** |

High coefficient of variation in a, x₀, and s explains why a single universal sigmoid fails.

**Part 5: Flatness Correlation — Correlational, Not Predictive**

| Test | Value | Interpretation |
|------|-------|----------------|
| Pearson r | **0.372** | Modest correlation |
| Spearman ρ | **0.208** | Weaker monotonic relationship |
| P-values | < 0.05 | Both significant |
| Cross-val R² | **-0.046** | No predictive power on held-out data |

⚠️ **CORRELATIONAL ONLY**: Flatness correlates with x₀ (p < 0.05), but has **no out-of-sample predictive power** (CV R² ≈ 0). This is association, not prediction.

**Corrected Final Claim:**

> *"Within individual SPARC galaxies, a sigmoid-like transition in x = V̂_bar + R̂ fits the rotation curve well (median R² = 0.97). However, a single universal sigmoid does NOT generalize across galaxies — parameters vary too much (especially asymptotes). The mean transition center x₀ = 0.895 with 95% CI [0.834, 0.959] excludes 1.0. The 'overflow' model is a useful within-galaxy approximation, not a universal law."*

---

## 4. Discussion

### 4.1 Physical Interpretation (Corrected)

The comprehensive analysis reveals that the "Gravitational Overflow Hypothesis" captures a real phenomenon **within individual galaxies** but does **not constitute a universal law** applicable across galaxies.

**What the data actually shows:**
- ✅ Within-galaxy sigmoid fits work well (median R² = 0.97)
- ✅ Sigmoid significantly beats linear within galaxies (97.5%, p < 0.0001)
- ❌ Mean transition center x₀ = 0.895, 95% CI [0.834, 0.959] — **x=1.0 is outside the CI**
- ❌ Universal sigmoid does NOT generalize across galaxies (R² = -35, linear is marginally better even after scale adjustment)
- ⚠️ Flatness correlates with x₀ (r = 0.37) but has no predictive power (CV R² ≈ 0)

**Honest interpretation:**
- Each galaxy has its own sigmoid-like transition behavior
- The transition center (x₀) varies significantly across galaxies (mean = 0.89, std = 0.32)
- The theoretical threshold of x = 1.0 is NOT a special point — it's outside the 95% CI
- No single universal curve predicts unseen galaxies better than a simple linear model
- The model may be capturing per-galaxy modified gravity effects that vary with galaxy properties

### 4.2 Comparison with Traditional Dark Matter Models

| Aspect | Dark Matter | Overflow Hypothesis |
|--------|-------------|---------------------|
| Predictive Accuracy | High (with fitted halo parameters) | High **within** individual galaxies |
| Free Parameters | Multiple (halo mass, concentration, etc.) | 4 per galaxy (sigmoid parameters) |
| Physical Interpretation | Invisible matter | Smooth nonlinear transition |
| Universality | Requires per-galaxy fitting | **Also requires per-galaxy fitting** |

**Note**: The Overflow model does NOT achieve universality — it requires per-galaxy parameter fitting just like dark matter halo models.

### 4.3 On the "Digital" Interpretation (Not Supported)

The original hypothesis proposed a "digital horizon" with discrete bit-level behavior. Our analysis found:

- ❌ Smooth curvature (quadratic) beats discrete kinks (piecewise) in 61% of galaxies
- ❌ The 70.2% bit-level correlation is better explained by smooth nonlinearity
- ❌ No evidence supports discrete "digital" physics over continuous transitions

The "digital" framing should be abandoned in favor of a continuous smooth transition interpretation.

### 4.4 Limitations and Caveats

1. **Correlation vs Causation**: Statistical correlation does not establish physical mechanism
2. **Sample Selection**: Results depend on SPARC sample characteristics
3. **No Universal Law**: A single universal sigmoid does NOT generalize across galaxies — held-out R² is -35 (raw), linear marginally better even with scale-adjustment
4. **x = 1.0 Not Special**: The theoretical threshold of 1.0 is outside the 95% CI for mean x₀ (0.895, CI [0.834, 0.959])
5. **Discrete vs Continuous**: The "digital" interpretation is NOT supported — smooth curvature beats discrete kinks in 61% of galaxies
6. **Galaxy-Dependent Transitions**: Sigmoid parameters vary substantially (x₀ CV = 0.35, s CV = 1.66, a CV = 3.35)
7. **Flatness Not Predictive**: Flatness correlation (r = 0.37) is significant but has no out-of-sample predictive power (CV R² ≈ 0)
8. **Mathematical Equivalence**: The overflow model may be mathematically equivalent to existing modified gravity theories (MOND) in a different parameterization

---

## 5. Conclusions

This independent verification **partially confirms but substantially revises** the Gravitational Overflow Hypothesis:

### 5.1 Primary Findings (Corrected)

1. ✅ **Within-Galaxy Sigmoid Works**: Sigmoid fits individual galaxies well (median R² = 0.97) and significantly beats linear (97.5%, p < 0.0001)

2. ✅ **Galaxy-Level Independence Confirmed**: Per-galaxy analysis (N=160) confirms velocity separation with p < 10⁻²⁷ (but this is consistent with smooth nonlinearity, not necessarily a discrete transition)

3. ❌ **x=1.0 NOT Within 95% CI**: Mean x₀ = 0.895, **95% CI [0.834, 0.959] does NOT contain 1.0**; T-test p = 0.002 (SIGNIFICANT)

4. ❌ **Universal Sigmoid Does NOT Generalize**: Raw out-of-galaxy R² = -35; even scale-adjusted, linear marginally beats sigmoid (0.685 vs 0.677)

5. ❌ **Sigmoid Shape Not Better Than Linear**: When predicting held-out galaxies (with scale adjustment), sigmoid wins only 40% of the time

6. ⚠️ **High Parameter Heterogeneity**: Sigmoid parameters vary substantially across galaxies (x₀ CV = 0.35, s CV = 1.66, a CV = 3.35)

7. ⚠️ **Flatness Correlation Weak**: Pearson r = 0.37 is significant (p < 0.001), but cross-validated R² ≈ 0 (no predictive power)

8. ⚠️ **Smooth Transition, Not Discrete Kink**: Quadratic beats piecewise in 61% of galaxies — no evidence for discrete physics

9. ✅ **Practical Utility Within Galaxies**: The "overflow" approximation works well for individual galaxies, even if not a universal law

### 5.2 Corrected Implications (Final)

The complete analysis with rigorous cross-validation definitively establishes:

**What IS supported:**
- ✅ Within-galaxy sigmoid fits work well (median R² = 0.97 in-sample, beats linear in 97.5%)
- ✅ A real descriptive within-galaxy relationship exists between V̂_bar + R̂ and V_obs
- ✅ The "overflow" approximation has practical utility for individual galaxies
- ✅ Galaxy-level velocity separation is robust (p < 10⁻²⁷, consistent with smooth nonlinearity)

**What is NOT supported:**
- ❌ **x = 1.0 is NOT the transition center** — mean x₀ = 0.895, 95% CI [0.834, 0.959] excludes 1.0
- ❌ **Universal sigmoid does NOT generalize** — held-out galaxy R² is -35 (raw) or 0.68 (scale-adjusted), linear is marginally better
- ❌ **No single f(x) works across galaxies** — parameters vary too much (especially asymptotes)
- ❌ **Flatness has no predictive power** — r = 0.37 is significant, but CV R² ≈ 0
- ❌ **Discrete "digital" physics** — quadratic beats piecewise in 61% of galaxies

**Honest scientific framing:**

> *"Within individual SPARC galaxies, normalized observed velocity follows a sigmoid-like function of x = V̂_bar + R̂ (median R² = 0.97). However, sigmoid parameters vary substantially across galaxies (x₀ mean = 0.89 ± 0.32), and a single universal sigmoid does NOT generalize to held-out galaxies better than a simple linear model. The 'overflow' threshold of x = 1.0 is outside the 95% CI for the mean transition point."*

**What the model actually is:**

The Gravitational Overflow Hypothesis is a **useful per-galaxy approximation** that captures real nonlinear behavior, but:
- It is NOT a universal law
- x = 1.0 is NOT a special threshold
- Parameters must be fit per-galaxy
- It may be a reparameterization of MOND-like physics

### 5.3 Future Directions

1. **External replication**: Test on independent galaxy surveys (THINGS, LITTLE THINGS)
2. **Compare with MOND**: Investigate mathematical relationship between sigmoid model and MOND interpolating functions
3. **Investigate flatness correlation**: Determine physical meaning of flatness-x₀ relationship
4. **Hierarchical model**: Fit per-galaxy sigmoids with partial pooling to estimate true population distribution
5. Compare with detailed dark matter halo simulations

---

## 6. Summary Statistics (Corrected)

| Metric | Value |
|--------|-------|
| Galaxies Analyzed | 143 (with ≥8 data points) |
| Total Data Points | ~3,375 |
| Velocity Separation p-value (per-galaxy) | 1.77 × 10⁻²⁷ |
| **Within-Galaxy Sigmoid Median R²** | **0.976** (in-sample) |
| **Within-Galaxy Sigmoid Mean R²** | **0.863** (in-sample) |
| Sigmoid beats Linear (within-galaxy) | **97.5%** (115/118) |
| **Mean Transition Center (x₀)** | **0.895** |
| **Median x₀** | **0.873** |
| **Std Dev x₀** | **0.319** |
| **95% CI for Mean x₀** | **[0.834, 0.959]** |
| **x = 1.0 within 95% CI?** | **❌ NO** |
| **T-test (mean = 1.0)** | **p = 0.002** (SIGNIFICANT) |
| **Universal Sigmoid R² (held-out, raw)** | **-35.1** |
| **Universal Sigmoid R² (scale-adjusted)** | **0.677** |
| **Universal Linear R² (scale-adjusted)** | **0.685** |
| **Sigmoid beats Linear (cross-galaxy)** | **40%** (47/118) |
| Flatness Correlation (Pearson r) | 0.372 (p < 0.001) |
| Flatness Cross-Val R² | **-0.046** (no predictive power) |

---

## Appendix A: Data Files Generated

| File | Description |
|------|-------------|
| `figure1_rotation_curves.png` | Example rotation curves showing the anomaly |
| `figure2_overflow_detection.png` | Velocity separation visualization |
| `figure3_model_comparison.png` | Predictive model comparison |
| `figure4_digital_horizon.png` | Bit-level accuracy analysis |
| `galaxy_phase_hist.png` | Per-galaxy velocity jump distribution |
| `phase_transition_rigorous.png` | Rigorous breakpoint analysis (5 tests) |
| `phase_transition_upgraded.png` | Upgraded ΔAIC-based analysis (6 tests) |
| `smooth_transition_analysis.png` | Initial smooth model analysis |
| `improved_smooth_analysis.png` | Statistically rigorous sigmoid analysis |
| `final_validation.png` | Robust statistics + cross-validation |
| `independent_analysis.py` | Python script for independent verification |
| `comprehensive_visualization.py` | Visualization generation script |
| `galaxy_phase_transition_test.py` | Galaxy-level independence test |
| `phase_transition_rigorous_test.py` | Rigorous breakpoint vs inner/outer analysis |
| `phase_transition_upgraded_test.py` | Upgraded ΔAIC controls + kink vs curvature |
| `smooth_transition_analysis.py` | Initial smooth transition analysis |
| `improved_smooth_analysis.py` | FDR-corrected sigmoid analysis |
| `final_validation.py` | Robust stats + cross-validation |
| `final_honest_validation.py` | **CORRECTED: Cross-galaxy generalization test (FINAL)** |

---

## Appendix B: Bit-Level Pattern Analysis (Full Details)

The original "Gravitational Overflow Hypothesis" proposed that galactic rotation curves exhibit discrete digital behavior at the bit level. We tested this by:

1. Digitizing normalized velocities to 8-bit representation (0-255)
2. Computing expected "carry-out" bit from normalized baryonic input
3. Comparing observed MSB with predicted carry-out

| Bit Position | Description | Carry-Out Match Rate |
|--------------|-------------|---------------------|
| Bit 7 (MSB) | Most Significant | **70.2%** |
| Bit 6 | Second | 58.1% |
| Bit 5 | Third | 52.3% |

The MSB correlation (70.2%) exceeds chance (50%), but this is fully explained by the smooth nonlinear relationship between x and V_obs. When x is high, both the MSB and the "carry-out" tend to be 1; when x is low, both tend to be 0. This correlation does not require discrete physics.

**Critical test:** Piecewise (discrete kink) vs Quadratic (smooth curvature) — Quadratic wins in 61.2% of galaxies (Wilcoxon p = 3.4×10⁻⁴). The data favor smooth transitions over discrete breaks.

**Conclusion:** The bit-level pattern is an artifact of the underlying smooth nonlinear relationship, not evidence of digital physics.

---

## Appendix C: Reproducibility

All analyses can be reproduced using:

```bash
cd dark_matter/
python independent_analysis.py
python comprehensive_visualization.py
python continuous_verification.py
python rmse_benchmark.py
python fair_baseline_benchmark.py
python galaxy_phase_transition_test.py  # Galaxy-level independence test
python phase_transition_rigorous_test.py  # Rigorous breakpoint analysis
python phase_transition_upgraded_test.py  # Upgraded ΔAIC + kink vs curvature
python smooth_transition_analysis.py  # Initial smooth model analysis
python improved_smooth_analysis.py  # FDR-corrected sigmoid analysis
python final_validation.py  # Initial robust stats
python final_honest_validation.py  # CORRECTED: Cross-galaxy generalization (FINAL)
```

---

## References

1. Lelli, F., McGaugh, S.S., Schombert, J.M. (2016). SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves. *The Astronomical Journal*, 152(6), 157.

2. McGaugh, S.S., Lelli, F., Schombert, J.M. (2016). Radial Acceleration Relation in Rotationally Supported Galaxies. *Physical Review Letters*, 117(20), 201101.

3. Milgrom, M. (1983). A modification of the Newtonian dynamics as a possible alternative to the hidden mass hypothesis. *The Astrophysical Journal*, 270, 365-370.

4. Rubin, V.C., Ford, W.K. Jr. (1970). Rotation of the Andromeda Nebula from a Spectroscopic Survey of Emission Regions. *The Astrophysical Journal*, 159, 379.

---

*This independent analysis was conducted to verify and extend the findings of the original Gravitational Overflow Hypothesis research. All code and data are available for reproducibility.*
