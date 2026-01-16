# Paper 2: Formula-Free Discovery of Galactic Acceleration Laws Using Fractal Logic Discovery

## Discovering the Dark Matter Relationship Without Assuming a Functional Form

**Date:** January 15, 2026  
**Dataset:** SPARC (Spitzer Photometry and Accurate Rotation Curves) - 175 Galaxies  
**Methodology:** Universal Discovery Engine (UDE) - Fractal Genetic Programming with 4-Law Validation  
**Builds On:** Paper 1 (Independent Verification of the Gravitational Overflow Hypothesis)

---

## Abstract

Paper 1 established that within-galaxy sigmoid fits work well but no universal sigmoid generalizes across galaxies. This paper takes a fundamentally different approach: instead of assuming a functional form (sigmoid, MOND interpolation, etc.), we use the Universal Discovery Engine (UDE) to **reverse-engineer** the acceleration law directly from SPARC rotation curve data.

**Key Questions:**
1. Can UDE discover MOND's interpolating function μ(x) = x/√(1+x²) from raw data?
2. Can UDE discover a novel formula that outperforms known models?
3. Does the discovered formula generalize across galaxies?

**Methodology:**
- Fractal Genetic Programming (O(1) complexity pattern discovery)
- 4-Law Statistical Validation (Benford, Zipf, LLN, Pareto)
- Cross-galaxy generalization testing
- Comparison with known physics (Newton, MOND, RAR)

---

## 1. Introduction

### 1.1 The Limitation of Assumed Functional Forms

Paper 1 tested a specific hypothesis: that V_obs follows a sigmoid function of x = V̂_bar + R̂. While sigmoids fit individual galaxies well, no universal sigmoid generalizes. This raises a deeper question: **what is the actual functional relationship?**

Traditional approaches assume a form:
- **Newton:** V_obs = V_bar
- **MOND:** g_obs = ν(g_bar/a₀) × g_bar, where ν is an interpolating function
- **RAR:** g_obs = g_bar / (1 - e^{-√(g_bar/g†)})
- **Paper 1 Sigmoid:** V_obs = a + (b-a)/(1 + exp(-s(x-x₀)))

Each assumes structure. What if we let the data speak?

### 1.2 The UDE Approach

The Universal Discovery Engine (UDE) uses **fractal genetic programming** to discover formulas from raw data without assuming a functional form. Key features:

1. **Zero-assumption discovery**: No pre-specified function class
2. **Fractal coordinate descent**: O(1) complexity for pattern detection
3. **4-Law validation**: Benford + Zipf + LLN + Pareto for statistical robustness
4. **Bit-level reconstruction**: Discovers formulas that reconstruct output bit-by-bit

### 1.3 Research Questions

1. **Discovery**: What formula does UDE find for V_obs = f(V_bar, R, galaxy_properties)?
2. **Comparison**: Does it match known physics (MOND, RAR)?
3. **Generalization**: Does the discovered formula predict held-out galaxies?
4. **Novelty**: Is there structure beyond known models?

---

## 2. Methodology

### 2.1 Data Preparation

From SPARC, for each data point we have:
- **Inputs**: V_bar (baryonic velocity), R (radius), V_gas, V_disk, V_bul
- **Output**: V_obs (observed velocity)
- **Galaxy-level**: Total mass, scale length, surface brightness, morphology

We normalize per-galaxy to create:
- x = V̂_bar + R̂ (composite coordinate from Paper 1)
- Alternative: raw (V_bar, R) without composite

### 2.2 UDE Discovery Pipeline

```
1. TRUTH TABLE GENERATION
   - Convert (V_bar, R) → V_obs mapping to bit-level truth table
   - 8-bit resolution for velocities (0-255 km/s range)

2. FRACTAL PATTERN DISCOVERY
   - For each output bit, discover formula using FractalGene
   - O(1) search via coordinate descent in fractal space

3. FORMULA RECONSTRUCTION
   - Combine bit-level formulas into full V_obs prediction
   - Simplify using algebraic rules

4. 4-LAW VALIDATION
   - Benford: First-digit distribution of residuals
   - Zipf: Rank-frequency of formula terms
   - LLN: Convergence across galaxy samples
   - Pareto: 80/20 rule for error distribution

5. CROSS-GALAXY GENERALIZATION
   - Train on 80% of galaxies
   - Test on held-out 20%
   - Compare with Newton, MOND, RAR baselines
```

### 2.3 Comparison Baselines

| Model | Formula | Free Parameters |
|-------|---------|-----------------|
| Newton | V_obs = V_bar | 0 |
| MOND (simple) | g_obs = g_bar × (1 + √(a₀/g_bar)) | 1 (a₀) |
| RAR | g_obs = g_bar / (1 - e^{-√(g_bar/g†)}) | 1 (g†) |
| Paper 1 Sigmoid | V_obs = a + (b-a)/(1+exp(-s(x-x₀))) | 4 per galaxy |
| **UDE Discovery** | **TBD** | **TBD** |

---

## 3. Results

### 3.1 Discovered Formula

[TO BE FILLED BY UDE ANALYSIS]

### 3.2 Comparison with Known Physics

[TO BE FILLED]

### 3.3 Cross-Galaxy Generalization

[TO BE FILLED]

### 3.4 4-Law Validation Scores

[TO BE FILLED]

---

## 4. Discussion

### 4.1 Does UDE Rediscover MOND?

[TO BE FILLED]

### 4.2 Novel Structure

[TO BE FILLED]

### 4.3 Implications for Dark Matter vs Modified Gravity

[TO BE FILLED]

---

## 5. Conclusions

[TO BE FILLED]

---

## Appendix A: UDE Technical Details

### A.1 Fractal Genetic Programming

Unlike traditional GP which evolves random trees, UDE projects logic into fractal coordinate space:

1. **Coordinate transformation**: Map inputs to Iterated Function System (IFS) coordinates
2. **Self-similarity detection**: Find patterns that repeat at multiple scales
3. **Rule extraction**: Convert fractal patterns to executable AST
4. **Verification**: Formal proof via induction

### A.2 4-Law Statistical Framework

Combined confidence score:
```
C = 0.30×Benford + 0.25×Zipf + 0.25×LLN + 0.20×Pareto
```

- **Benford**: Natural emergence indicator (first-digit distribution)
- **Zipf**: Anomaly detection (rank-frequency)
- **LLN**: Sample sufficiency (convergence rate)
- **Pareto**: Imbalance detection (80/20 rule)

---

## References

1. Paper 1: Independent Verification of the Gravitational Overflow Hypothesis
2. McGaugh, S.S., Lelli, F., Schombert, J.M. (2016). Radial Acceleration Relation in Rotationally Supported Galaxies.
3. Milgrom, M. (1983). A modification of the Newtonian dynamics.
4. UDE-CORE Documentation: Fractal-based logic discovery engine.

---

*This paper uses the Universal Discovery Engine to discover physics formulas from first principles, without assuming a functional form.*
