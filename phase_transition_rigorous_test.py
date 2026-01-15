"""
RIGOROUS PHASE TRANSITION TEST
==============================
Addressing Critique: "The gate is entangled with radius - outer > inner 
doesn't prove a discontinuity/kink at the threshold."

Goal: Establish that there is a genuine BREAK/KINK at x=1.0, not just 
that outer regions have higher velocities than inner regions.

Tests Implemented:
1. BREAKPOINT MODEL COMPARISON - Smooth vs Piecewise at x=1.0 (AIC/BIC)
2. LOCAL JUMP TEST - Narrow band around threshold only
3. RANDOM THRESHOLD CONTROLS - Show effect disappears with random gates
4. THRESHOLD SWEEP - Show score peaks at x=1.0
5. SIGN-BASED ROBUSTNESS - Fraction positive + bootstrap CI

This directly addresses whether the "phase transition" is a real 
discontinuity or just an artifact of inner/outer splitting.
"""

import os
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import re
import glob
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

SPARC_DIR = os.path.join(os.path.dirname(__file__), "dark_matter", "sparc_galaxies")

def load_galaxy_data(path):
    if not os.path.exists(path): return None
    radius, vobs, vgas, vdisk, vbul = [], [], [], [], []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip() or line.strip().startswith("#"): continue
            try:
                parts = re.split(r"\s+", line.strip())
                radius.append(float(parts[0]))
                vobs.append(float(parts[1]))
                vgas.append(abs(float(parts[3])))
                vdisk.append(abs(float(parts[4])))
                vbul.append(abs(float(parts[5])) if len(parts) > 5 else 0.0)
            except: continue
    
    vbar = np.sqrt(np.array(vgas)**2 + np.array(vdisk)**2 + np.array(vbul)**2)
    return np.array(radius), vbar, np.array(vobs)

def normalize_galaxy(rad, vbar, vobs):
    """Blind normalization per galaxy."""
    v_scale = vbar.max() if vbar.max() > 0 else 1.0
    r_scale = rad.max() if rad.max() > 0 else 1.0
    return rad / r_scale, vbar / v_scale, vobs / v_scale

# =============================================================================
# TEST 1: BREAKPOINT MODEL COMPARISON (AIC/BIC)
# =============================================================================
def fit_linear(x, y):
    """Fit y = a + b*x, return residuals and params."""
    if len(x) < 3: return None, None, np.inf
    A = np.column_stack([np.ones_like(x), x])
    try:
        params, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
        y_pred = A @ params
        ss_res = np.sum((y - y_pred)**2)
        return params, y_pred, ss_res
    except:
        return None, None, np.inf

def fit_piecewise_linear(x, y, breakpoint=1.0):
    """Fit piecewise linear with break at breakpoint (continuous)."""
    if len(x) < 5: return None, None, np.inf
    
    # y = a + b1*x for x < breakpoint
    # y = a + b1*breakpoint + b2*(x - breakpoint) for x >= breakpoint
    # Continuous at breakpoint by construction
    
    x_below = np.minimum(x, breakpoint)
    x_above = np.maximum(x - breakpoint, 0)
    
    A = np.column_stack([np.ones_like(x), x_below, x_above])
    try:
        params, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
        y_pred = A @ params
        ss_res = np.sum((y - y_pred)**2)
        return params, y_pred, ss_res
    except:
        return None, None, np.inf

def compute_aic_bic(n, k, ss_res):
    """Compute AIC and BIC from residual sum of squares."""
    if ss_res <= 0 or n <= k: return np.inf, np.inf
    # Log-likelihood assuming Gaussian errors
    log_lik = -n/2 * (np.log(2*np.pi) + np.log(ss_res/n) + 1)
    aic = 2*k - 2*log_lik
    bic = k*np.log(n) - 2*log_lik
    return aic, bic

def test_breakpoint_model(galaxies_data):
    """Compare smooth linear vs piecewise linear at x=1.0."""
    print("\n" + "="*60)
    print("TEST 1: BREAKPOINT MODEL COMPARISON (Smooth vs Piecewise)")
    print("="*60)
    
    results = []
    
    for name, (n_rad, n_vbar, n_vobs) in galaxies_data.items():
        x = n_vbar + n_rad
        y = n_vobs
        n = len(x)
        
        if n < 6: continue
        
        # Fit both models
        _, _, ss_linear = fit_linear(x, y)
        _, _, ss_piecewise = fit_piecewise_linear(x, y, breakpoint=1.0)
        
        if np.isinf(ss_linear) or np.isinf(ss_piecewise): continue
        
        # k = number of parameters (linear: 2, piecewise: 3)
        aic_linear, bic_linear = compute_aic_bic(n, 2, ss_linear)
        aic_piecewise, bic_piecewise = compute_aic_bic(n, 3, ss_piecewise)
        
        results.append({
            'name': name,
            'n': n,
            'aic_linear': aic_linear,
            'aic_piecewise': aic_piecewise,
            'bic_linear': bic_linear,
            'bic_piecewise': bic_piecewise,
            'delta_aic': aic_linear - aic_piecewise,  # Positive = piecewise better
            'delta_bic': bic_linear - bic_piecewise,
        })
    
    # Aggregate results
    delta_aic = np.array([r['delta_aic'] for r in results])
    delta_bic = np.array([r['delta_bic'] for r in results])
    
    piecewise_wins_aic = np.sum(delta_aic > 0)
    piecewise_wins_bic = np.sum(delta_bic > 0)
    total = len(results)
    
    print(f"\nGalaxies analyzed: {total}")
    print(f"\nPiecewise model wins (AIC): {piecewise_wins_aic}/{total} ({100*piecewise_wins_aic/total:.1f}%)")
    print(f"Piecewise model wins (BIC): {piecewise_wins_bic}/{total} ({100*piecewise_wins_bic/total:.1f}%)")
    print(f"\nMean ΔAIC (Linear - Piecewise): {np.mean(delta_aic):.2f}")
    print(f"Mean ΔBIC (Linear - Piecewise): {np.mean(delta_bic):.2f}")
    print(f"Median ΔAIC: {np.median(delta_aic):.2f}")
    print(f"Median ΔBIC: {np.median(delta_bic):.2f}")
    
    # Wilcoxon test on delta scores
    if len(delta_aic) > 10:
        _, p_aic = stats.wilcoxon(delta_aic)
        _, p_bic = stats.wilcoxon(delta_bic)
        print(f"\nWilcoxon p-value (ΔAIC ≠ 0): {p_aic:.4e}")
        print(f"Wilcoxon p-value (ΔBIC ≠ 0): {p_bic:.4e}")
    
    return results, delta_aic, delta_bic

# =============================================================================
# TEST 2: LOCAL JUMP TEST (Narrow Band Around Threshold)
# =============================================================================
def test_local_jump(galaxies_data, epsilon=0.15):
    """Test for jump using only points near the threshold."""
    print("\n" + "="*60)
    print(f"TEST 2: LOCAL JUMP TEST (ε = {epsilon})")
    print("="*60)
    print(f"Using only points where {1-epsilon:.2f} ≤ x < 1.0 vs 1.0 ≤ x ≤ {1+epsilon:.2f}")
    
    local_jumps = []
    valid_galaxies = 0
    
    for name, (n_rad, n_vbar, n_vobs) in galaxies_data.items():
        x = n_vbar + n_rad
        
        # Narrow bands around threshold
        mask_below = (x >= 1.0 - epsilon) & (x < 1.0)
        mask_above = (x >= 1.0) & (x <= 1.0 + epsilon)
        
        if np.sum(mask_below) >= 1 and np.sum(mask_above) >= 1:
            mean_below = np.mean(n_vobs[mask_below])
            mean_above = np.mean(n_vobs[mask_above])
            local_jump = mean_above - mean_below
            local_jumps.append(local_jump)
            valid_galaxies += 1
    
    local_jumps = np.array(local_jumps)
    
    print(f"\nGalaxies with data in both narrow bands: {valid_galaxies}")
    print(f"Mean local jump: {np.mean(local_jumps):+.4f}")
    print(f"Median local jump: {np.median(local_jumps):+.4f}")
    print(f"Std dev: {np.std(local_jumps):.4f}")
    
    # Fraction positive
    frac_positive = np.mean(local_jumps > 0)
    print(f"\nFraction of galaxies with positive jump: {frac_positive:.1%}")
    
    # Statistical tests
    if len(local_jumps) > 10:
        _, p_wilcox = stats.wilcoxon(local_jumps)
        _, p_ttest = stats.ttest_1samp(local_jumps, 0)
        print(f"Wilcoxon p-value: {p_wilcox:.4e}")
        print(f"T-test p-value: {p_ttest:.4e}")
    
    return local_jumps

# =============================================================================
# TEST 3: RANDOM THRESHOLD CONTROLS
# =============================================================================
def test_random_thresholds(galaxies_data, n_permutations=1000):
    """Compare real threshold (1.0) against random thresholds."""
    print("\n" + "="*60)
    print("TEST 3: RANDOM THRESHOLD CONTROLS")
    print("="*60)
    
    def compute_mean_jump(data, threshold):
        """Compute mean jump across all galaxies for a given threshold."""
        jumps = []
        for name, (n_rad, n_vbar, n_vobs) in data.items():
            x = n_vbar + n_rad
            mask_high = x >= threshold
            mask_low = x < threshold
            if np.sum(mask_high) > 0 and np.sum(mask_low) > 0:
                jump = np.mean(n_vobs[mask_high]) - np.mean(n_vobs[mask_low])
                jumps.append(jump)
        return np.mean(jumps) if jumps else 0
    
    # Real threshold score
    real_score = compute_mean_jump(galaxies_data, 1.0)
    print(f"Mean jump at threshold=1.0: {real_score:.4f}")
    
    # Generate random thresholds that give similar split ratios
    # First, find the typical x range
    all_x = []
    for name, (n_rad, n_vbar, n_vobs) in galaxies_data.items():
        all_x.extend(n_vbar + n_rad)
    all_x = np.array(all_x)
    
    # Random thresholds from observed x distribution
    random_scores = []
    for i in range(n_permutations):
        # Pick random threshold from data range (avoiding extremes)
        rand_thresh = np.random.uniform(np.percentile(all_x, 20), np.percentile(all_x, 80))
        score = compute_mean_jump(galaxies_data, rand_thresh)
        random_scores.append(score)
    
    random_scores = np.array(random_scores)
    
    # How extreme is the real score?
    percentile = np.mean(random_scores <= real_score) * 100
    z_score = (real_score - np.mean(random_scores)) / np.std(random_scores)
    
    print(f"\nRandom threshold distribution:")
    print(f"  Mean: {np.mean(random_scores):.4f}")
    print(f"  Std: {np.std(random_scores):.4f}")
    print(f"  Min: {np.min(random_scores):.4f}")
    print(f"  Max: {np.max(random_scores):.4f}")
    print(f"\nReal threshold (1.0) percentile: {percentile:.1f}%")
    print(f"Z-score vs random: {z_score:.2f}")
    
    if percentile > 95:
        print("✅ Real threshold significantly better than random")
    else:
        print("⚠️ Real threshold not significantly different from random")
    
    return real_score, random_scores

# =============================================================================
# TEST 4: THRESHOLD SWEEP (Peak Detection)
# =============================================================================
def test_threshold_sweep(galaxies_data):
    """Sweep threshold values and find where the effect peaks."""
    print("\n" + "="*60)
    print("TEST 4: THRESHOLD SWEEP (Peak Detection)")
    print("="*60)
    
    thresholds = np.linspace(0.5, 1.5, 51)
    scores = []
    
    for thresh in thresholds:
        jumps = []
        for name, (n_rad, n_vbar, n_vobs) in galaxies_data.items():
            x = n_vbar + n_rad
            mask_high = x >= thresh
            mask_low = x < thresh
            if np.sum(mask_high) > 0 and np.sum(mask_low) > 0:
                jump = np.mean(n_vobs[mask_high]) - np.mean(n_vobs[mask_low])
                jumps.append(jump)
        scores.append(np.mean(jumps) if jumps else 0)
    
    scores = np.array(scores)
    
    # Find peak
    peak_idx = np.argmax(scores)
    peak_thresh = thresholds[peak_idx]
    peak_score = scores[peak_idx]
    
    # Score at 1.0
    idx_1 = np.argmin(np.abs(thresholds - 1.0))
    score_at_1 = scores[idx_1]
    
    print(f"Peak threshold: {peak_thresh:.2f}")
    print(f"Peak score: {peak_score:.4f}")
    print(f"Score at threshold=1.0: {score_at_1:.4f}")
    print(f"Distance from peak to 1.0: {abs(peak_thresh - 1.0):.2f}")
    
    # Is 1.0 near the peak?
    if abs(peak_thresh - 1.0) <= 0.1:
        print("✅ Threshold 1.0 is at or near the peak")
    elif abs(peak_thresh - 1.0) <= 0.2:
        print("⚠️ Threshold 1.0 is close to but not at the peak")
    else:
        print("❌ Threshold 1.0 is far from the peak")
    
    return thresholds, scores, peak_thresh

# =============================================================================
# TEST 5: SIGN-BASED ROBUSTNESS + BOOTSTRAP CI
# =============================================================================
def test_sign_robustness(galaxies_data, n_bootstrap=10000):
    """Report sign-based statistics and bootstrap confidence intervals."""
    print("\n" + "="*60)
    print("TEST 5: SIGN-BASED ROBUSTNESS + BOOTSTRAP CI")
    print("="*60)
    
    # Compute jumps
    jumps = []
    for name, (n_rad, n_vbar, n_vobs) in galaxies_data.items():
        x = n_vbar + n_rad
        mask_high = x >= 1.0
        mask_low = x < 1.0
        if np.sum(mask_high) > 0 and np.sum(mask_low) > 0:
            jump = np.mean(n_vobs[mask_high]) - np.mean(n_vobs[mask_low])
            jumps.append(jump)
    
    jumps = np.array(jumps)
    n = len(jumps)
    
    # Sign-based statistics
    n_positive = np.sum(jumps > 0)
    n_negative = np.sum(jumps < 0)
    n_zero = np.sum(jumps == 0)
    
    print(f"\nTotal galaxies: {n}")
    print(f"Positive jumps: {n_positive} ({100*n_positive/n:.1f}%)")
    print(f"Negative jumps: {n_negative} ({100*n_negative/n:.1f}%)")
    print(f"Zero jumps: {n_zero}")
    
    # Binomial test (null: 50% chance of positive)
    binom_result = stats.binomtest(n_positive, n, 0.5, alternative='greater')
    p_binom = binom_result.pvalue
    print(f"\nBinomial test p-value (H0: 50% positive): {p_binom:.4e}")
    
    # Bootstrap CI for median
    bootstrap_medians = []
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(jumps, size=n, replace=True)
        bootstrap_medians.append(np.median(sample))
        bootstrap_means.append(np.mean(sample))
    
    median_ci = np.percentile(bootstrap_medians, [2.5, 97.5])
    mean_ci = np.percentile(bootstrap_means, [2.5, 97.5])
    
    print(f"\nMedian jump: {np.median(jumps):.4f}")
    print(f"95% Bootstrap CI for median: [{median_ci[0]:.4f}, {median_ci[1]:.4f}]")
    print(f"\nMean jump: {np.mean(jumps):.4f}")
    print(f"95% Bootstrap CI for mean: [{mean_ci[0]:.4f}, {mean_ci[1]:.4f}]")
    
    # Does CI exclude zero?
    if median_ci[0] > 0:
        print("\n✅ 95% CI for median excludes zero (robust effect)")
    else:
        print("\n⚠️ 95% CI for median includes zero")
    
    return jumps, median_ci, mean_ci

# =============================================================================
# VISUALIZATION
# =============================================================================
def create_visualizations(delta_aic, local_jumps, random_scores, real_score, 
                          thresholds, sweep_scores, jumps):
    """Create comprehensive visualization of all tests."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. AIC comparison histogram
    ax = axes[0, 0]
    ax.hist(delta_aic, bins=25, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Equal fit')
    ax.axvline(x=np.median(delta_aic), color='green', linestyle='-', linewidth=2, label=f'Median={np.median(delta_aic):.1f}')
    ax.set_xlabel('ΔAIC (Linear - Piecewise)')
    ax.set_ylabel('Count')
    ax.set_title('Test 1: Breakpoint Model Comparison')
    ax.legend()
    
    # 2. Local jump histogram
    ax = axes[0, 1]
    ax.hist(local_jumps, bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No effect')
    ax.axvline(x=np.median(local_jumps), color='green', linestyle='-', linewidth=2, label=f'Median={np.median(local_jumps):.3f}')
    ax.set_xlabel('Local Jump (narrow band)')
    ax.set_ylabel('Count')
    ax.set_title('Test 2: Local Jump Near Threshold')
    ax.legend()
    
    # 3. Random threshold comparison
    ax = axes[0, 2]
    ax.hist(random_scores, bins=30, color='gray', alpha=0.7, edgecolor='black', label='Random thresholds')
    ax.axvline(x=real_score, color='red', linestyle='-', linewidth=3, label=f'Threshold=1.0: {real_score:.3f}')
    ax.set_xlabel('Mean Jump Score')
    ax.set_ylabel('Count')
    ax.set_title('Test 3: Real vs Random Thresholds')
    ax.legend()
    
    # 4. Threshold sweep
    ax = axes[1, 0]
    ax.plot(thresholds, sweep_scores, 'b-', linewidth=2)
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Threshold=1.0')
    peak_idx = np.argmax(sweep_scores)
    ax.axvline(x=thresholds[peak_idx], color='green', linestyle=':', linewidth=2, label=f'Peak={thresholds[peak_idx]:.2f}')
    ax.set_xlabel('Threshold Value')
    ax.set_ylabel('Mean Jump Score')
    ax.set_title('Test 4: Threshold Sweep')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Full jump distribution with CI
    ax = axes[1, 1]
    ax.hist(jumps, bins=25, color='teal', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No effect')
    ax.axvline(x=np.median(jumps), color='green', linestyle='-', linewidth=2, label=f'Median={np.median(jumps):.3f}')
    ax.set_xlabel('Velocity Jump (Per Galaxy)')
    ax.set_ylabel('Count')
    ax.set_title('Test 5: Sign-Based Analysis')
    ax.legend()
    
    # 6. Summary text
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"""
    RIGOROUS PHASE TRANSITION ANALYSIS
    ===================================
    
    Test 1 - Breakpoint Model:
      Piecewise wins: {np.sum(delta_aic > 0)}/{len(delta_aic)} galaxies
      Median ΔAIC: {np.median(delta_aic):.1f}
    
    Test 2 - Local Jump:
      Galaxies tested: {len(local_jumps)}
      Median local jump: {np.median(local_jumps):.4f}
      Fraction positive: {np.mean(local_jumps > 0):.1%}
    
    Test 3 - Random Controls:
      Real score: {real_score:.4f}
      Random mean: {np.mean(random_scores):.4f}
      Z-score: {(real_score - np.mean(random_scores))/np.std(random_scores):.1f}σ
    
    Test 4 - Threshold Sweep:
      Peak at: {thresholds[np.argmax(sweep_scores)]:.2f}
      Score at 1.0: {sweep_scores[np.argmin(np.abs(thresholds-1.0))]:.4f}
    
    Test 5 - Sign Robustness:
      Positive: {np.sum(jumps > 0)}/{len(jumps)} ({100*np.mean(jumps > 0):.0f}%)
    """
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('phase_transition_rigorous.png', dpi=150)
    print("\n[SAVED] phase_transition_rigorous.png")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*60)
    print("RIGOROUS PHASE TRANSITION ANALYSIS")
    print("Addressing: Is there a REAL break at x=1.0, or just outer>inner?")
    print("="*60)
    
    # Load all galaxies
    files = glob.glob(os.path.join(SPARC_DIR, "*_rotmod.dat"))
    print(f"\nLoading {len(files)} galaxy files...")
    
    galaxies_data = {}
    for f in files:
        data = load_galaxy_data(f)
        if data is None or len(data[0]) < 6: continue
        
        rad, vbar, vobs = data
        n_rad, n_vbar, n_vobs = normalize_galaxy(rad, vbar, vobs)
        
        name = os.path.basename(f).replace('_rotmod.dat', '')
        galaxies_data[name] = (n_rad, n_vbar, n_vobs)
    
    print(f"Valid galaxies loaded: {len(galaxies_data)}")
    
    # Run all tests
    results_bp, delta_aic, delta_bic = test_breakpoint_model(galaxies_data)
    local_jumps = test_local_jump(galaxies_data, epsilon=0.15)
    real_score, random_scores = test_random_thresholds(galaxies_data, n_permutations=1000)
    thresholds, sweep_scores, peak_thresh = test_threshold_sweep(galaxies_data)
    jumps, median_ci, mean_ci = test_sign_robustness(galaxies_data)
    
    # Create visualizations
    create_visualizations(delta_aic, local_jumps, random_scores, real_score,
                          thresholds, sweep_scores, jumps)
    
    # Final Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Breakpoint wins majority
    bp_win_rate = np.mean(delta_aic > 0)
    if bp_win_rate > 0.5:
        print(f"✅ Test 1 PASSED: Piecewise model wins {bp_win_rate:.1%} of galaxies")
        tests_passed += 1
    else:
        print(f"❌ Test 1 FAILED: Piecewise model wins only {bp_win_rate:.1%}")
    
    # Test 2: Local jump significant
    if len(local_jumps) > 10:
        _, p_local = stats.wilcoxon(local_jumps)
        if p_local < 0.05 and np.median(local_jumps) > 0:
            print(f"✅ Test 2 PASSED: Local jump significant (p={p_local:.2e})")
            tests_passed += 1
        else:
            print(f"❌ Test 2 FAILED: Local jump not significant (p={p_local:.2e})")
    
    # Test 3: Real beats random
    z_score = (real_score - np.mean(random_scores)) / np.std(random_scores)
    if z_score > 1.96:
        print(f"✅ Test 3 PASSED: Real threshold beats random (z={z_score:.1f})")
        tests_passed += 1
    else:
        print(f"❌ Test 3 FAILED: Real threshold not better than random (z={z_score:.1f})")
    
    # Test 4: Peak near 1.0
    if abs(peak_thresh - 1.0) <= 0.15:
        print(f"✅ Test 4 PASSED: Peak at {peak_thresh:.2f}, close to 1.0")
        tests_passed += 1
    else:
        print(f"❌ Test 4 FAILED: Peak at {peak_thresh:.2f}, far from 1.0")
    
    # Test 5: Strong sign consistency
    frac_positive = np.mean(jumps > 0)
    if frac_positive > 0.9 and median_ci[0] > 0:
        print(f"✅ Test 5 PASSED: {frac_positive:.1%} positive, CI excludes zero")
        tests_passed += 1
    else:
        print(f"⚠️ Test 5 PARTIAL: {frac_positive:.1%} positive")
        if median_ci[0] > 0:
            tests_passed += 0.5
    
    print(f"\n{'='*60}")
    print(f"TESTS PASSED: {tests_passed}/{total_tests}")
    
    if tests_passed >= 4:
        print("\n✅✅ STRONG EVIDENCE: Phase transition is a REAL discontinuity")
        print("   The effect is NOT just 'outer > inner' - there is a genuine break.")
    elif tests_passed >= 3:
        print("\n⚠️ MODERATE EVIDENCE: Some support for phase transition")
        print("   More investigation needed.")
    else:
        print("\n❌ WEAK EVIDENCE: Phase transition may be an artifact")
        print("   The 'inner vs outer' explanation cannot be ruled out.")

if __name__ == "__main__":
    main()
