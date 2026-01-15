"""
UPGRADED PHASE TRANSITION TEST
==============================
Addressing Critique: "Tests 3-4 measured 'outer > inner' not breakpoint detection.
Need to use ΔAIC for random controls and sweeps, plus compare vs quadratic."

Upgraded Tests:
1. BREAKPOINT MODEL COMPARISON - Piecewise vs Linear at x=1.0 (unchanged)
2. LOCAL JUMP TEST - Narrow band around threshold (unchanged)  
3. RANDOM BREAKPOINT ΔAIC CONTROL - Is x=1.0 special for ΔAIC, not mean jump?
4. ΔAIC BREAKPOINT SWEEP - Where does ΔAIC peak? Distribution of best-fit breakpoints
5. PIECEWISE vs QUADRATIC - Both have 3 params, does kink beat smooth curvature?
6. SIGN-BASED ROBUSTNESS - Fraction positive + bootstrap CI (unchanged)

This directly tests: "Is x=1.0 a special breakpoint location for a kink model?"
"""

import os
import numpy as np
import scipy.stats as stats
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
# MODEL FITTING UTILITIES
# =============================================================================
def fit_linear(x, y):
    """Fit y = a + b*x, return SS_res and k=2."""
    if len(x) < 3: return np.inf, 2
    A = np.column_stack([np.ones_like(x), x])
    try:
        params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        y_pred = A @ params
        ss_res = np.sum((y - y_pred)**2)
        return ss_res, 2
    except:
        return np.inf, 2

def fit_piecewise_linear(x, y, breakpoint=1.0):
    """Fit piecewise linear with break at breakpoint (continuous), k=3."""
    if len(x) < 5: return np.inf, 3
    
    x_below = np.minimum(x, breakpoint)
    x_above = np.maximum(x - breakpoint, 0)
    
    A = np.column_stack([np.ones_like(x), x_below, x_above])
    try:
        params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        y_pred = A @ params
        ss_res = np.sum((y - y_pred)**2)
        return ss_res, 3
    except:
        return np.inf, 3

def fit_quadratic(x, y):
    """Fit y = a + b*x + c*x^2, return SS_res and k=3."""
    if len(x) < 4: return np.inf, 3
    A = np.column_stack([np.ones_like(x), x, x**2])
    try:
        params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        y_pred = A @ params
        ss_res = np.sum((y - y_pred)**2)
        return ss_res, 3
    except:
        return np.inf, 3

def compute_aic(n, k, ss_res):
    """Compute AIC from residual sum of squares."""
    if ss_res <= 0 or n <= k: return np.inf
    log_lik = -n/2 * (np.log(2*np.pi) + np.log(ss_res/n) + 1)
    aic = 2*k - 2*log_lik
    return aic

def compute_delta_aic_at_breakpoint(x, y, breakpoint):
    """Compute ΔAIC = AIC_linear - AIC_piecewise at given breakpoint."""
    n = len(x)
    ss_linear, k_lin = fit_linear(x, y)
    ss_piecewise, k_pw = fit_piecewise_linear(x, y, breakpoint)
    
    aic_linear = compute_aic(n, k_lin, ss_linear)
    aic_piecewise = compute_aic(n, k_pw, ss_piecewise)
    
    return aic_linear - aic_piecewise  # Positive = piecewise better

# =============================================================================
# TEST 1: BREAKPOINT MODEL COMPARISON (Linear vs Piecewise at x=1.0)
# =============================================================================
def test_breakpoint_model(galaxies_data):
    """Compare smooth linear vs piecewise linear at x=1.0."""
    print("\n" + "="*60)
    print("TEST 1: BREAKPOINT MODEL COMPARISON (Linear vs Piecewise @ x=1.0)")
    print("="*60)
    
    delta_aic_list = []
    
    for name, (n_rad, n_vbar, n_vobs) in galaxies_data.items():
        x = n_vbar + n_rad
        y = n_vobs
        
        if len(x) < 6: continue
        
        delta_aic = compute_delta_aic_at_breakpoint(x, y, 1.0)
        if not np.isinf(delta_aic):
            delta_aic_list.append(delta_aic)
    
    delta_aic = np.array(delta_aic_list)
    
    piecewise_wins = np.sum(delta_aic > 0)
    total = len(delta_aic)
    
    print(f"\nGalaxies analyzed: {total}")
    print(f"Piecewise model wins (AIC): {piecewise_wins}/{total} ({100*piecewise_wins/total:.1f}%)")
    print(f"Mean ΔAIC: {np.mean(delta_aic):.2f}")
    print(f"Median ΔAIC: {np.median(delta_aic):.2f}")
    
    if len(delta_aic) > 10:
        _, p_val = stats.wilcoxon(delta_aic)
        print(f"Wilcoxon p-value (ΔAIC ≠ 0): {p_val:.4e}")
    
    return delta_aic

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
    
    for name, (n_rad, n_vbar, n_vobs) in galaxies_data.items():
        x = n_vbar + n_rad
        
        mask_below = (x >= 1.0 - epsilon) & (x < 1.0)
        mask_above = (x >= 1.0) & (x <= 1.0 + epsilon)
        
        if np.sum(mask_below) >= 1 and np.sum(mask_above) >= 1:
            mean_below = np.mean(n_vobs[mask_below])
            mean_above = np.mean(n_vobs[mask_above])
            local_jumps.append(mean_above - mean_below)
    
    local_jumps = np.array(local_jumps)
    
    print(f"\nGalaxies with data in both bands: {len(local_jumps)}")
    print(f"Mean local jump: {np.mean(local_jumps):+.4f}")
    print(f"Median local jump: {np.median(local_jumps):+.4f}")
    print(f"Fraction positive: {np.mean(local_jumps > 0):.1%}")
    
    if len(local_jumps) > 10:
        _, p_val = stats.wilcoxon(local_jumps)
        print(f"Wilcoxon p-value: {p_val:.4e}")
    
    return local_jumps

# =============================================================================
# TEST 3 (UPGRADED): RANDOM BREAKPOINT ΔAIC CONTROL
# =============================================================================
def test_random_breakpoint_aic(galaxies_data, n_random=50):
    """
    For each galaxy, compute ΔAIC at x=1.0 vs random breakpoints.
    Ask: Is ΔAIC at 1.0 unusually large relative to galaxy's null distribution?
    """
    print("\n" + "="*60)
    print("TEST 3 (UPGRADED): RANDOM BREAKPOINT ΔAIC CONTROL")
    print("="*60)
    print("Testing: Is x=1.0 a special breakpoint location for ΔAIC?")
    
    rank_percentiles = []  # What percentile is x=1.0 within each galaxy?
    
    for name, (n_rad, n_vbar, n_vobs) in galaxies_data.items():
        x = n_vbar + n_rad
        y = n_vobs
        
        if len(x) < 8: continue
        
        # ΔAIC at x=1.0
        delta_aic_1 = compute_delta_aic_at_breakpoint(x, y, 1.0)
        if np.isinf(delta_aic_1): continue
        
        # ΔAIC at random breakpoints within data range
        x_min, x_max = np.percentile(x, [10, 90])
        random_breakpoints = np.random.uniform(x_min, x_max, n_random)
        
        random_delta_aics = []
        for bp in random_breakpoints:
            d_aic = compute_delta_aic_at_breakpoint(x, y, bp)
            if not np.isinf(d_aic):
                random_delta_aics.append(d_aic)
        
        if len(random_delta_aics) < 10: continue
        
        # What percentile is x=1.0 among random breakpoints?
        percentile = np.mean(np.array(random_delta_aics) <= delta_aic_1) * 100
        rank_percentiles.append(percentile)
    
    rank_percentiles = np.array(rank_percentiles)
    
    print(f"\nGalaxies analyzed: {len(rank_percentiles)}")
    print(f"Mean percentile of x=1.0: {np.mean(rank_percentiles):.1f}%")
    print(f"Median percentile of x=1.0: {np.median(rank_percentiles):.1f}%")
    
    # How many galaxies have x=1.0 in top 20%?
    top_20 = np.sum(rank_percentiles >= 80)
    top_10 = np.sum(rank_percentiles >= 90)
    print(f"Galaxies where x=1.0 is in top 20%: {top_20}/{len(rank_percentiles)} ({100*top_20/len(rank_percentiles):.1f}%)")
    print(f"Galaxies where x=1.0 is in top 10%: {top_10}/{len(rank_percentiles)} ({100*top_10/len(rank_percentiles):.1f}%)")
    
    # Test if mean percentile > 50 (expected under null)
    _, p_val = stats.wilcoxon(rank_percentiles - 50)
    print(f"Wilcoxon p-value (percentile > 50%): {p_val:.4e}")
    
    if np.mean(rank_percentiles) > 60:
        print("✅ x=1.0 is a better-than-average breakpoint location")
    else:
        print("⚠️ x=1.0 is not significantly special among breakpoints")
    
    return rank_percentiles

# =============================================================================
# TEST 4 (UPGRADED): ΔAIC BREAKPOINT SWEEP - Distribution of Best-Fit Breakpoints
# =============================================================================
def test_aic_breakpoint_sweep(galaxies_data):
    """
    For each galaxy, sweep breakpoints and find argmax(ΔAIC).
    Plot distribution of best-fit breakpoints. Do they cluster near 1.0?
    """
    print("\n" + "="*60)
    print("TEST 4 (UPGRADED): ΔAIC BREAKPOINT SWEEP")
    print("="*60)
    print("Finding best-fit breakpoint per galaxy via ΔAIC maximization")
    
    breakpoints_to_test = np.linspace(0.5, 1.5, 41)
    best_breakpoints = []
    delta_aic_at_best = []
    delta_aic_at_1 = []
    
    for name, (n_rad, n_vbar, n_vobs) in galaxies_data.items():
        x = n_vbar + n_rad
        y = n_vobs
        
        if len(x) < 8: continue
        
        # Sweep breakpoints
        delta_aics = []
        for bp in breakpoints_to_test:
            d_aic = compute_delta_aic_at_breakpoint(x, y, bp)
            delta_aics.append(d_aic if not np.isinf(d_aic) else -np.inf)
        
        delta_aics = np.array(delta_aics)
        if np.all(np.isinf(delta_aics)): continue
        
        # Best breakpoint for this galaxy
        best_idx = np.argmax(delta_aics)
        best_bp = breakpoints_to_test[best_idx]
        best_breakpoints.append(best_bp)
        delta_aic_at_best.append(delta_aics[best_idx])
        
        # ΔAIC at x=1.0
        idx_1 = np.argmin(np.abs(breakpoints_to_test - 1.0))
        delta_aic_at_1.append(delta_aics[idx_1])
    
    best_breakpoints = np.array(best_breakpoints)
    delta_aic_at_best = np.array(delta_aic_at_best)
    delta_aic_at_1 = np.array(delta_aic_at_1)
    
    print(f"\nGalaxies analyzed: {len(best_breakpoints)}")
    print(f"\nDistribution of best-fit breakpoints:")
    print(f"  Mean: {np.mean(best_breakpoints):.3f}")
    print(f"  Median: {np.median(best_breakpoints):.3f}")
    print(f"  Std Dev: {np.std(best_breakpoints):.3f}")
    
    # How many cluster near 1.0?
    near_1 = np.sum(np.abs(best_breakpoints - 1.0) <= 0.15)
    print(f"\nBest breakpoints within ±0.15 of 1.0: {near_1}/{len(best_breakpoints)} ({100*near_1/len(best_breakpoints):.1f}%)")
    
    # One-sample test: is mean breakpoint = 1.0?
    _, p_val = stats.ttest_1samp(best_breakpoints, 1.0)
    print(f"T-test p-value (mean = 1.0): {p_val:.4e}")
    
    # Mean ΔAIC at best vs at 1.0
    print(f"\nMean ΔAIC at best breakpoint: {np.mean(delta_aic_at_best):.2f}")
    print(f"Mean ΔAIC at x=1.0: {np.mean(delta_aic_at_1):.2f}")
    print(f"Efficiency of x=1.0: {100*np.mean(delta_aic_at_1)/np.mean(delta_aic_at_best):.1f}% of optimal")
    
    if np.abs(np.mean(best_breakpoints) - 1.0) <= 0.1:
        print("✅ Best breakpoints cluster near x=1.0")
    elif np.abs(np.mean(best_breakpoints) - 1.0) <= 0.2:
        print("⚠️ Best breakpoints near but not exactly at x=1.0")
    else:
        print("❌ Best breakpoints do not cluster at x=1.0")
    
    return best_breakpoints, breakpoints_to_test, delta_aic_at_best, delta_aic_at_1

# =============================================================================
# TEST 5: PIECEWISE vs QUADRATIC (Both 3 Parameters)
# =============================================================================
def test_piecewise_vs_quadratic(galaxies_data):
    """
    Compare piecewise-linear (3 params) vs quadratic (3 params).
    If piecewise beats quadratic, that's strong evidence for kink vs curvature.
    """
    print("\n" + "="*60)
    print("TEST 5: PIECEWISE LINEAR vs QUADRATIC (Both 3 Parameters)")
    print("="*60)
    print("Testing: Does a KINK beat smooth CURVATURE?")
    
    delta_aic_list = []  # AIC_quadratic - AIC_piecewise (positive = piecewise better)
    
    for name, (n_rad, n_vbar, n_vobs) in galaxies_data.items():
        x = n_vbar + n_rad
        y = n_vobs
        n = len(x)
        
        if n < 6: continue
        
        ss_piecewise, k_pw = fit_piecewise_linear(x, y, 1.0)
        ss_quadratic, k_quad = fit_quadratic(x, y)
        
        if np.isinf(ss_piecewise) or np.isinf(ss_quadratic): continue
        
        aic_piecewise = compute_aic(n, k_pw, ss_piecewise)
        aic_quadratic = compute_aic(n, k_quad, ss_quadratic)
        
        delta_aic = aic_quadratic - aic_piecewise  # Positive = piecewise better
        delta_aic_list.append(delta_aic)
    
    delta_aic = np.array(delta_aic_list)
    
    piecewise_wins = np.sum(delta_aic > 0)
    total = len(delta_aic)
    
    print(f"\nGalaxies analyzed: {total}")
    print(f"Piecewise beats Quadratic: {piecewise_wins}/{total} ({100*piecewise_wins/total:.1f}%)")
    print(f"Mean ΔAIC (Quad - Piecewise): {np.mean(delta_aic):.2f}")
    print(f"Median ΔAIC: {np.median(delta_aic):.2f}")
    
    if len(delta_aic) > 10:
        _, p_val = stats.wilcoxon(delta_aic)
        print(f"Wilcoxon p-value: {p_val:.4e}")
    
    if piecewise_wins/total > 0.5 and np.median(delta_aic) > 0:
        print("✅ Piecewise (kink) beats Quadratic (curvature) - evidence for discontinuity")
    else:
        print("⚠️ Quadratic (curvature) fits as well as Piecewise (kink)")
    
    return delta_aic

# =============================================================================
# TEST 6: SIGN-BASED ROBUSTNESS + BOOTSTRAP CI
# =============================================================================
def test_sign_robustness(galaxies_data, n_bootstrap=10000):
    """Report sign-based statistics and bootstrap confidence intervals."""
    print("\n" + "="*60)
    print("TEST 6: SIGN-BASED ROBUSTNESS + BOOTSTRAP CI")
    print("="*60)
    
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
    
    n_positive = np.sum(jumps > 0)
    print(f"\nTotal galaxies: {n}")
    print(f"Positive jumps: {n_positive} ({100*n_positive/n:.1f}%)")
    
    # Bootstrap CI
    bootstrap_medians = [np.median(np.random.choice(jumps, n, replace=True)) for _ in range(n_bootstrap)]
    median_ci = np.percentile(bootstrap_medians, [2.5, 97.5])
    
    print(f"Median jump: {np.median(jumps):.4f}")
    print(f"95% Bootstrap CI: [{median_ci[0]:.4f}, {median_ci[1]:.4f}]")
    
    if median_ci[0] > 0:
        print("✅ 95% CI excludes zero")
    
    return jumps, median_ci

# =============================================================================
# VISUALIZATION
# =============================================================================
def create_visualizations(delta_aic_1, local_jumps, rank_percentiles, 
                          best_breakpoints, delta_aic_pw_quad, jumps):
    """Create comprehensive visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Test 1: ΔAIC at x=1.0
    ax = axes[0, 0]
    ax.hist(delta_aic_1, bins=25, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=np.median(delta_aic_1), color='green', linestyle='-', linewidth=2)
    ax.set_xlabel('ΔAIC (Linear - Piecewise @ x=1.0)')
    ax.set_ylabel('Count')
    ax.set_title('Test 1: Piecewise vs Linear at x=1.0')
    
    # 2. Test 2: Local jumps
    ax = axes[0, 1]
    ax.hist(local_jumps, bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Local Jump (±0.15 band)')
    ax.set_ylabel('Count')
    ax.set_title(f'Test 2: Local Jump ({np.mean(local_jumps>0):.0%} positive)')
    
    # 3. Test 3: Rank percentiles of x=1.0
    ax = axes[0, 2]
    ax.hist(rank_percentiles, bins=20, color='orange', alpha=0.7, edgecolor='black')
    ax.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
    ax.axvline(x=np.mean(rank_percentiles), color='green', linestyle='-', linewidth=2, 
               label=f'Mean={np.mean(rank_percentiles):.0f}%')
    ax.set_xlabel('Percentile Rank of x=1.0')
    ax.set_ylabel('Count')
    ax.set_title('Test 3: x=1.0 Rank Among Random Breakpoints')
    ax.legend()
    
    # 4. Test 4: Distribution of best breakpoints
    ax = axes[1, 0]
    ax.hist(best_breakpoints, bins=20, color='teal', alpha=0.7, edgecolor='black')
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='x=1.0')
    ax.axvline(x=np.mean(best_breakpoints), color='green', linestyle='-', linewidth=2,
               label=f'Mean={np.mean(best_breakpoints):.2f}')
    ax.set_xlabel('Best-Fit Breakpoint')
    ax.set_ylabel('Count')
    ax.set_title('Test 4: Distribution of Optimal Breakpoints')
    ax.legend()
    
    # 5. Test 5: Piecewise vs Quadratic
    ax = axes[1, 1]
    ax.hist(delta_aic_pw_quad, bins=25, color='coral', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=np.median(delta_aic_pw_quad), color='green', linestyle='-', linewidth=2)
    ax.set_xlabel('ΔAIC (Quadratic - Piecewise)')
    ax.set_ylabel('Count')
    ax.set_title(f'Test 5: Piecewise vs Quadratic ({np.mean(delta_aic_pw_quad>0):.0%} piecewise wins)')
    
    # 6. Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate pass/fail
    t1_pass = np.mean(delta_aic_1 > 0) > 0.5
    t2_pass = np.mean(local_jumps > 0) > 0.8
    t3_pass = np.mean(rank_percentiles) > 60
    t4_pass = np.abs(np.mean(best_breakpoints) - 1.0) <= 0.15
    t5_pass = np.mean(delta_aic_pw_quad > 0) > 0.5
    t6_pass = np.mean(jumps > 0) > 0.9
    
    tests_passed = sum([t1_pass, t2_pass, t3_pass, t4_pass, t5_pass, t6_pass])
    
    summary = f"""
    UPGRADED PHASE TRANSITION ANALYSIS
    ===================================
    
    Test 1 (Piecewise vs Linear @ 1.0):
      {'✅' if t1_pass else '❌'} Piecewise wins {np.mean(delta_aic_1>0):.0%}
    
    Test 2 (Local Jump ±0.15):
      {'✅' if t2_pass else '❌'} {np.mean(local_jumps>0):.0%} positive
    
    Test 3 (ΔAIC Rank of x=1.0):
      {'✅' if t3_pass else '❌'} Mean rank: {np.mean(rank_percentiles):.0f}%
    
    Test 4 (Best Breakpoint Location):
      {'✅' if t4_pass else '❌'} Mean: {np.mean(best_breakpoints):.2f}
    
    Test 5 (Piecewise vs Quadratic):
      {'✅' if t5_pass else '❌'} Piecewise wins {np.mean(delta_aic_pw_quad>0):.0%}
    
    Test 6 (Sign Robustness):
      {'✅' if t6_pass else '❌'} {np.mean(jumps>0):.0%} positive
    
    ===================================
    TESTS PASSED: {tests_passed}/6
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('phase_transition_upgraded.png', dpi=150)
    print("\n[SAVED] phase_transition_upgraded.png")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*60)
    print("UPGRADED PHASE TRANSITION ANALYSIS")
    print("Testing: Is x=1.0 a SPECIAL breakpoint location?")
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
    delta_aic_1 = test_breakpoint_model(galaxies_data)
    local_jumps = test_local_jump(galaxies_data, epsilon=0.15)
    rank_percentiles = test_random_breakpoint_aic(galaxies_data, n_random=50)
    best_breakpoints, bp_range, delta_best, delta_1 = test_aic_breakpoint_sweep(galaxies_data)
    delta_aic_pw_quad = test_piecewise_vs_quadratic(galaxies_data)
    jumps, median_ci = test_sign_robustness(galaxies_data)
    
    # Create visualizations
    create_visualizations(delta_aic_1, local_jumps, rank_percentiles,
                          best_breakpoints, delta_aic_pw_quad, jumps)
    
    # Final Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    tests_passed = 0
    
    # Test 1
    t1_rate = np.mean(delta_aic_1 > 0)
    if t1_rate > 0.5:
        print(f"✅ Test 1 PASSED: Piecewise @ x=1.0 wins {t1_rate:.0%}")
        tests_passed += 1
    else:
        print(f"❌ Test 1 FAILED: Piecewise wins only {t1_rate:.0%}")
    
    # Test 2
    t2_rate = np.mean(local_jumps > 0)
    if t2_rate > 0.8 and np.median(local_jumps) > 0:
        print(f"✅ Test 2 PASSED: Local jump {t2_rate:.0%} positive")
        tests_passed += 1
    else:
        print(f"❌ Test 2 FAILED: Local jump only {t2_rate:.0%} positive")
    
    # Test 3 (UPGRADED)
    t3_mean = np.mean(rank_percentiles)
    if t3_mean > 60:
        print(f"✅ Test 3 PASSED: x=1.0 ranks at {t3_mean:.0f}th percentile (above random)")
        tests_passed += 1
    else:
        print(f"❌ Test 3 FAILED: x=1.0 ranks at {t3_mean:.0f}th percentile (not special)")
    
    # Test 4 (UPGRADED)
    t4_mean = np.mean(best_breakpoints)
    if np.abs(t4_mean - 1.0) <= 0.15:
        print(f"✅ Test 4 PASSED: Best breakpoints cluster at {t4_mean:.2f} (near 1.0)")
        tests_passed += 1
    elif np.abs(t4_mean - 1.0) <= 0.25:
        print(f"⚠️ Test 4 PARTIAL: Best breakpoints at {t4_mean:.2f} (close to 1.0)")
        tests_passed += 0.5
    else:
        print(f"❌ Test 4 FAILED: Best breakpoints at {t4_mean:.2f} (far from 1.0)")
    
    # Test 5
    t5_rate = np.mean(delta_aic_pw_quad > 0)
    if t5_rate > 0.5 and np.median(delta_aic_pw_quad) > 0:
        print(f"✅ Test 5 PASSED: Piecewise beats Quadratic {t5_rate:.0%} (kink > curvature)")
        tests_passed += 1
    else:
        print(f"❌ Test 5 FAILED: Piecewise wins only {t5_rate:.0%}")
    
    # Test 6
    t6_rate = np.mean(jumps > 0)
    if t6_rate > 0.9 and median_ci[0] > 0:
        print(f"✅ Test 6 PASSED: {t6_rate:.0%} positive, CI excludes zero")
        tests_passed += 1
    else:
        print(f"⚠️ Test 6 PARTIAL: {t6_rate:.0%} positive")
        if median_ci[0] > 0:
            tests_passed += 0.5
    
    print(f"\n{'='*60}")
    print(f"TESTS PASSED: {tests_passed}/6")
    
    if tests_passed >= 5:
        print("\n✅✅ STRONG EVIDENCE: x=1.0 is a special breakpoint location")
        print("   The phase transition is a REAL discontinuity at this threshold.")
    elif tests_passed >= 4:
        print("\n✅ GOOD EVIDENCE: Breakpoint exists, x=1.0 is reasonably special")
        print("   Some refinement of threshold location may improve the model.")
    elif tests_passed >= 3:
        print("\n⚠️ MODERATE EVIDENCE: Breakpoint exists but x=1.0 may not be optimal")
        print("   The effect is real but threshold needs refinement.")
    else:
        print("\n❌ WEAK EVIDENCE: Cannot confirm x=1.0 as a special location")

if __name__ == "__main__":
    main()
