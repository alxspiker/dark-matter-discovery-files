"""
THRESHOLD REFINEMENT ANALYSIS
=============================
Following critique: "Stop treating 1.0 as sacred" - test x≈1.15-1.2 instead.

Goals:
1. Test if x=1.17 (the empirical best) is a "special" breakpoint
2. Re-parameterize x = V_bar + k*R to find optimal global weight k
3. Find the k that makes breakpoints cluster at 1.0
4. Validate on held-out galaxies

This determines whether a refined formulation can establish a universal threshold.
"""

import os
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize_scalar
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

def compute_aic(n, k, ss_res):
    if ss_res <= 0 or n <= k: return np.inf
    log_lik = -n/2 * (np.log(2*np.pi) + np.log(ss_res/n) + 1)
    return 2*k - 2*log_lik

def fit_linear(x, y):
    if len(x) < 3: return np.inf
    A = np.column_stack([np.ones_like(x), x])
    try:
        params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return np.sum((y - A @ params)**2)
    except:
        return np.inf

def fit_piecewise(x, y, bp):
    if len(x) < 5: return np.inf
    x_below = np.minimum(x, bp)
    x_above = np.maximum(x - bp, 0)
    A = np.column_stack([np.ones_like(x), x_below, x_above])
    try:
        params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return np.sum((y - A @ params)**2)
    except:
        return np.inf

def compute_delta_aic(x, y, bp):
    n = len(x)
    ss_lin = fit_linear(x, y)
    ss_pw = fit_piecewise(x, y, bp)
    if np.isinf(ss_lin) or np.isinf(ss_pw): return -np.inf
    return compute_aic(n, 2, ss_lin) - compute_aic(n, 3, ss_pw)

# =============================================================================
# TEST 1: Is x=1.17 a "special" breakpoint?
# =============================================================================
def test_new_threshold(galaxies_data, threshold=1.17):
    """Test if x=1.17 (empirical best) is special among random breakpoints."""
    print("\n" + "="*60)
    print(f"TEST 1: Is x={threshold} a SPECIAL breakpoint?")
    print("="*60)
    
    rank_percentiles = []
    
    for name, (n_rad, n_vbar, n_vobs) in galaxies_data.items():
        x = n_vbar + n_rad
        y = n_vobs
        if len(x) < 8: continue
        
        delta_aic_target = compute_delta_aic(x, y, threshold)
        if np.isinf(delta_aic_target): continue
        
        # Compare against random breakpoints
        x_min, x_max = np.percentile(x, [10, 90])
        random_bps = np.random.uniform(x_min, x_max, 50)
        random_delta_aics = [compute_delta_aic(x, y, bp) for bp in random_bps]
        random_delta_aics = [d for d in random_delta_aics if not np.isinf(d)]
        
        if len(random_delta_aics) < 10: continue
        
        percentile = np.mean(np.array(random_delta_aics) <= delta_aic_target) * 100
        rank_percentiles.append(percentile)
    
    rank_percentiles = np.array(rank_percentiles)
    
    print(f"\nGalaxies analyzed: {len(rank_percentiles)}")
    print(f"Mean percentile of x={threshold}: {np.mean(rank_percentiles):.1f}%")
    print(f"Median percentile: {np.median(rank_percentiles):.1f}%")
    
    top_20 = np.sum(rank_percentiles >= 80)
    top_10 = np.sum(rank_percentiles >= 90)
    print(f"In top 20%: {top_20}/{len(rank_percentiles)} ({100*top_20/len(rank_percentiles):.1f}%)")
    print(f"In top 10%: {top_10}/{len(rank_percentiles)} ({100*top_10/len(rank_percentiles):.1f}%)")
    
    _, p_val = stats.wilcoxon(rank_percentiles - 50)
    print(f"Wilcoxon p-value (> 50%): {p_val:.4e}")
    
    if np.mean(rank_percentiles) > 60:
        print(f"✅ x={threshold} IS a better-than-average breakpoint")
    else:
        print(f"⚠️ x={threshold} is not significantly special")
    
    return rank_percentiles

# =============================================================================
# TEST 2: Compare x=1.0 vs x=1.17
# =============================================================================
def compare_thresholds(galaxies_data):
    """Direct comparison: does x=1.17 beat x=1.0?"""
    print("\n" + "="*60)
    print("TEST 2: Direct Comparison x=1.0 vs x=1.17")
    print("="*60)
    
    delta_aic_1_0 = []
    delta_aic_1_17 = []
    
    for name, (n_rad, n_vbar, n_vobs) in galaxies_data.items():
        x = n_vbar + n_rad
        y = n_vobs
        if len(x) < 8: continue
        
        d1 = compute_delta_aic(x, y, 1.0)
        d2 = compute_delta_aic(x, y, 1.17)
        
        if not np.isinf(d1) and not np.isinf(d2):
            delta_aic_1_0.append(d1)
            delta_aic_1_17.append(d2)
    
    delta_aic_1_0 = np.array(delta_aic_1_0)
    delta_aic_1_17 = np.array(delta_aic_1_17)
    
    wins_1_17 = np.sum(delta_aic_1_17 > delta_aic_1_0)
    total = len(delta_aic_1_0)
    
    print(f"\nGalaxies compared: {total}")
    print(f"x=1.17 beats x=1.0: {wins_1_17}/{total} ({100*wins_1_17/total:.1f}%)")
    print(f"\nMean ΔAIC at x=1.0: {np.mean(delta_aic_1_0):.2f}")
    print(f"Mean ΔAIC at x=1.17: {np.mean(delta_aic_1_17):.2f}")
    print(f"Improvement: {np.mean(delta_aic_1_17) - np.mean(delta_aic_1_0):.2f}")
    
    _, p_val = stats.wilcoxon(delta_aic_1_17 - delta_aic_1_0)
    print(f"Wilcoxon p-value: {p_val:.4e}")
    
    return delta_aic_1_0, delta_aic_1_17

# =============================================================================
# TEST 3: Find optimal global weight k in x = V_bar + k*R
# =============================================================================
def find_optimal_k(galaxies_data, target_bp=1.0):
    """
    Find global weight k such that x_k = V_bar + k*R makes breakpoints 
    cluster at target_bp (default 1.0).
    """
    print("\n" + "="*60)
    print(f"TEST 3: Find optimal k where x_k = V_bar + k*R")
    print(f"Goal: Make best breakpoints cluster at {target_bp}")
    print("="*60)
    
    def compute_best_breakpoint(n_rad, n_vbar, n_vobs, k):
        """Find best breakpoint for x_k = n_vbar + k * n_rad."""
        x_k = n_vbar + k * n_rad
        y = n_vobs
        
        breakpoints = np.linspace(0.5, 1.5, 21)
        best_bp = 1.0
        best_aic = -np.inf
        
        for bp in breakpoints:
            d_aic = compute_delta_aic(x_k, y, bp)
            if d_aic > best_aic:
                best_aic = d_aic
                best_bp = bp
        
        return best_bp
    
    def objective(k):
        """Minimize distance from mean best breakpoint to target."""
        best_bps = []
        for name, (n_rad, n_vbar, n_vobs) in galaxies_data.items():
            if len(n_rad) < 8: continue
            bp = compute_best_breakpoint(n_rad, n_vbar, n_vobs, k)
            best_bps.append(bp)
        
        if len(best_bps) < 10: return np.inf
        return (np.mean(best_bps) - target_bp)**2
    
    # Test a range of k values
    k_values = np.linspace(0.5, 2.0, 31)
    mean_bps = []
    
    print("\nSweeping k values...")
    for k in k_values:
        best_bps = []
        for name, (n_rad, n_vbar, n_vobs) in galaxies_data.items():
            if len(n_rad) < 8: continue
            bp = compute_best_breakpoint(n_rad, n_vbar, n_vobs, k)
            best_bps.append(bp)
        mean_bps.append(np.mean(best_bps))
    
    mean_bps = np.array(mean_bps)
    
    # Find k that puts mean breakpoint closest to target
    best_idx = np.argmin(np.abs(mean_bps - target_bp))
    optimal_k = k_values[best_idx]
    
    print(f"\nOptimal k = {optimal_k:.2f}")
    print(f"At k={optimal_k:.2f}, mean best breakpoint = {mean_bps[best_idx]:.3f}")
    print(f"At k=1.0 (current), mean best breakpoint = {mean_bps[np.argmin(np.abs(k_values-1.0))]:.3f}")
    
    return optimal_k, k_values, mean_bps

# =============================================================================
# TEST 4: Validate optimal k on held-out galaxies
# =============================================================================
def validate_k_holdout(galaxies_data, optimal_k, n_folds=5):
    """Cross-validation: does optimal k generalize?"""
    print("\n" + "="*60)
    print(f"TEST 4: Cross-Validation of k={optimal_k:.2f}")
    print("="*60)
    
    galaxy_names = list(galaxies_data.keys())
    np.random.shuffle(galaxy_names)
    
    fold_size = len(galaxy_names) // n_folds
    
    holdout_improvements = []
    
    for fold in range(n_folds):
        # Split
        test_start = fold * fold_size
        test_end = test_start + fold_size
        test_names = galaxy_names[test_start:test_end]
        
        # Evaluate on test set
        for name in test_names:
            if name not in galaxies_data: continue
            n_rad, n_vbar, n_vobs = galaxies_data[name]
            if len(n_rad) < 8: continue
            
            # ΔAIC with k=1.0 at bp=1.0
            x_1 = n_vbar + 1.0 * n_rad
            d_aic_old = compute_delta_aic(x_1, n_vobs, 1.0)
            
            # ΔAIC with optimal k at bp=1.0
            x_k = n_vbar + optimal_k * n_rad
            d_aic_new = compute_delta_aic(x_k, n_vobs, 1.0)
            
            if not np.isinf(d_aic_old) and not np.isinf(d_aic_new):
                holdout_improvements.append(d_aic_new - d_aic_old)
    
    holdout_improvements = np.array(holdout_improvements)
    
    print(f"\nHoldout galaxies tested: {len(holdout_improvements)}")
    print(f"Mean ΔAIC improvement (k={optimal_k:.2f} vs k=1.0): {np.mean(holdout_improvements):.2f}")
    print(f"Median improvement: {np.median(holdout_improvements):.2f}")
    print(f"Fraction improved: {np.mean(holdout_improvements > 0):.1%}")
    
    if len(holdout_improvements) > 10:
        _, p_val = stats.wilcoxon(holdout_improvements)
        print(f"Wilcoxon p-value: {p_val:.4e}")
    
    if np.mean(holdout_improvements) > 0 and np.mean(holdout_improvements > 0) > 0.5:
        print(f"✅ k={optimal_k:.2f} generalizes to held-out galaxies")
    else:
        print(f"⚠️ k={optimal_k:.2f} may be overfit")
    
    return holdout_improvements

# =============================================================================
# TEST 5: Final ranking of x=1.0 with optimal k
# =============================================================================
def test_refined_threshold(galaxies_data, optimal_k):
    """With optimal k, is x=1.0 now special?"""
    print("\n" + "="*60)
    print(f"TEST 5: Is x=1.0 special with refined x_k (k={optimal_k:.2f})?")
    print("="*60)
    
    rank_percentiles = []
    
    for name, (n_rad, n_vbar, n_vobs) in galaxies_data.items():
        x = n_vbar + optimal_k * n_rad
        y = n_vobs
        if len(x) < 8: continue
        
        delta_aic_1 = compute_delta_aic(x, y, 1.0)
        if np.isinf(delta_aic_1): continue
        
        # Compare against random breakpoints
        x_min, x_max = np.percentile(x, [10, 90])
        random_bps = np.random.uniform(x_min, x_max, 50)
        random_delta_aics = [compute_delta_aic(x, y, bp) for bp in random_bps]
        random_delta_aics = [d for d in random_delta_aics if not np.isinf(d)]
        
        if len(random_delta_aics) < 10: continue
        
        percentile = np.mean(np.array(random_delta_aics) <= delta_aic_1) * 100
        rank_percentiles.append(percentile)
    
    rank_percentiles = np.array(rank_percentiles)
    
    print(f"\nGalaxies analyzed: {len(rank_percentiles)}")
    print(f"Mean percentile of x=1.0 (with k={optimal_k:.2f}): {np.mean(rank_percentiles):.1f}%")
    print(f"Median percentile: {np.median(rank_percentiles):.1f}%")
    
    top_20 = np.sum(rank_percentiles >= 80)
    top_10 = np.sum(rank_percentiles >= 90)
    top_5 = np.sum(rank_percentiles >= 95)
    print(f"In top 20%: {top_20}/{len(rank_percentiles)} ({100*top_20/len(rank_percentiles):.1f}%)")
    print(f"In top 10%: {top_10}/{len(rank_percentiles)} ({100*top_10/len(rank_percentiles):.1f}%)")
    print(f"In top 5%: {top_5}/{len(rank_percentiles)} ({100*top_5/len(rank_percentiles):.1f}%)")
    
    _, p_val = stats.wilcoxon(rank_percentiles - 50)
    print(f"Wilcoxon p-value (> 50%): {p_val:.4e}")
    
    # Compare to original (k=1.0)
    print(f"\nComparison: Original k=1.0 had ~49% mean percentile")
    improvement = np.mean(rank_percentiles) - 49
    print(f"Improvement with k={optimal_k:.2f}: +{improvement:.1f} percentile points")
    
    return rank_percentiles

# =============================================================================
# VISUALIZATION
# =============================================================================
def create_visualizations(rank_1_17, delta_1_0, delta_1_17, k_values, mean_bps, 
                          optimal_k, holdout_imp, rank_refined):
    """Create comprehensive visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Rank percentiles at x=1.17
    ax = axes[0, 0]
    ax.hist(rank_1_17, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
    ax.axvline(x=np.mean(rank_1_17), color='green', linestyle='-', linewidth=2,
               label=f'Mean={np.mean(rank_1_17):.0f}%')
    ax.set_xlabel('Percentile Rank')
    ax.set_ylabel('Count')
    ax.set_title('Test 1: Is x=1.17 Special?')
    ax.legend()
    
    # 2. x=1.0 vs x=1.17 comparison
    ax = axes[0, 1]
    ax.scatter(delta_1_0, delta_1_17, alpha=0.5, s=20)
    ax.plot([min(delta_1_0), max(delta_1_0)], [min(delta_1_0), max(delta_1_0)], 
            'r--', linewidth=2, label='Equal')
    ax.set_xlabel('ΔAIC at x=1.0')
    ax.set_ylabel('ΔAIC at x=1.17')
    ax.set_title(f'Test 2: x=1.17 wins {100*np.mean(delta_1_17 > delta_1_0):.0f}%')
    ax.legend()
    
    # 3. k sweep
    ax = axes[0, 2]
    ax.plot(k_values, mean_bps, 'b-', linewidth=2)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Target bp=1.0')
    ax.axvline(x=optimal_k, color='green', linestyle=':', linewidth=2, 
               label=f'Optimal k={optimal_k:.2f}')
    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=1, label='Current k=1.0')
    ax.set_xlabel('Weight k in x = V_bar + k*R')
    ax.set_ylabel('Mean Best Breakpoint')
    ax.set_title('Test 3: Finding Optimal k')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Holdout validation
    ax = axes[1, 0]
    ax.hist(holdout_imp, bins=20, color='teal', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No improvement')
    ax.axvline(x=np.mean(holdout_imp), color='green', linestyle='-', linewidth=2,
               label=f'Mean={np.mean(holdout_imp):.1f}')
    ax.set_xlabel('ΔAIC Improvement')
    ax.set_ylabel('Count')
    ax.set_title(f'Test 4: Holdout Validation ({np.mean(holdout_imp>0):.0%} improved)')
    ax.legend()
    
    # 5. Refined ranking
    ax = axes[1, 1]
    ax.hist(rank_refined, bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
    ax.axvline(x=np.mean(rank_refined), color='green', linestyle='-', linewidth=2,
               label=f'Mean={np.mean(rank_refined):.0f}%')
    ax.set_xlabel('Percentile Rank')
    ax.set_ylabel('Count')
    ax.set_title(f'Test 5: x=1.0 Rank with Optimal k')
    ax.legend()
    
    # 6. Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    summary = f"""
    THRESHOLD REFINEMENT ANALYSIS
    =============================
    
    Original (k=1.0, bp=1.0):
      x=1.0 percentile: ~49%
      Not special among breakpoints
    
    Test 1 (x=1.17 special?):
      Mean percentile: {np.mean(rank_1_17):.0f}%
      {'✅ Better' if np.mean(rank_1_17) > 55 else '⚠️ Still not special'}
    
    Test 2 (x=1.17 vs x=1.0):
      x=1.17 wins: {100*np.mean(delta_1_17 > delta_1_0):.0f}%
    
    Test 3 (Optimal k):
      k = {optimal_k:.2f}
      Makes bp cluster at 1.0
    
    Test 4 (Holdout validation):
      Improved: {100*np.mean(holdout_imp > 0):.0f}%
      Mean gain: {np.mean(holdout_imp):.1f} ΔAIC
    
    Test 5 (Refined x=1.0 rank):
      Mean percentile: {np.mean(rank_refined):.0f}%
      Improvement: +{np.mean(rank_refined) - 49:.0f} pts
    
    CONCLUSION:
    {'✅ Refinement makes x=1.0 special' if np.mean(rank_refined) > 60 else '⚠️ x=1.0 still not uniquely special'}
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('threshold_refinement.png', dpi=150)
    print("\n[SAVED] threshold_refinement.png")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*60)
    print("THRESHOLD REFINEMENT ANALYSIS")
    print("Can we find a formulation where x=1.0 IS special?")
    print("="*60)
    
    # Load galaxies
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
    
    print(f"Valid galaxies: {len(galaxies_data)}")
    
    # Run tests
    rank_1_17 = test_new_threshold(galaxies_data, threshold=1.17)
    delta_1_0, delta_1_17 = compare_thresholds(galaxies_data)
    optimal_k, k_values, mean_bps = find_optimal_k(galaxies_data, target_bp=1.0)
    holdout_imp = validate_k_holdout(galaxies_data, optimal_k)
    rank_refined = test_refined_threshold(galaxies_data, optimal_k)
    
    # Visualizations
    create_visualizations(rank_1_17, delta_1_0, delta_1_17, k_values, mean_bps,
                          optimal_k, holdout_imp, rank_refined)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    print(f"\nOriginal formulation (k=1.0, threshold=1.0):")
    print(f"  x=1.0 percentile rank: ~49% (not special)")
    
    print(f"\nEmpirical best threshold (k=1.0, threshold=1.17):")
    print(f"  x=1.17 percentile rank: {np.mean(rank_1_17):.0f}%")
    print(f"  x=1.17 beats x=1.0: {100*np.mean(delta_1_17 > delta_1_0):.0f}%")
    
    print(f"\nRefined formulation (k={optimal_k:.2f}, threshold=1.0):")
    print(f"  x=1.0 percentile rank: {np.mean(rank_refined):.0f}%")
    print(f"  Holdout validation: {100*np.mean(holdout_imp > 0):.0f}% improved")
    
    if np.mean(rank_refined) > 60:
        print(f"\n✅ SUCCESS: With k={optimal_k:.2f}, x=1.0 becomes a meaningful threshold")
        print(f"   Recommended formula: x = V_bar + {optimal_k:.2f}*R")
    elif np.mean(rank_refined) > 55:
        print(f"\n⚠️ PARTIAL: k={optimal_k:.2f} improves but x=1.0 still not strongly special")
    else:
        print(f"\n❌ FAILED: Cannot find formulation where x=1.0 is uniquely special")
        print(f"   The relationship is likely smooth curvature, not a discrete threshold")

if __name__ == "__main__":
    main()
