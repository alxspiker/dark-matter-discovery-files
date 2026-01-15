"""
ULTIMATE BENCHMARK (The Steel Man Test)
=======================================
Addressing Reviewer Critique: "Threshold Tuning Fairness."

Goal: Compare models when EVERY model is allowed to optimize its own
threshold parameter (t) on the training data.

Methodology per Galaxy:
1. Split 50/50 Train/Test.
2. For each Model (Radius, Mass, Overflow):
   - Sweep threshold 't' from 0.1 to 2.0 on TRAIN data.
   - Find the 't' that gives the lowest Training RMSE.
   - Learn V_flat at that optimal 't'.
3. Apply that optimal 't' and V_flat to TEST data.

This gives the baselines the best possible chance to win. 
If Overflow still wins, the signal is structural, not a parameter tuning artifact.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import re
import glob
from scipy import stats

SPARC_DIR = os.path.join(os.path.dirname(__file__), "sparc_galaxies")

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

def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted)**2))

def optimize_and_predict(input_feature, vbar, vobs, train_idx, test_idx):
    """
    1. Sweeps thresholds on TRAIN to find best t.
    2. Fits V_flat at best t.
    3. Predicts on TEST.
    """
    best_t = 0.5
    best_train_err = float('inf')
    best_v_flat = vbar.max()
    
    # Sweep thresholds to find the best fit for this specific galaxy/feature
    # giving the baseline every advantage.
    thresholds = np.linspace(0.1, 2.0, 40)
    
    for t in thresholds:
        # Evaluate on TRAIN
        train_gate = input_feature[train_idx] >= t
        
        # Fit V_flat
        if np.sum(train_gate) > 0:
            v_flat = np.mean(vobs[train_idx][train_gate])
        else:
            v_flat = vbar.max()
            
        # Predict Train
        pred = np.copy(vbar[train_idx])
        pred[train_gate] = v_flat
        err = calculate_rmse(vobs[train_idx], pred)
        
        if err < best_train_err:
            best_train_err = err
            best_t = t
            best_v_flat = v_flat
            
    # --- EVALUATE ON TEST ---
    test_gate = input_feature[test_idx] >= best_t
    pred_test = np.copy(vbar[test_idx])
    pred_test[test_gate] = best_v_flat
    
    return calculate_rmse(vobs[test_idx], pred_test), best_t

def run_steel_man_test():
    print(f"🦾 RUNNING ULTIMATE 'STEEL MAN' BENCHMARK")
    print(f"   Optimization: Every model gets to pick its perfect threshold.")
    print(f"   This gives baselines MAXIMUM advantage.")
    
    files = glob.glob(os.path.join(SPARC_DIR, "*_rotmod.dat"))
    
    results = []
    
    for f in files:
        data = load_galaxy_data(f)
        if data is None or len(data[0]) < 6: continue
        
        rad, vbar, vobs = data
        n_points = len(rad)
        
        # Split
        train_idx = np.arange(0, n_points, 2)
        test_idx = np.arange(1, n_points, 2)
        
        # Scaling
        v_scale = vbar.max() if vbar.max() > 0 else 1.0
        r_scale = rad.max() if rad.max() > 0 else 1.0
        
        n_vbar = vbar / v_scale
        n_rad = rad / r_scale
        n_sum = n_vbar + n_rad
        
        # Run Optimizers - each model gets to find its BEST threshold
        rmse_rad, t_rad = optimize_and_predict(n_rad, vbar, vobs, train_idx, test_idx)
        rmse_mass, t_mass = optimize_and_predict(n_vbar, vbar, vobs, train_idx, test_idx)
        rmse_sum, t_sum = optimize_and_predict(n_sum, vbar, vobs, train_idx, test_idx)
        
        results.append({
            'name': os.path.basename(f).replace('_rotmod.dat', ''),
            'rad': rmse_rad,
            'mass': rmse_mass,
            'sum': rmse_sum,
            't_rad': t_rad,
            't_mass': t_mass,
            't_sum': t_sum
        })

    # Stats
    wins_sum = 0
    wins_rad = 0
    wins_mass = 0
    
    for r in results:
        best = min(r['rad'], r['mass'], r['sum'])
        if r['sum'] == best: wins_sum += 1
        elif r['rad'] == best: wins_rad += 1
        elif r['mass'] == best: wins_mass += 1
    
    total = len(results)
    
    rmse_rad_arr = np.array([r['rad'] for r in results])
    rmse_mass_arr = np.array([r['mass'] for r in results])
    rmse_sum_arr = np.array([r['sum'] for r in results])
    
    avg_rad = np.mean(rmse_rad_arr)
    avg_mass = np.mean(rmse_mass_arr)
    avg_sum = np.mean(rmse_sum_arr)
    
    med_rad = np.median(rmse_rad_arr)
    med_mass = np.median(rmse_mass_arr)
    med_sum = np.median(rmse_sum_arr)
    
    print("\n" + "="*65)
    print("ULTIMATE BENCHMARK RESULTS (All Models Fully Optimized)")
    print("="*65)
    print(f"Galaxies Analyzed: {total}")
    print(f"\n{'Model':<30} {'Mean RMSE':>12} {'Median':>10} {'Wins':>10}")
    print("-"*65)
    print(f"{'Radius Latch (Optimized)':<30} {avg_rad:>10.2f}   {med_rad:>10.2f}   {wins_rad:>5}/{total}")
    print(f"{'Mass Latch (Optimized)':<30} {avg_mass:>10.2f}   {med_mass:>10.2f}   {wins_mass:>5}/{total}")
    print(f"{'Overflow Latch (Optimized)':<30} {avg_sum:>10.2f}   {med_sum:>10.2f}   {wins_sum:>5}/{total}")
    print("-"*65)
    
    # Statistical tests
    print("\n" + "="*65)
    print("STATISTICAL SIGNIFICANCE (Even with Optimized Baselines)")
    print("="*65)
    
    # Overflow vs Radius (optimized)
    overflow_beats_rad = np.sum(rmse_sum_arr < rmse_rad_arr)
    rad_beats_overflow = np.sum(rmse_rad_arr < rmse_sum_arr)
    ties_rad = np.sum(rmse_sum_arr == rmse_rad_arr)
    diff_rad = rmse_rad_arr - rmse_sum_arr
    stat1, pval1 = stats.wilcoxon(rmse_rad_arr, rmse_sum_arr, alternative='greater')
    
    print(f"\nOverflow vs Radius (both optimized):")
    print(f"  Overflow wins: {overflow_beats_rad}/{total} ({overflow_beats_rad/total:.1%})")
    print(f"  Radius wins:   {rad_beats_overflow}/{total} ({rad_beats_overflow/total:.1%})")
    print(f"  Ties:          {ties_rad}/{total}")
    print(f"  Mean improvement: {np.mean(diff_rad):.2f} km/s")
    print(f"  Wilcoxon p-value: {pval1:.6f}")
    
    # Overflow vs Mass (optimized)
    overflow_beats_mass = np.sum(rmse_sum_arr < rmse_mass_arr)
    diff_mass = rmse_mass_arr - rmse_sum_arr
    stat2, pval2 = stats.wilcoxon(rmse_mass_arr, rmse_sum_arr, alternative='greater')
    
    print(f"\nOverflow vs Mass (both optimized):")
    print(f"  Overflow wins: {overflow_beats_mass}/{total} ({overflow_beats_mass/total:.1%})")
    print(f"  Mean improvement: {np.mean(diff_mass):.2f} km/s")
    print(f"  Wilcoxon p-value: {pval2:.6f}")
    
    # Report optimal thresholds
    print("\n" + "="*65)
    print("OPTIMAL THRESHOLD ANALYSIS")
    print("="*65)
    t_rad_arr = np.array([r['t_rad'] for r in results])
    t_mass_arr = np.array([r['t_mass'] for r in results])
    t_sum_arr = np.array([r['t_sum'] for r in results])
    
    print(f"Radius model optimal t:   mean={np.mean(t_rad_arr):.2f}, median={np.median(t_rad_arr):.2f}")
    print(f"Mass model optimal t:     mean={np.mean(t_mass_arr):.2f}, median={np.median(t_mass_arr):.2f}")
    print(f"Overflow model optimal t: mean={np.mean(t_sum_arr):.2f}, median={np.median(t_sum_arr):.2f}")
    
    # Final verdict
    print("\n" + "="*65)
    if pval1 < 0.05 and overflow_beats_rad > total * 0.5:
        print("✅ ULTIMATE SUCCESS: Overflow beats OPTIMIZED baselines!")
        print("   Even when Radius/Mass models pick their BEST threshold,")
        print("   the Overflow Gate (V_bar + R) still wins.")
        print("   This is STRUCTURAL superiority, not parameter tuning.")
    elif overflow_beats_rad > total * 0.5:
        print("✅ POSITIVE: Overflow wins majority against optimized baselines.")
    else:
        print("⚠️  Optimized baselines match or beat Overflow.")
    print("="*65)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Box plot
    ax = axes[0]
    bp = ax.boxplot([rmse_rad_arr, rmse_mass_arr, rmse_sum_arr], 
                     tick_labels=['Radius\n(Optimized)', 'Mass\n(Optimized)', 'Overflow\n(Optimized)'])
    ax.set_ylabel("Test RMSE (km/s)")
    ax.set_title("Ultimate Benchmark: All Models Optimized")
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Scatter - Overflow vs best baseline
    ax = axes[1]
    best_baseline = np.minimum(rmse_rad_arr, rmse_mass_arr)
    ax.scatter(best_baseline, rmse_sum_arr, alpha=0.5, c='blue', s=30)
    max_val = max(best_baseline.max(), rmse_sum_arr.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Equal')
    overflow_beats_best = np.sum(rmse_sum_arr < best_baseline)
    ax.set_xlabel("Best Baseline RMSE (min of Radius, Mass)")
    ax.set_ylabel("Overflow RMSE")
    ax.set_title(f"Overflow vs BEST Baseline\nOverflow wins: {overflow_beats_best}/{total} ({overflow_beats_best/total:.0%})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "ultimate_benchmark.png")
    plt.savefig(out_path, dpi=150)
    print(f"\n[SAVED] {out_path}")

if __name__ == "__main__":
    run_steel_man_test()
