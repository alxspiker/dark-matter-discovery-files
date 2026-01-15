"""
FAIR BASELINE BENCHMARK
=======================
Addressing Reviewer Critique: "Comparing against Fair Baselines."

Goal: Prove that the "Overflow Gate" (Sum >= 1.0) is a better predictor
than simple Radius or Mass thresholds, when all models are allowed 
to fit the same single parameter (V_flat).

Models (All use 50/50 Train/Test Split):
1. Radius Latch: If R_norm >= 0.5, V = V_flat. Else V = V_bar.
2. Mass Latch:   If V_bar_norm >= 0.5, V = V_flat. Else V = V_bar.
3. Overflow Latch (Ours): If Sum >= 1.0, V = V_flat. Else V = V_bar.

Hypothesis: The Overflow Latch will have the lowest RMSE because it
captures the specific physical condition where gravity changes behavior.
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

def fit_and_predict(rad, vbar, vobs, train_idx, test_idx, gate_mask):
    """
    Fits V_flat on training data where gate is TRUE.
    Predicts on test data.
    """
    # Fit Parameter (V_flat)
    train_gate = gate_mask[train_idx]
    train_vobs = vobs[train_idx]
    
    if np.sum(train_gate) > 0:
        v_flat = np.mean(train_vobs[train_gate])
    else:
        v_flat = vbar.max() # Fallback
        
    # Predict
    test_gate = gate_mask[test_idx]
    pred = np.copy(vbar[test_idx]) # Default Newtonian
    pred[test_gate] = v_flat       # Latch
    
    return calculate_rmse(vobs[test_idx], pred)

def run_fair_benchmark():
    print(f"🥊 RUNNING FAIR BASELINE BENCHMARK")
    print(f"   Competition: Who picks the best 'Latch Point'?")
    print(f"   All models get: 1 fitted parameter (V_flat), 50/50 train/test split")
    
    files = glob.glob(os.path.join(SPARC_DIR, "*_rotmod.dat"))
    
    wins_overflow = 0
    wins_radius = 0
    wins_mass = 0
    ties = 0
    
    results = [] # (name, rmse_rad, rmse_mass, rmse_overflow)
    
    for f in files:
        name = os.path.basename(f).replace("_rotmod.dat", "")
        data = load_galaxy_data(f)
        if data is None or len(data[0]) < 6: continue
        
        rad, vbar, vobs = data
        n_points = len(rad)
        
        # Split
        train_idx = np.arange(0, n_points, 2)
        test_idx = np.arange(1, n_points, 2)
        
        # Scaling (Inputs Only)
        v_scale = vbar.max() if vbar.max() > 0 else 1.0
        r_scale = rad.max() if rad.max() > 0 else 1.0
        n_vbar = vbar / v_scale
        n_rad = rad / r_scale
        
        # --- MODEL 1: RADIUS LATCH (Gate: R >= 0.5) ---
        # The critic's hypothesis: "Just flatten the outer half"
        rmse_rad = fit_and_predict(rad, vbar, vobs, train_idx, test_idx, n_rad >= 0.5)
        
        # --- MODEL 2: MASS LATCH (Gate: Vbar >= 0.5) ---
        rmse_mass = fit_and_predict(rad, vbar, vobs, train_idx, test_idx, n_vbar >= 0.5)
        
        # --- MODEL 3: OVERFLOW LATCH (Gate: Sum >= 1.0) ---
        rmse_overflow = fit_and_predict(rad, vbar, vobs, train_idx, test_idx, (n_vbar + n_rad) >= 1.0)
        
        results.append({
            'name': name,
            'rmse_rad': rmse_rad,
            'rmse_mass': rmse_mass,
            'rmse_overflow': rmse_overflow
        })
        
        # Who won?
        best = min(rmse_rad, rmse_mass, rmse_overflow)
        if rmse_overflow == best and rmse_rad == best:
            ties += 1
        elif rmse_overflow == best:
            wins_overflow += 1
        elif rmse_rad == best:
            wins_radius += 1
        elif rmse_mass == best:
            wins_mass += 1

    # Stats
    total = len(results)
    rmse_rad_arr = np.array([r['rmse_rad'] for r in results])
    rmse_mass_arr = np.array([r['rmse_mass'] for r in results])
    rmse_overflow_arr = np.array([r['rmse_overflow'] for r in results])
    
    avg_rad = np.mean(rmse_rad_arr)
    avg_mass = np.mean(rmse_mass_arr)
    avg_overflow = np.mean(rmse_overflow_arr)
    
    med_rad = np.median(rmse_rad_arr)
    med_mass = np.median(rmse_mass_arr)
    med_overflow = np.median(rmse_overflow_arr)
    
    print("\n" + "="*60)
    print("HEAD-TO-HEAD RESULTS (All models: 1 param, same train/test)")
    print("="*60)
    print(f"Galaxies Analyzed: {total}")
    print(f"\n{'Model':<25} {'Mean RMSE':>12} {'Median':>10} {'Wins':>10}")
    print("-"*60)
    print(f"{'Radius Latch (r≥0.5)':<25} {avg_rad:>10.2f}   {med_rad:>10.2f}   {wins_radius:>5}/{total}")
    print(f"{'Mass Latch (v≥0.5)':<25} {avg_mass:>10.2f}   {med_mass:>10.2f}   {wins_mass:>5}/{total}")
    print(f"{'Overflow Latch (sum≥1)':<25} {avg_overflow:>10.2f}   {med_overflow:>10.2f}   {wins_overflow:>5}/{total}")
    print("-"*60)
    
    # Statistical tests: Overflow vs Radius
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE")
    print("="*60)
    
    # Paired comparison: Overflow vs Radius
    diff_vs_radius = rmse_rad_arr - rmse_overflow_arr  # Positive = Overflow wins
    overflow_beats_radius = np.sum(diff_vs_radius > 0)
    
    stat, pval = stats.wilcoxon(rmse_rad_arr, rmse_overflow_arr, alternative='greater')
    print(f"\nOverflow vs Radius Latch:")
    print(f"  Overflow wins: {overflow_beats_radius}/{total} ({overflow_beats_radius/total:.1%})")
    print(f"  Mean improvement: {np.mean(diff_vs_radius):.2f} km/s")
    print(f"  Wilcoxon p-value: {pval:.4f}")
    
    # Paired comparison: Overflow vs Mass
    diff_vs_mass = rmse_mass_arr - rmse_overflow_arr
    overflow_beats_mass = np.sum(diff_vs_mass > 0)
    
    stat2, pval2 = stats.wilcoxon(rmse_mass_arr, rmse_overflow_arr, alternative='greater')
    print(f"\nOverflow vs Mass Latch:")
    print(f"  Overflow wins: {overflow_beats_mass}/{total} ({overflow_beats_mass/total:.1%})")
    print(f"  Mean improvement: {np.mean(diff_vs_mass):.2f} km/s")
    print(f"  Wilcoxon p-value: {pval2:.4f}")
    
    print("\n" + "="*60)
    if pval < 0.05 and overflow_beats_radius > total * 0.5:
        print("✅ SUCCESS: Overflow Gate SIGNIFICANTLY beats Radius baseline!")
        print("   The (V_bar + R) logic is physically meaningful, not arbitrary.")
    elif overflow_beats_radius > total * 0.5:
        print("✅ POSITIVE: Overflow wins majority, but not statistically significant.")
    else:
        print("⚠️  The Overflow Gate does not clearly beat the Radius baseline.")
    print("="*60)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Scatter - Overflow vs Radius
    ax = axes[0]
    ax.scatter(rmse_rad_arr, rmse_overflow_arr, alpha=0.5, c='blue', s=30)
    max_val = max(rmse_rad_arr.max(), rmse_overflow_arr.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Equal')
    ax.set_xlabel("Radius Latch RMSE (km/s)")
    ax.set_ylabel("Overflow Latch RMSE (km/s)")
    ax.set_title(f"Overflow vs Radius Latch\nOverflow wins: {overflow_beats_radius}/{total} ({overflow_beats_radius/total:.0%})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Box plot comparison
    ax = axes[1]
    box_data = [rmse_rad_arr, rmse_mass_arr, rmse_overflow_arr]
    bp = ax.boxplot(box_data, tick_labels=['Radius\n(r≥0.5)', 'Mass\n(v≥0.5)', 'Overflow\n(sum≥1)'])
    ax.set_ylabel("RMSE (km/s)")
    ax.set_title("Distribution of Errors by Gate Logic")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "fair_benchmark.png")
    plt.savefig(out_path, dpi=150)
    print(f"\n[SAVED] {out_path}")

if __name__ == "__main__":
    run_fair_benchmark()
