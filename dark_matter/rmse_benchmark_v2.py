"""
RMSE BENCHMARK V2 - NO LEAKAGE
==============================
Addressing Reviewer Critique: "V_flat must not peek at V_obs"

This version uses THREE approaches to set V_flat WITHOUT target leakage:
1. V_flat = max(V_bar) - Pure input-only
2. V_flat = mean(V_bar[overflow]) - Input-only, overflow-aware
3. V_flat = V_bar at overflow point - The "latch value"

Baselines:
1. Newtonian: V_pred = V_bar (weak null)
2. Fitted Constant: V_pred = V_bar inner, mean(V_obs) outer (EQUALLY UNFAIR leak)
   - If this performs same as Digital Latch, the gate isn't helping
3. Simple Flat: V_pred = max(V_bar) everywhere in outer region (no gate logic)

The test: If Digital Latch (input-only) beats all baselines, the overflow gate 
is doing real predictive work.
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

def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted)**2))

def run_benchmark():
    print("📊 RMSE BENCHMARK V2 - NO TARGET LEAKAGE")
    print("="*60)
    
    files = glob.glob(os.path.join(SPARC_DIR, "*_rotmod.dat"))
    
    results = []
    
    for f in files:
        name = os.path.basename(f).replace("_rotmod.dat", "")
        data = load_galaxy_data(f)
        if data is None or len(data[0]) < 5: continue
        
        rad, vbar, vobs = data
        n = len(rad)
        
        # Normalize inputs (NO V_obs in scaling!)
        v_scale = vbar.max() if vbar.max() > 0 else 1.0
        r_scale = rad.max() if rad.max() > 0 else 1.0
        
        n_vbar = vbar / v_scale
        n_rad = rad / r_scale
        
        # Overflow mask (the gate)
        overflow_mask = (n_vbar + n_rad) >= 1.0
        n_overflow = np.sum(overflow_mask)
        
        # Skip if not enough points in each region
        if n_overflow < 2 or (n - n_overflow) < 2:
            continue
        
        # ============================================================
        # MODEL 1: NEWTONIAN (Weak Null - known to fail)
        # ============================================================
        pred_newton = vbar
        rmse_newton = rmse(vobs, pred_newton)
        
        # ============================================================
        # MODEL 2: DIGITAL LATCH - INPUT ONLY V_flat
        # V_flat = max(V_bar) - no peeking at V_obs!
        # ============================================================
        v_flat_input = vbar.max()  # PURE INPUT-ONLY
        pred_digital_input = np.copy(vbar)
        pred_digital_input[overflow_mask] = v_flat_input
        rmse_digital_input = rmse(vobs, pred_digital_input)
        
        # ============================================================
        # MODEL 3: DIGITAL LATCH - V_flat from V_bar at overflow points
        # Still input-only: mean of V_bar where overflow occurs
        # ============================================================
        v_flat_bar_overflow = np.mean(vbar[overflow_mask])
        pred_digital_bar = np.copy(vbar)
        pred_digital_bar[overflow_mask] = v_flat_bar_overflow
        rmse_digital_bar = rmse(vobs, pred_digital_bar)
        
        # ============================================================
        # BASELINE A: "SIMPLE FLAT" - No gate logic
        # Just use max(V_bar) for outer 50% of radii
        # ============================================================
        outer_mask = n_rad >= 0.5  # Simple radius cutoff, no gate
        pred_simple_flat = np.copy(vbar)
        pred_simple_flat[outer_mask] = vbar.max()
        rmse_simple_flat = rmse(vobs, pred_simple_flat)
        
        # ============================================================
        # BASELINE B: "FITTED CONSTANT" (Equally unfair - uses V_obs)
        # This is to show the gate matters, not just the constant
        # ============================================================
        v_flat_cheating = np.mean(vobs[overflow_mask])  # LEAKY!
        pred_cheating = np.copy(vbar)
        pred_cheating[overflow_mask] = v_flat_cheating
        rmse_cheating = rmse(vobs, pred_cheating)
        
        results.append({
            'name': name,
            'n_points': n,
            'n_overflow': n_overflow,
            'rmse_newton': rmse_newton,
            'rmse_digital_input': rmse_digital_input,
            'rmse_digital_bar': rmse_digital_bar,
            'rmse_simple_flat': rmse_simple_flat,
            'rmse_cheating': rmse_cheating,
        })
    
    # ============================================================
    # ANALYSIS
    # ============================================================
    n_galaxies = len(results)
    
    print(f"\nGalaxies Analyzed: {n_galaxies}")
    print("\n" + "="*60)
    print("MODEL COMPARISON (Lower RMSE = Better)")
    print("="*60)
    
    # Extract arrays
    rmse_newton = np.array([r['rmse_newton'] for r in results])
    rmse_digital_input = np.array([r['rmse_digital_input'] for r in results])
    rmse_digital_bar = np.array([r['rmse_digital_bar'] for r in results])
    rmse_simple_flat = np.array([r['rmse_simple_flat'] for r in results])
    rmse_cheating = np.array([r['rmse_cheating'] for r in results])
    
    models = [
        ("Newtonian (V_bar only)", rmse_newton),
        ("Simple Flat (r>0.5, no gate)", rmse_simple_flat),
        ("Digital Latch (V_flat=max(V_bar))", rmse_digital_input),
        ("Digital Latch (V_flat=mean(V_bar[over]))", rmse_digital_bar),
        ("Fitted Constant (LEAKY baseline)", rmse_cheating),
    ]
    
    print(f"\n{'Model':<45} {'Mean RMSE':>12} {'Median':>10} {'Wins vs Newton':>15}")
    print("-"*85)
    
    for name, rmse_arr in models:
        wins = np.sum(rmse_arr < rmse_newton)
        pct = wins / n_galaxies * 100
        print(f"{name:<45} {np.mean(rmse_arr):>10.2f}   {np.median(rmse_arr):>10.2f}   {wins:>5}/{n_galaxies} ({pct:>5.1f}%)")
    
    # ============================================================
    # KEY TEST: Does the GATE help vs simple radius cutoff?
    # ============================================================
    print("\n" + "="*60)
    print("KEY DIAGNOSTIC: Does the Overflow Gate add value?")
    print("="*60)
    
    # Compare Digital Latch (input-only) vs Simple Flat
    gate_wins_vs_simple = np.sum(rmse_digital_input < rmse_simple_flat)
    gate_improvement = (rmse_simple_flat - rmse_digital_input) / rmse_simple_flat * 100
    
    print(f"\nDigital Latch vs Simple Flat (r>0.5):")
    print(f"  Gate wins: {gate_wins_vs_simple}/{n_galaxies} ({gate_wins_vs_simple/n_galaxies:.1%})")
    print(f"  Mean improvement: {np.mean(gate_improvement):.1f}%")
    print(f"  Median improvement: {np.median(gate_improvement):.1f}%")
    
    # Sign test
    stat, pval = stats.wilcoxon(rmse_simple_flat, rmse_digital_input, alternative='greater')
    print(f"  Wilcoxon signed-rank p-value: {pval:.4f}")
    
    if pval < 0.05 and gate_wins_vs_simple > n_galaxies * 0.5:
        print("\n✅ The Overflow Gate provides SIGNIFICANT improvement over simple radius cutoff!")
    else:
        print("\n⚠️  The Overflow Gate does NOT clearly beat simple radius cutoff.")
    
    # ============================================================
    # How much does leaking V_obs help?
    # ============================================================
    print("\n" + "="*60)
    print("LEAKAGE DIAGNOSTIC: How much does peeking at V_obs help?")
    print("="*60)
    
    leakage_gain = (rmse_digital_input - rmse_cheating) / rmse_digital_input * 100
    print(f"  Extra gain from using V_obs: {np.mean(leakage_gain):.1f}% mean, {np.median(leakage_gain):.1f}% median")
    print(f"  (This is the unfair advantage we removed)")
    
    # ============================================================
    # PLOTTING
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Newton vs Digital Latch (Input-Only)
    ax = axes[0]
    ax.scatter(rmse_newton, rmse_digital_input, alpha=0.5, c='blue', s=20)
    max_val = max(rmse_newton.max(), rmse_digital_input.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Equal')
    ax.set_xlabel("Newtonian RMSE (km/s)")
    ax.set_ylabel("Digital Latch RMSE (km/s)")
    ax.set_title(f"Newton vs Digital Latch\n(Input-Only, NO Leakage)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Simple Flat vs Digital Latch (The key comparison)
    ax = axes[1]
    ax.scatter(rmse_simple_flat, rmse_digital_input, alpha=0.5, c='green', s=20)
    max_val = max(rmse_simple_flat.max(), rmse_digital_input.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Equal')
    ax.set_xlabel("Simple Flat (r>0.5) RMSE")
    ax.set_ylabel("Digital Latch RMSE")
    ax.set_title(f"Does the Gate Help?\nGate wins: {gate_wins_vs_simple}/{n_galaxies}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of improvements
    ax = axes[2]
    improvement_vs_newton = (rmse_newton - rmse_digital_input) / rmse_newton * 100
    ax.hist(improvement_vs_newton, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', label='No improvement')
    ax.axvline(x=np.median(improvement_vs_newton), color='g', linestyle='-', 
               label=f'Median: {np.median(improvement_vs_newton):.1f}%')
    ax.set_xlabel("% Improvement vs Newtonian")
    ax.set_ylabel("Number of Galaxies")
    ax.set_title("Distribution of Improvements\n(Input-Only Digital Latch)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "rmse_benchmark_v2.png")
    plt.savefig(out_path, dpi=150)
    print(f"\n[SAVED] {out_path}")

if __name__ == "__main__":
    run_benchmark()
