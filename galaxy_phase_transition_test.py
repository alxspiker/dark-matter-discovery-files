"""
GALAXY-LEVEL PHASE TRANSITION TEST
==================================
Addressing Critique: "Pseudo-replication / Independence."

Goal: Verify the "Phase Transition" exists when treating the GALAXY
as the unit of analysis, rather than pooling 3000+ individual points.

Methodology:
1. For each galaxy:
   - Normalize inputs (Blind).
   - Split points into "Low Regime" (Sum < 1.0) and "High Regime" (Sum >= 1.0).
   - Calculate Mean V_obs for both regimes.
   - Calculate the "Jump" (Mean_High - Mean_Low).
2. Perform a Paired T-Test / Wilcoxon Test on the collection of Jumps.

This removes the bias of massive galaxies dominating the statistics.
"""

import os
import numpy as np
import scipy.stats as stats
import re
import glob
import matplotlib.pyplot as plt

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

def run_galaxy_level_test():
    print(f"📉 RUNNING GALAXY-LEVEL PHASE TRANSITION TEST")
    
    files = glob.glob(os.path.join(SPARC_DIR, "*_rotmod.dat"))
    
    jumps = []
    valid_galaxies = 0
    
    for f in files:
        data = load_galaxy_data(f)
        if data is None or len(data[0]) < 4: continue
        
        rad, vbar, vobs = data
        
        # Blind Normalization
        v_scale = vbar.max() if vbar.max() > 0 else 1.0
        r_scale = rad.max() if rad.max() > 0 else 1.0
        
        n_vbar = vbar / v_scale
        n_rad = rad / r_scale
        n_vobs = vobs / v_scale
        
        # The Gate
        x = n_vbar + n_rad
        mask_high = x >= 1.0
        mask_low = x < 1.0
        
        # We need data in BOTH regimes to measure a jump
        if np.sum(mask_high) > 0 and np.sum(mask_low) > 0:
            mean_low = np.mean(n_vobs[mask_low])
            mean_high = np.mean(n_vobs[mask_high])
            
            # The Metric: Difference in normalized velocity
            jump = mean_high - mean_low
            jumps.append(jump)
            valid_galaxies += 1

    # --- STATISTICS ---
    jumps = np.array(jumps)
    
    # 1. Wilcoxon Signed-Rank Test (Non-parametric, robust to outliers)
    # Null Hypothesis: The median jump is 0 (no effect).
    w_stat, p_val = stats.wilcoxon(jumps)
    
    # 2. T-Test (Parametric)
    t_stat, t_p_val = stats.ttest_1samp(jumps, 0.0)
    
    print("\n[RESULTS]")
    print(f"Valid Galaxies (Crossing Threshold): {valid_galaxies}")
    print(f"Mean Velocity Jump: +{np.mean(jumps):.3f} (Normalized Units)")
    print(f"Median Velocity Jump: +{np.median(jumps):.3f}")
    print("-" * 40)
    print(f"Wilcoxon p-value: {p_val:.10e}")
    print(f"T-Test p-value:   {t_p_val:.10e}")
    
    if p_val < 0.001:
        print("\n✅ SUCCESS: The Phase Transition is statistically significant per-galaxy.")
        print(f"   Probability this is random noise: ~1 in {1/p_val:.0e}")
    else:
        print("\n❌ FAILED: The effect disappears when averaging per galaxy.")

    # Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(jumps, bins=20, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label="No Effect")
    plt.title("Distribution of Velocity Jumps (Per Galaxy)")
    plt.xlabel("Velocity Increase at Threshold (Normalized)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("galaxy_phase_hist.png")
    print("[SAVED] galaxy_phase_hist.png")

if __name__ == "__main__":
    run_galaxy_level_test()
