"""
RIGOROUS VERIFICATION ENGINE
============================
Addressing Reviewer Critique: "Target Leakage in Normalization"

Goal: Test the 'Gravitational Overflow' hypothesis WITHOUT using v_obs
to scale the inputs.

Methodology:
1. Normalize V_bar by V_bar.max() (Strict Input-Only Scaling).
2. Normalize R by R.max().
3. Calculate Digital Sum = Norm(V_bar) + Norm(R).
4. Analyze correlation with V_obs WITHOUT previous leakage.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import re

# The 31 "Golden Set" galaxies
GOLDEN_SET = [
    "D512-2", "D564-8", "DDO064", "DDO168", "ESO116-G012",
    "ESO444-G084", "F571-V1", "NGC3109", "NGC3949", "NGC4085",
    "NGC4389", "NGC5005", "NGC5585", "UGC00891", "UGC02023",
    "UGC02259", "UGC04278", "UGC04483", "UGC05005", "UGC05414",
    "UGC05721", "UGC06923", "UGC07089", "UGC07261", "UGC07323",
    "UGC07866", "UGC08837", "UGC11820", "UGCA281", "UGCA442",
    "UGCA444"
]

SPARC_DIR = os.path.join(os.path.dirname(__file__), "sparc_galaxies")

def load_galaxy_data(name):
    path = os.path.join(SPARC_DIR, f"{name}_rotmod.dat")
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

def run_blind_test():
    print(f"📉 RUNNING RIGOROUS 'BLIND' TEST")
    print(f"   Sample: {len(GOLDEN_SET)} Galaxies")
    print(f"   Constraint: NO target leakage. V_bar scaled by V_bar only.")
    
    x_vals = [] # Input Sum
    y_vals = [] # V_obs (Normalized by V_bar max to see excess)
    
    r_vals = [] # Just radius (control)
    
    for name in GOLDEN_SET:
        data = load_galaxy_data(name)
        if data is None: continue
        rad, vbar, vobs = data
        
        # --- CRITIC FIX: NO TARGET LEAKAGE ---
        # We scale inputs using ONLY input statistics.
        
        # Scale Velocity by the max BARYONIC velocity of this galaxy
        # (If V_obs goes higher than this, y > 1.0, showing Dark Matter)
        v_scale = vbar.max() 
        
        # Scale Radius by max radius
        r_scale = rad.max()
        
        # Normalized Inputs
        n_vbar = vbar / v_scale
        n_rad = rad / r_scale
        
        # The Hypothesis: Sum > 1.0 triggers "Carry Out" state
        digital_sum = n_vbar + n_rad
        
        # Output: We normalize V_obs by V_bar_max.
        # If y > 1.0, it means V_obs > V_bar_max (Dark Matter is required).
        n_vobs = vobs / v_scale
        
        x_vals.extend(digital_sum)
        y_vals.extend(n_vobs)
        r_vals.extend(n_rad)

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    r_vals = np.array(r_vals)
    
    # --- STATISTICAL TEST ---
    
    # The Threshold is 1.0 (The theoretical Carry Out point)
    mask_high = x_vals >= 1.0
    mask_low = x_vals < 1.0
    
    mean_low = np.mean(y_vals[mask_low])
    mean_high = np.mean(y_vals[mask_high])
    
    print("\n[RESULTS]")
    print(f"Mean V_obs/V_bar (Below Threshold): {mean_low:.3f}")
    print(f"Mean V_obs/V_bar (Above Threshold): {mean_high:.3f}")
    print(f"Ratio (High/Low): {mean_high/mean_low:.2f}x")
    
    # --- PLOTTING ---
    plt.figure(figsize=(12, 5))
    
    # Plot 1: The Hypothesis
    plt.subplot(1, 2, 1)
    plt.scatter(x_vals, y_vals, alpha=0.3, s=10, c='blue')
    plt.axvline(x=1.0, color='r', linestyle='--', linewidth=2, label="Overflow Threshold")
    plt.axhline(y=1.0, color='k', linestyle=':', label="Baryonic Limit")
    plt.title("Hypothesis: Input Sum vs Velocity")
    plt.xlabel("Input Sum (Norm V_bar + Norm R)")
    plt.ylabel("V_obs / Max(V_bar)")
    plt.grid(True, alpha=0.3)
    
    # Plot 2: The Control (Just Radius)
    plt.subplot(1, 2, 2)
    plt.scatter(r_vals, y_vals, alpha=0.3, s=10, c='gray')
    plt.title("Control: Radius vs Velocity")
    plt.xlabel("Normalized Radius")
    plt.ylabel("V_obs / Max(V_bar)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "rigorous_check.png")
    plt.savefig(out_path)
    print(f"\n[SAVED] {out_path}")
    
    # Interpretation logic
    if mean_high > 1.2 and (mean_high - mean_low) > 0.3:
        print("\n✅ PASSED: The signal survives blind normalization.")
        print("   V_obs is significantly higher when Input Sum > 1.0.")
    else:
        print("\n❌ FAILED: The signal collapsed.")
        print("   The critic was right; it was likely an artifact of scaling.")

if __name__ == "__main__":
    run_blind_test()
