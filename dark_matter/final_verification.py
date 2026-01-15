"""
FINAL VERIFICATION ENGINE
=========================
Addressing the "Hostile Reviewer" Checklist.

1. Independence: Statistics calculated per-galaxy, not pooled.
2. Baselines: Comparing (Vbar + R) against Vbar-only and R-only.
3. Robustness: Sweeping the threshold (0.5 to 1.5) to check for a peak at 1.0.
4. Generalization: Testing on the 'Golden Set' vs the 'Rest of SPARC'.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import re
import glob
from scipy.stats import sem # Standard Error of Mean

# The 31 "Golden Set" galaxies (Training Data)
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

def get_galaxy_jump_factor(rad, vbar, vobs, predictor_type='sum', threshold=1.0):
    """
    Calculates the 'Jump Factor' (Mean High / Mean Low) for a single galaxy.
    """
    # 1. Blind Normalization (Input Scale Only)
    v_scale = vbar.max() if vbar.max() > 0 else 1.0
    r_scale = rad.max() if rad.max() > 0 else 1.0
    
    n_vbar = vbar / v_scale
    n_rad = rad / r_scale
    n_vobs = vobs / v_scale # Normalize output by input scale to see excess
    
    # 2. Select Predictor
    if predictor_type == 'sum':
        x = n_vbar + n_rad
    elif predictor_type == 'vbar':
        x = n_vbar
    elif predictor_type == 'rad':
        x = n_rad
    
    # 3. Calculate Jump
    mask_high = x >= threshold
    mask_low = x < threshold
    
    # Require at least 2 points in each bin to be valid
    if np.sum(mask_high) < 2 or np.sum(mask_low) < 2:
        return None
        
    mean_high = np.mean(n_vobs[mask_high])
    mean_low = np.mean(n_vobs[mask_low])
    
    # Avoid divide by zero
    if mean_low == 0: return None
    
    return mean_high / mean_low

def analyze_population(galaxy_list, label):
    print(f"\n--- Analyzing {label} ({len(galaxy_list)} galaxies) ---")
    
    # 1. Threshold Sweep (Robustness Check)
    # For Sum model: range is [0, 2], so sweep [0.5, 1.5]
    # The "overflow" point is at 1.0
    thresholds = np.linspace(0.5, 1.5, 21)
    
    med_jumps_sum = []
    med_jumps_vbar = []
    med_jumps_rad = []
    
    for t in thresholds:
        # Sum Model - threshold directly
        jumps = [get_galaxy_jump_factor(g[0], g[1], g[2], 'sum', t) for g in galaxy_list]
        jumps = [j for j in jumps if j is not None]
        med_jumps_sum.append(np.median(jumps) if jumps else 0)
        
        # Vbar Model - scale threshold to [0,1] range
        # t=0.5 in sum space -> 0.25 in vbar space; t=1.0 -> 0.5; t=1.5 -> 0.75
        t_scaled = (t - 0.5) / 2 + 0.25  # Maps [0.5,1.5] -> [0.25, 0.75]
        jumps = [get_galaxy_jump_factor(g[0], g[1], g[2], 'vbar', t_scaled) for g in galaxy_list]
        jumps = [j for j in jumps if j is not None]
        med_jumps_vbar.append(np.median(jumps) if jumps else 0)

        # Rad Model - same scaling
        jumps = [get_galaxy_jump_factor(g[0], g[1], g[2], 'rad', t_scaled) for g in galaxy_list]
        jumps = [j for j in jumps if j is not None]
        med_jumps_rad.append(np.median(jumps) if jumps else 0)
    
    # Find peak for Sum model
    peak_idx = np.argmax(med_jumps_sum)
    peak_val = med_jumps_sum[peak_idx]
    peak_thresh = thresholds[peak_idx]
    
    print(f"Peak Jump Factor (Sum Model): {peak_val:.2f}x at Threshold {peak_thresh:.2f}")
    
    # Plot Sweep
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, med_jumps_sum, 'b-o', linewidth=2, label='Proposed (Mass + Radius)')
    plt.plot(thresholds, med_jumps_vbar, 'g--', label='Baseline (Mass Only)')
    plt.plot(thresholds, med_jumps_rad, 'c:', label='Baseline (Radius Only)')
    plt.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='Theoretical Overflow (1.0)')
    
    plt.title(f"Threshold Robustness: {label}")
    plt.xlabel("Threshold Value")
    plt.ylabel("Median Velocity Jump (High/Low)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(os.path.dirname(__file__), f"threshold_sweep_{label.replace(' ', '_')}.png")
    plt.savefig(out_path)
    print(f"Saved {out_path}")

    # 2. Boxplot Comparison - use appropriate thresholds for each model
    # Sum model: threshold 1.0 (where overflow happens)
    # Vbar model: threshold 0.5 (midpoint of 0-1 range)
    # Rad model: threshold 0.5 (midpoint of 0-1 range)
    
    j_sum = [get_galaxy_jump_factor(g[0], g[1], g[2], 'sum', 1.0) for g in galaxy_list]
    j_vbar = [get_galaxy_jump_factor(g[0], g[1], g[2], 'vbar', 0.5) for g in galaxy_list]
    j_rad = [get_galaxy_jump_factor(g[0], g[1], g[2], 'rad', 0.5) for g in galaxy_list]
    
    j_sum = [j for j in j_sum if j is not None]
    j_vbar = [j for j in j_vbar if j is not None]
    j_rad = [j for j in j_rad if j is not None]
    
    plt.figure(figsize=(10, 6))
    plt.boxplot([j_vbar, j_rad, j_sum], tick_labels=['Mass Only\n(t=0.5)', 'Radius Only\n(t=0.5)', 'Mass + Radius\n(t=1.0)'])
    plt.title(f"Model Comparison ({label})")
    plt.ylabel("Velocity Jump Factor (per galaxy)")
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(os.path.dirname(__file__), f"boxplot_{label.replace(' ', '_')}.png")
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    
    # Statistical Significance (Median Difference)
    med_sum = np.median(j_sum) if j_sum else 0
    med_vbar = np.median(j_vbar) if j_vbar else 0
    med_rad = np.median(j_rad) if j_rad else 0
    
    print(f"Median Jump (Sum Model @ t=1.0):    {med_sum:.2f}x  (n={len(j_sum)})")
    print(f"Median Jump (Mass Only @ t=0.5):    {med_vbar:.2f}x  (n={len(j_vbar)})")
    print(f"Median Jump (Radius Only @ t=0.5):  {med_rad:.2f}x  (n={len(j_rad)})")
    print(f"Improvement over Mass-Only: +{med_sum - med_vbar:.2f}")
    print(f"Improvement over Radius-Only: +{med_sum - med_rad:.2f}")

def main():
    # Load ALL data first
    all_files = glob.glob(os.path.join(SPARC_DIR, "*_rotmod.dat"))
    all_names = [os.path.basename(f).replace("_rotmod.dat", "") for f in all_files]
    
    golden_data = []
    other_data = []
    
    print("Loading data...")
    for name in all_names:
        data = load_galaxy_data(os.path.join(SPARC_DIR, f"{name}_rotmod.dat"))
        if data is None: continue
        
        if name in GOLDEN_SET:
            golden_data.append(data)
        else:
            other_data.append(data)
    
    print(f"Loaded {len(golden_data)} Golden Set galaxies")
    print(f"Loaded {len(other_data)} Other SPARC galaxies")
            
    # 1. Analyze Golden Set (In-Sample Validation)
    analyze_population(golden_data, "Golden Set")
    
    # 2. Analyze Others (Out-of-Sample Validation)
    if other_data:
        analyze_population(other_data, "Rest of SPARC")
    else:
        print("\nNo out-of-sample galaxies found.")

if __name__ == "__main__":
    main()
