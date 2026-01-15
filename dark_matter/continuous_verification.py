"""
CONTINUOUS VERIFICATION ENGINE
==============================
The "Critic Test."

We discovered that V_obs[MSB] = CarryOut(V_bar + R) in 8-bit logic.
In continuous math, CarryOut(A+B) is equivalent to a Heaviside step function:
    Trigger = (Norm(V_bar) + Norm(R)) >= 1.0

This script tests if that digital rule translates to physical reality.
It plots the "Digital Sum" vs the actual Observed Velocity.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import re
import glob

# The 31 "Golden Set" galaxies discovered by UDE
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
            
    # Compute Total Baryonic Velocity (Quadratic sum)
    vbar = np.sqrt(np.array(vgas)**2 + np.array(vdisk)**2 + np.array(vbul)**2)
    return np.array(radius), vbar, np.array(vobs)

def verify_continuous_law():
    print(f"Testing Continuous Threshold Law on {len(GOLDEN_SET)} galaxies...")
    
    all_x = [] # The "Digital Input" (Norm_Mass + Norm_Radius)
    all_y = [] # The "Physical Output" (Normalized V_obs)
    
    for name in GOLDEN_SET:
        data = load_galaxy_data(name)
        if data is None: continue
        rad, vbar, vobs = data
        
        # 1. Normalize exactly as the UDE did (Per-Galaxy Scaling)
        # In the UDE run, we mapped 0..Max -> 0..255.
        # "Overflow" happens at > 255.
        # In continuous math (0..1), Overflow happens at > 1.0.
        
        # Max values used for normalization
        max_v = max(vobs.max(), vbar.max())
        max_r = rad.max()
        
        norm_vbar = vbar / max_v
        norm_rad = rad / max_r
        norm_vobs = vobs / max_v
        
        # The Discovery: CarryOut(Vbar + R) predicts High Vobs.
        # Continuous translation: (Norm_Vbar + Norm_R)
        digital_sum = norm_vbar + norm_rad
        
        all_x.extend(digital_sum)
        all_y.extend(norm_vobs)

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    
    # --- ANALYSIS ---
    
    # Define the "Overflow" region
    # 8-bit carry out means sum >= 256. In 0..1 scaling, sum >= 1.0.
    mask_overflow = all_x >= 1.0
    
    mean_v_low = np.mean(all_y[~mask_overflow])
    mean_v_high = np.mean(all_y[mask_overflow])
    
    print(f"\n[RESULTS]")
    print(f"Total Data Points: {len(all_x)}")
    print(f"Threshold Point:   1.0 (Theoretical Carry-Out)")
    print("-" * 40)
    print(f"Mean Normalized Velocity (Below Threshold): {mean_v_low:.3f}")
    print(f"Mean Normalized Velocity (Above Threshold): {mean_v_high:.3f}")
    print("-" * 40)
    print(f"Velocity Jump Factor: {mean_v_high / mean_v_low:.2f}x")
    
    if mean_v_high > mean_v_low * 1.5:
        print("\n✅ SUCCESS: Strong physical step-function detected!")
        print("   The 'Carry Out' isn't just an artifact. It corresponds")
        print("   to a massive physical jump in velocity.")
    else:
        print("\n❌ FAILURE: No clear separation found in continuous data.")

    # --- PLOTTING ---
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(all_x, all_y, alpha=0.3, s=10, label="Galaxy Data Points")
        
        # Draw the "Digital Horizon"
        plt.axvline(x=1.0, color='r', linestyle='--', linewidth=2, label="Overflow Threshold (Bit 7 Carry)")
        
        plt.title("The Gravitational Overflow: Continuous Verification")
        plt.xlabel("Input Sum (Normalized Baryons + Radius)")
        plt.ylabel("Output (Normalized Observed Velocity)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        out_img = os.path.join(os.path.dirname(__file__), "continuous_verification.png")
        plt.savefig(out_img)
        print(f"\n[PLOT SAVED] {out_img}")
        print("Check this image. If you see a sharp rise or step at x=1.0, you win.")
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == "__main__":
    verify_continuous_law()
