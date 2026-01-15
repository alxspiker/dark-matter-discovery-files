"""
PREDICTIVE BENCHMARK V2 (The "Honest" Test)
===========================================
Addressing Reviewer Critique: "Target Leakage via Parameter Fitting."

Goal: Compare models using strict Train/Test separation per galaxy.

Methodology:
1. Split every galaxy's data points into TRAIN (Even indices) and TEST (Odd indices).
2. Model A: Newtonian (0 parameters). V_pred = V_bar.
3. Model B: Digital Latch (1 parameter). 
   - Determine Overflow mask using INPUTS only (V_bar + R).
   - Fit V_flat using ONLY TRAIN data in the overflow region.
   - Predict V_obs on TEST data using that learned constant.

If Model B beats Model A on the TEST set, the predictive power is real.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import re
import glob

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

def run_honest_benchmark():
    print(f"⚖️ RUNNING 'HONEST' TRAIN/TEST BENCHMARK")
    print(f"   Splitting each galaxy: 50% Train (Fit V_flat) / 50% Test (Eval)")
    
    files = glob.glob(os.path.join(SPARC_DIR, "*_rotmod.dat"))
    
    results = [] # (name, rmse_newton, rmse_latch, improvement)
    
    for f in files:
        name = os.path.basename(f).replace("_rotmod.dat", "")
        data = load_galaxy_data(f)
        if data is None or len(data[0]) < 6: continue # Need points for split
        
        rad, vbar, vobs = data
        n_points = len(rad)
        
        # --- 1. SPLIT DATA ---
        # Interleaved split minimizes radius bias
        train_idx = np.arange(0, n_points, 2)
        test_idx = np.arange(1, n_points, 2)
        
        # --- 2. NEWTONIAN MODEL (0-Parameter) ---
        # Prediction is simply V_bar. No fitting needed.
        # Evaluate on TEST set.
        pred_newton = vbar[test_idx]
        rmse_newton = calculate_rmse(vobs[test_idx], pred_newton)
        
        # --- 3. DIGITAL LATCH MODEL (1-Parameter) ---
        # Step A: Determine Overflow Mask (Inputs Only - No Leakage)
        v_scale = vbar.max() if vbar.max() > 0 else 1.0
        r_scale = rad.max() if rad.max() > 0 else 1.0
        
        # Calculate sum for ALL points first
        digital_sum = (vbar / v_scale) + (rad / r_scale)
        overflow_mask = digital_sum >= 1.0
        
        # Step B: Fit Parameter (V_flat) on TRAIN set only
        train_overflow_mask = overflow_mask[train_idx]
        train_vobs = vobs[train_idx]
        
        # If we have training data in the overflow region, learn the latch value
        if np.sum(train_overflow_mask) > 0:
            v_flat = np.mean(train_vobs[train_overflow_mask])
        else:
            # Fallback: If no overflow in training, default to max V_bar (Newtonian-ish)
            v_flat = vbar.max()
            
        # Step C: Predict on TEST set
        # For test points in overflow: predict V_flat
        # For test points NOT in overflow: predict V_bar (Newtonian)
        test_overflow_mask = overflow_mask[test_idx]
        
        pred_latch = np.copy(vbar[test_idx]) # Default to V_bar
        pred_latch[test_overflow_mask] = v_flat # Apply Latch
        
        rmse_latch = calculate_rmse(vobs[test_idx], pred_latch)
        
        # Store Result
        imp = (rmse_newton - rmse_latch) / rmse_newton * 100
        results.append({
            'name': name,
            'rmse_newton': rmse_newton,
            'rmse_latch': rmse_latch,
            'imp': imp
        })

    # --- AGGREGATE STATS ---
    median_imp = np.median([r['imp'] for r in results])
    mean_imp = np.mean([r['imp'] for r in results])
    wins = sum(1 for r in results if r['rmse_latch'] < r['rmse_newton'])
    total = len(results)
    
    print("\n" + "="*60)
    print("RESULTS ON HELD-OUT DATA (No Leakage)")
    print("="*60)
    print(f"Galaxies Analyzed:      {total}")
    print(f"Digital Latch Wins:     {wins} ({wins/total:.1%})")
    print(f"Median Error Reduction: {median_imp:.1f}%")
    print(f"Mean Error Reduction:   {mean_imp:.1f}%")
    print("="*60)
    
    if wins/total >= 0.8:
        print("\n✅ STRONG RESULT: >80% win rate on held-out data!")
        print("   The Digital Latch has genuine predictive power.")
    elif wins/total >= 0.6:
        print("\n✅ POSITIVE RESULT: >60% win rate on held-out data.")
        print("   The Digital Latch provides moderate improvement.")
    else:
        print("\n⚠️  WEAK RESULT: <60% win rate.")
        print("   The Digital Latch doesn't clearly beat Newtonian.")
    
    # --- PLOT ---
    newton_errs = [r['rmse_newton'] for r in results]
    latch_errs = [r['rmse_latch'] for r in results]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(newton_errs, latch_errs, alpha=0.5, c='blue')
    
    # y=x line
    limit = max(max(newton_errs), max(latch_errs))
    plt.plot([0, limit], [0, limit], 'r--', label="Equal Performance")
    
    plt.title("Out-of-Sample Predictive Error\n(Train on 50%, Test on 50%)")
    plt.xlabel("Newtonian RMSE (km/s)")
    plt.ylabel("Digital Latch RMSE (km/s)")
    plt.text(limit*0.05, limit*0.9, f"Digital Wins: {wins/total:.0%}\nMedian Improvement: {median_imp:.1f}%", 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(os.path.dirname(__file__), "honest_benchmark.png")
    plt.savefig(out_path)
    print(f"\n[SAVED] {out_path}")

if __name__ == "__main__":
    run_honest_benchmark()
