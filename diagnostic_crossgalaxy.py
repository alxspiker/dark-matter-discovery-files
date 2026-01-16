"""
DIAGNOSTIC: Why do both models have terrible cross-galaxy R²?
============================================================

The issue is likely that galaxies have very different V_obs SCALES
after normalization (normalized to their own max, not a global max).

A model trained on galaxies with V_obs ~ 0.8 will fail on galaxies
with V_obs ~ 0.3, regardless of shape.

Let's check:
1. What's the range of V_obs values across galaxies?
2. Are we predicting shape or scale?
3. What's the right baseline comparison?
"""

import os
import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
import glob
import warnings
warnings.filterwarnings('ignore')

SPARC_DIR = os.path.join(os.path.dirname(__file__), "dark_matter", "sparc_galaxies")

def load_galaxy_data(path):
    import re
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
    """Normalize by vbar scale (so V_obs range varies by galaxy)."""
    v_scale = vbar.max() if vbar.max() > 0 else 1.0
    r_scale = rad.max() if rad.max() > 0 else 1.0
    return rad / r_scale, vbar / v_scale, vobs / v_scale

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot < 1e-10: return 0
    return 1 - ss_res / ss_tot

# Load data
files = glob.glob(os.path.join(SPARC_DIR, "*_rotmod.dat"))
print(f"Loading {len(files)} galaxy files...")

galaxies_data = []
for f in files:
    data = load_galaxy_data(f)
    if data is None or len(data[0]) < 8: continue
    
    rad, vbar, vobs = data
    n_rad, n_vbar, n_vobs = normalize_galaxy(rad, vbar, vobs)
    x = n_vbar + n_rad
    y = n_vobs
    
    name = os.path.basename(f).replace('_rotmod.dat', '')
    galaxies_data.append({
        'name': name,
        'x': x,
        'y': y,
        'y_mean': np.mean(y),
        'y_std': np.std(y),
        'y_min': np.min(y),
        'y_max': np.max(y),
        'x_range': (np.min(x), np.max(x))
    })

print(f"Valid galaxies: {len(galaxies_data)}")

# Diagnostic 1: What's the distribution of y values?
print("\n" + "="*60)
print("DIAGNOSTIC 1: Distribution of normalized V_obs")
print("="*60)

y_means = [g['y_mean'] for g in galaxies_data]
y_stds = [g['y_std'] for g in galaxies_data]
y_maxs = [g['y_max'] for g in galaxies_data]

print(f"Mean of galaxy means: {np.mean(y_means):.3f} ± {np.std(y_means):.3f}")
print(f"Range of galaxy means: [{np.min(y_means):.3f}, {np.max(y_means):.3f}]")
print(f"Mean of galaxy maxes: {np.mean(y_maxs):.3f} ± {np.std(y_maxs):.3f}")

# Diagnostic 2: Are highly-varying y_means the problem?
print("\n" + "="*60)
print("DIAGNOSTIC 2: Correlate y_mean with cross-galaxy prediction error")
print("="*60)

# Quick test: predict each galaxy's y with global mean
global_y = np.concatenate([g['y'] for g in galaxies_data])
global_mean = np.mean(global_y)

r2_vs_global = []
for g in galaxies_data:
    y_pred = np.full_like(g['y'], global_mean)
    r2 = compute_r2(g['y'], y_pred)
    r2_vs_global.append(r2)

print(f"R² predicting with global mean: {np.mean(r2_vs_global):.3f}")
print(f"This shows the 'scale problem' - each galaxy has different mean V_obs")

# Diagnostic 3: What if we de-mean before comparison?
print("\n" + "="*60)
print("DIAGNOSTIC 3: Shape prediction (de-meaned)")
print("="*60)
print("If we remove the LEVEL (mean) from each galaxy and just predict SHAPE...")

def sigmoid(x, a, b, x0, s):
    return a + (b - a) / (1 + np.exp(-np.clip(s * (x - x0), -50, 50)))

# Pool training data (80%)
np.random.seed(42)
np.random.shuffle(galaxies_data)
n_train = int(0.8 * len(galaxies_data))
train_galaxies = galaxies_data[:n_train]
test_galaxies = galaxies_data[n_train:]

train_x = np.concatenate([g['x'] for g in train_galaxies])
train_y = np.concatenate([g['y'] for g in train_galaxies])

# Fit models on training
slope, intercept, _, _, _ = stats.linregress(train_x, train_y)

try:
    popt, _ = curve_fit(sigmoid, train_x, train_y,
                       p0=[0.1, 1.0, 1.0, 3.0],
                       bounds=([-0.5, 0.3, 0.1, 0.1], [1.0, 2.0, 3.0, 50]),
                       maxfev=10000)
    sig_fit = True
except:
    sig_fit = False
    popt = None

print(f"\nPooled models trained on {n_train} galaxies")
print(f"Linear: y = {intercept:.3f} + {slope:.3f}*x")
if sig_fit:
    print(f"Sigmoid: a={popt[0]:.3f}, b={popt[1]:.3f}, x0={popt[2]:.3f}, s={popt[3]:.2f}")

# Test on held-out galaxies with DIFFERENT metrics
print(f"\nTesting on {len(test_galaxies)} held-out galaxies:")

results = []
for g in test_galaxies:
    x, y = g['x'], g['y']
    
    # Standard predictions
    y_pred_lin = intercept + slope * x
    y_pred_sig = sigmoid(x, *popt) if sig_fit else y_pred_lin
    
    # Standard R²
    r2_lin = compute_r2(y, y_pred_lin)
    r2_sig = compute_r2(y, y_pred_sig)
    
    # De-meaned R² (shape only)
    y_dm = y - np.mean(y)
    y_pred_lin_dm = y_pred_lin - np.mean(y_pred_lin)
    y_pred_sig_dm = y_pred_sig - np.mean(y_pred_sig)
    
    r2_lin_dm = compute_r2(y_dm, y_pred_lin_dm)
    r2_sig_dm = compute_r2(y_dm, y_pred_sig_dm)
    
    # Scale+shift corrected
    def best_scale_shift(y_true, y_pred):
        """Find best a, b such that a + b*y_pred matches y_true."""
        if np.std(y_pred) < 1e-10:
            return np.mean(y_true), 1.0
        slope, intercept, _, _, _ = stats.linregress(y_pred, y_true)
        return intercept, slope
    
    a_lin, b_lin = best_scale_shift(y, y_pred_lin)
    y_pred_lin_adj = a_lin + b_lin * y_pred_lin
    r2_lin_adj = compute_r2(y, y_pred_lin_adj)
    
    a_sig, b_sig = best_scale_shift(y, y_pred_sig)
    y_pred_sig_adj = a_sig + b_sig * y_pred_sig
    r2_sig_adj = compute_r2(y, y_pred_sig_adj)
    
    results.append({
        'name': g['name'],
        'y_mean': g['y_mean'],
        'r2_lin': r2_lin,
        'r2_sig': r2_sig,
        'r2_lin_dm': r2_lin_dm,
        'r2_sig_dm': r2_sig_dm,
        'r2_lin_adj': r2_lin_adj,
        'r2_sig_adj': r2_sig_adj
    })

# Summarize
print("\n--- Standard R² (level + shape) ---")
print(f"Linear:  {np.mean([r['r2_lin'] for r in results]):.3f}")
print(f"Sigmoid: {np.mean([r['r2_sig'] for r in results]):.3f}")

print("\n--- De-meaned R² (shape only) ---")
print(f"Linear:  {np.mean([r['r2_lin_dm'] for r in results]):.3f}")
print(f"Sigmoid: {np.mean([r['r2_sig_dm'] for r in results]):.3f}")

print("\n--- Scale-adjusted R² (best linear transform) ---")
print(f"Linear:  {np.mean([r['r2_lin_adj'] for r in results]):.3f}")
print(f"Sigmoid: {np.mean([r['r2_sig_adj'] for r in results]):.3f}")

# Interpretation
print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

mean_r2_sig = np.mean([r['r2_sig'] for r in results])
mean_r2_sig_adj = np.mean([r['r2_sig_adj'] for r in results])

if mean_r2_sig < 0 and mean_r2_sig_adj > 0:
    print("""
The problem is SCALE, not SHAPE:
- Raw R² is terrible because galaxies have different V_obs levels
- Scale-adjusted R² is reasonable because the SHAPE transfers

This means:
- The sigmoid describes the SHAPE of the transition correctly
- But the LEVEL (asymptotes a, b) varies by galaxy
- A hierarchical model with per-galaxy scale would work
""")
elif mean_r2_sig_adj < 0:
    print("""
The shape itself doesn't transfer - not just a scale issue.
The universal sigmoid fails fundamentally.
""")
else:
    print("""
Universal sigmoid works reasonably well.
""")

# Final check: correlation between shape R² and galaxy properties
print("\n" + "="*60)
print("FINAL: Which galaxies does the shape transfer to?")
print("="*60)

r2_adj = np.array([r['r2_sig_adj'] for r in results])
y_means_test = np.array([r['y_mean'] for r in results])

r, p = stats.pearsonr(r2_adj, y_means_test)
print(f"Correlation(R²_adj, galaxy_mean_Vobs): r={r:.3f}, p={p:.3f}")

# How many have good shape transfer?
n_good = np.sum(r2_adj > 0.5)
print(f"Galaxies with R²_adj > 0.5: {n_good}/{len(results)} ({100*n_good/len(results):.1f}%)")
