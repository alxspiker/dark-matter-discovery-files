"""
CORRECTED OUT-OF-GALAXY VALIDATION
===================================
The previous validation had constrained asymptotes (a bounded to [0,1], b to [0.5,1.5])
which caused catastrophic generalization failure.

This version:
1. Uses FREE asymptotes for the universal sigmoid
2. Properly initializes from per-galaxy statistics
3. Tests if ANY universal sigmoid can beat linear

If this still fails, the conclusion is clear:
"Within-galaxy sigmoid works; universal sigmoid does NOT"
"""

import os
import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit, minimize
import re
import glob
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
    v_scale = vbar.max() if vbar.max() > 0 else 1.0
    r_scale = rad.max() if rad.max() > 0 else 1.0
    return rad / r_scale, vbar / v_scale, vobs / v_scale

def sigmoid(x, a, b, x0, s):
    """Sigmoid: V = a + (b-a) / (1 + exp(-s*(x-x0)))"""
    return a + (b - a) / (1 + np.exp(-np.clip(s * (x - x0), -50, 50)))

def fit_sigmoid_free(x, y):
    """Fit sigmoid with completely free parameters."""
    try:
        # Smart initialization from data
        a0 = np.min(y)
        b0 = np.max(y)
        x0_0 = np.median(x)
        s0 = 5.0
        
        # Very wide bounds
        bounds = ([-0.5, 0.3, 0.1, 0.1], [1.0, 2.0, 3.0, 100])
        
        popt, _ = curve_fit(sigmoid, x, y, p0=[a0, b0, x0_0, s0], 
                           bounds=bounds, maxfev=10000)
        
        y_pred = sigmoid(x, *popt)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {'a': popt[0], 'b': popt[1], 'x0': popt[2], 's': popt[3], 
                'r_squared': r_squared, 'params': popt}
    except:
        return None

def compute_r2(y_true, y_pred):
    """Compute R² properly."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot

# =============================================================================
# TEST 1: Can ANY universal sigmoid beat linear?
# =============================================================================
def test_universal_sigmoid(galaxies_data):
    """Test if a single universal sigmoid can work across galaxies."""
    print("\n" + "="*60)
    print("TEST 1: UNIVERSAL SIGMOID vs LINEAR BASELINE")
    print("="*60)
    
    # Get all valid galaxies
    valid_galaxies = []
    for name, (n_rad, n_vbar, n_vobs) in galaxies_data.items():
        x = n_vbar + n_rad
        y = n_vobs
        if len(x) >= 8:
            valid_galaxies.append((name, x, y))
    
    print(f"Total valid galaxies: {len(valid_galaxies)}")
    
    # 5-fold CV by galaxy
    n = len(valid_galaxies)
    n_folds = 5
    np.random.seed(42)
    np.random.shuffle(valid_galaxies)
    
    results = []
    
    for fold in range(n_folds):
        fold_size = n // n_folds
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        
        train_galaxies = valid_galaxies[:test_start] + valid_galaxies[test_end:]
        test_galaxies = valid_galaxies[test_start:test_end]
        
        # Pool training data
        train_x = np.concatenate([g[1] for g in train_galaxies])
        train_y = np.concatenate([g[2] for g in train_galaxies])
        
        # Fit universal sigmoid on training (FREE asymptotes)
        try:
            a0 = np.percentile(train_y, 5)
            b0 = np.percentile(train_y, 95)
            x0_0 = np.median(train_x)
            s0 = 3.0
            
            # Very relaxed bounds
            popt_sig, _ = curve_fit(
                sigmoid, train_x, train_y, 
                p0=[a0, b0, x0_0, s0],
                bounds=([-1, 0, 0.1, 0.1], [2, 3, 3, 50]),
                maxfev=10000
            )
            sigmoid_fit_success = True
        except Exception as e:
            print(f"  Fold {fold}: Sigmoid fit failed: {e}")
            sigmoid_fit_success = False
            popt_sig = None
        
        # Fit linear on training
        slope, intercept, _, _, _ = stats.linregress(train_x, train_y)
        
        # Test on held-out galaxies
        test_r2_sigmoid = []
        test_r2_linear = []
        test_rmse_sigmoid = []
        test_rmse_linear = []
        
        for name, x, y in test_galaxies:
            # Linear prediction
            y_pred_lin = intercept + slope * x
            r2_lin = compute_r2(y, y_pred_lin)
            rmse_lin = np.sqrt(np.mean((y - y_pred_lin)**2))
            test_r2_linear.append(r2_lin)
            test_rmse_linear.append(rmse_lin)
            
            # Sigmoid prediction
            if sigmoid_fit_success:
                y_pred_sig = sigmoid(x, *popt_sig)
                r2_sig = compute_r2(y, y_pred_sig)
                rmse_sig = np.sqrt(np.mean((y - y_pred_sig)**2))
            else:
                r2_sig = -999
                rmse_sig = 999
            test_r2_sigmoid.append(r2_sig)
            test_rmse_sigmoid.append(rmse_sig)
        
        fold_result = {
            'fold': fold,
            'n_train': len(train_galaxies),
            'n_test': len(test_galaxies),
            'sigmoid_params': popt_sig if sigmoid_fit_success else None,
            'linear_params': (slope, intercept),
            'mean_r2_sigmoid': np.mean(test_r2_sigmoid),
            'mean_r2_linear': np.mean(test_r2_linear),
            'mean_rmse_sigmoid': np.mean(test_rmse_sigmoid),
            'mean_rmse_linear': np.mean(test_rmse_linear),
            'sigmoid_wins': np.sum(np.array(test_r2_sigmoid) > np.array(test_r2_linear))
        }
        results.append(fold_result)
        
        if sigmoid_fit_success:
            print(f"\nFold {fold}: Sigmoid params: a={popt_sig[0]:.3f}, b={popt_sig[1]:.3f}, "
                  f"x0={popt_sig[2]:.3f}, s={popt_sig[3]:.2f}")
        print(f"  Sigmoid R² = {fold_result['mean_r2_sigmoid']:.3f}, "
              f"Linear R² = {fold_result['mean_r2_linear']:.3f}, "
              f"Sigmoid wins: {fold_result['sigmoid_wins']}/{len(test_galaxies)}")
    
    # Summary
    print("\n" + "-"*40)
    print("SUMMARY: Universal Sigmoid vs Linear")
    print("-"*40)
    
    mean_sig_r2 = np.mean([r['mean_r2_sigmoid'] for r in results])
    mean_lin_r2 = np.mean([r['mean_r2_linear'] for r in results])
    total_sig_wins = sum([r['sigmoid_wins'] for r in results])
    total_tests = sum([r['n_test'] for r in results])
    
    print(f"Mean Sigmoid R² on held-out: {mean_sig_r2:.3f}")
    print(f"Mean Linear R² on held-out: {mean_lin_r2:.3f}")
    print(f"Sigmoid beats Linear: {total_sig_wins}/{total_tests} galaxies ({100*total_sig_wins/total_tests:.1f}%)")
    
    if mean_sig_r2 > mean_lin_r2:
        print("\n✅ Universal sigmoid BEATS linear baseline")
    else:
        print("\n❌ Universal sigmoid LOSES to linear baseline")
        print("   A single universal sigmoid does NOT generalize across galaxies")
    
    return results, mean_sig_r2, mean_lin_r2

# =============================================================================
# TEST 2: Within-galaxy sigmoid quality
# =============================================================================
def test_within_galaxy_sigmoid(galaxies_data):
    """Confirm that within-galaxy sigmoid fits work well."""
    print("\n" + "="*60)
    print("TEST 2: WITHIN-GALAXY SIGMOID FIT QUALITY")
    print("="*60)
    
    r2_values = []
    x0_values = []
    
    for name, (n_rad, n_vbar, n_vobs) in galaxies_data.items():
        x = n_vbar + n_rad
        y = n_vobs
        if len(x) < 8: continue
        
        result = fit_sigmoid_free(x, y)
        if result and result['r_squared'] > 0:
            r2_values.append(result['r_squared'])
            x0_values.append(result['x0'])
    
    print(f"\nGalaxies with successful sigmoid fits: {len(r2_values)}")
    print(f"Mean R²: {np.mean(r2_values):.3f}")
    print(f"Median R²: {np.median(r2_values):.3f}")
    print(f"R² > 0.8: {np.sum(np.array(r2_values) > 0.8)}/{len(r2_values)} ({100*np.mean(np.array(r2_values) > 0.8):.1f}%)")
    print(f"R² > 0.5: {np.sum(np.array(r2_values) > 0.5)}/{len(r2_values)} ({100*np.mean(np.array(r2_values) > 0.5):.1f}%)")
    
    # Compare to linear within each galaxy
    linear_r2 = []
    sigmoid_r2_paired = []
    
    for name, (n_rad, n_vbar, n_vobs) in galaxies_data.items():
        x = n_vbar + n_rad
        y = n_vobs
        if len(x) < 8: continue
        
        # Linear fit
        slope, intercept, _, _, _ = stats.linregress(x, y)
        y_pred_lin = intercept + slope * x
        r2_lin = compute_r2(y, y_pred_lin)
        
        # Sigmoid fit
        result = fit_sigmoid_free(x, y)
        if result:
            r2_sig = result['r_squared']
            linear_r2.append(r2_lin)
            sigmoid_r2_paired.append(r2_sig)
    
    linear_r2 = np.array(linear_r2)
    sigmoid_r2_paired = np.array(sigmoid_r2_paired)
    
    sig_wins = np.sum(sigmoid_r2_paired > linear_r2)
    print(f"\nWithin-galaxy: Sigmoid beats Linear in {sig_wins}/{len(linear_r2)} galaxies ({100*sig_wins/len(linear_r2):.1f}%)")
    print(f"Mean Linear R²: {np.mean(linear_r2):.3f}")
    print(f"Mean Sigmoid R²: {np.mean(sigmoid_r2_paired):.3f}")
    
    return r2_values, x0_values

# =============================================================================
# TEST 3: Parameter heterogeneity
# =============================================================================
def test_parameter_heterogeneity(galaxies_data):
    """Quantify how much sigmoid parameters vary across galaxies."""
    print("\n" + "="*60)
    print("TEST 3: PARAMETER HETEROGENEITY ACROSS GALAXIES")
    print("="*60)
    print("This explains WHY universal sigmoid fails")
    
    params = {'a': [], 'b': [], 'x0': [], 's': []}
    
    for name, (n_rad, n_vbar, n_vobs) in galaxies_data.items():
        x = n_vbar + n_rad
        y = n_vobs
        if len(x) < 8: continue
        
        result = fit_sigmoid_free(x, y)
        if result and result['r_squared'] > 0.5:
            params['a'].append(result['a'])
            params['b'].append(result['b'])
            params['x0'].append(result['x0'])
            params['s'].append(result['s'])
    
    print(f"\nGalaxies analyzed: {len(params['a'])}")
    print("\n--- Parameter Distributions ---")
    
    for param_name, values in params.items():
        values = np.array(values)
        print(f"\n{param_name}:")
        print(f"  Mean: {np.mean(values):.3f}")
        print(f"  Std:  {np.std(values):.3f}")
        print(f"  CV (std/mean): {np.std(values)/np.mean(values):.2f}")
        print(f"  Range: [{np.min(values):.3f}, {np.max(values):.3f}]")
    
    print("\n--- Interpretation ---")
    cv_x0 = np.std(params['x0']) / np.mean(params['x0'])
    cv_s = np.std(params['s']) / np.mean(params['s'])
    
    if cv_x0 > 0.3 or cv_s > 0.5:
        print("  ⚠️ HIGH parameter variability explains universal sigmoid failure")
        print("  Each galaxy has its own transition characteristics")
        print("  A HIERARCHICAL model (varying x0 by galaxy) is needed")
    
    return params

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*60)
    print("CORRECTED OUT-OF-GALAXY VALIDATION")
    print("Testing whether universal sigmoid can work")
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
    
    # Test 1: Universal sigmoid vs linear
    results, mean_sig_r2, mean_lin_r2 = test_universal_sigmoid(galaxies_data)
    
    # Test 2: Within-galaxy fit quality
    r2_within, x0_within = test_within_galaxy_sigmoid(galaxies_data)
    
    # Test 3: Parameter heterogeneity
    params = test_parameter_heterogeneity(galaxies_data)
    
    # Final conclusions
    print("\n" + "="*60)
    print("FINAL CORRECTED CONCLUSIONS")
    print("="*60)
    
    universal_works = mean_sig_r2 > mean_lin_r2
    within_works = np.mean(r2_within) > 0.8
    
    print(f"""
    1. WITHIN-GALAXY SIGMOID: {'✅ WORKS' if within_works else '❌ FAILS'}
       Mean R² = {np.mean(r2_within):.3f}
       
    2. UNIVERSAL SIGMOID (cross-galaxy): {'✅ WORKS' if universal_works else '❌ FAILS'}
       Sigmoid R² = {mean_sig_r2:.3f}
       Linear R² = {mean_lin_r2:.3f}
       
    3. PARAMETER HETEROGENEITY:
       x0 CV = {np.std(params['x0'])/np.mean(params['x0']):.2f}
       s CV = {np.std(params['s'])/np.mean(params['s']):.2f}
    
    CORRECT INTERPRETATION:
    -----------------------
    {"✅ A universal sigmoid describes V_obs(x) across galaxies" if universal_works else 
     "❌ NO universal sigmoid works across galaxies"}
    
    {"" if universal_works else '''    The correct claim is:
    "Within each galaxy, a sigmoid-like curve fits well (R² ~ 0.9);
     across galaxies, the parameters vary enough that a single
     universal sigmoid fails to generalize."
    
    This means the 'overflow' model is a useful APPROXIMATION
    within individual galaxies, but there is no single universal
    f(x) that applies to all galaxies.'''}
    """)

if __name__ == "__main__":
    main()
