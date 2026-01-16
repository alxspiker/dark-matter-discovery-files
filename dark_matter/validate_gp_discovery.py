#!/usr/bin/env python3
"""
Paper 2c VALIDATION: Is GP better than a simple cubic?
=======================================================

The reviewer correctly noted:
1. Discovered formula is essentially a cubic in log-log space
2. Need to compare against fitted cubic baseline
3. Need to fit RAR on training data (not use fixed g†)
4. Need multi-seed stability test
5. Physics penalties were too weak

This script runs the critical validation checks.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.optimize import minimize, curve_fit
import json
import warnings
warnings.filterwarnings('ignore')

G_DAGGER = 1.2e-10


def load_sparc(data_dir: str = "sparc_galaxies") -> Dict[str, pd.DataFrame]:
    """Load SPARC data."""
    data_path = Path(data_dir)
    galaxies = {}
    
    for f in sorted(data_path.glob("*.dat")):
        name = f.stem.replace('_rotmod', '')
        try:
            df = pd.read_csv(f, sep=r'\s+', comment='#',
                            names=['R', 'V_obs', 'e_V_obs', 'V_gas', 'V_disk', 'V_bul', 'SBdisk', 'SBbul'],
                            usecols=range(8))
            
            if len(df) < 5:
                continue
            
            df = df[(df['V_obs'] > 0) & (df['R'] > 0)].copy()
            
            V_bar_sq = df['V_gas']**2 + df['V_disk']**2 + df['V_bul']**2
            df['V_bar'] = np.sqrt(np.maximum(V_bar_sq, 1e-10))
            df['g_bar'] = df['V_bar']**2 / (df['R'] * 3.086e19) * 1e6
            df['g_obs'] = df['V_obs']**2 / (df['R'] * 3.086e19) * 1e6
            
            df = df[(df['g_bar'] > 1e-15) & (df['g_obs'] > 1e-15)].copy()
            df['log_g_bar'] = np.log10(df['g_bar'])
            df['log_g_obs'] = np.log10(df['g_obs'])
            
            if len(df) >= 5:
                galaxies[name] = df
        except Exception:
            continue
    
    return galaxies


def split_galaxies(galaxies: Dict[str, pd.DataFrame], seed: int = 42):
    names = sorted(galaxies.keys())
    np.random.seed(seed)
    np.random.shuffle(names)
    n = len(names)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    return names[:n_train], names[n_train:n_train+n_val], names[n_train+n_val:]


def get_data(galaxies, names):
    """Get concatenated data with galaxy slices."""
    x_list, y_list, slices = [], [], []
    idx = 0
    for name in names:
        df = galaxies[name]
        n = len(df)
        x_list.extend(df['log_g_bar'].values)
        y_list.extend(df['log_g_obs'].values)
        slices.append((idx, idx + n))
        idx += n
    return np.array(x_list), np.array(y_list), slices


def galaxy_weighted_mae(pred, true, slices):
    """Galaxy-weighted MAE in log space."""
    total = 0.0
    for s, e in slices:
        if e > s:
            total += np.mean(np.abs(pred[s:e] - true[s:e]))
    return total / len(slices)


# =============================================================================
# BASELINE MODELS
# =============================================================================

def fit_cubic(x_train, y_train, slices_train):
    """Fit cubic polynomial: y = a + bx + cx² + dx³"""
    def objective(params):
        a, b, c, d = params
        pred = a + b*x_train + c*x_train**2 + d*x_train**3
        return galaxy_weighted_mae(pred, y_train, slices_train)
    
    # Initial guess from linear regression
    result = minimize(objective, [0, 1, 0, 0], method='Powell')
    return result.x


def fit_rar_log(x_train, y_train, slices_train):
    """Fit RAR in log space with galaxy-weighted objective."""
    def rar_log(x, log_g_dagger):
        g_bar = 10**x
        g_dagger = 10**log_g_dagger
        g_obs = g_bar / (1 - np.exp(-np.sqrt(g_bar / g_dagger)))
        return np.log10(np.maximum(g_obs, 1e-30))
    
    def objective(params):
        pred = rar_log(x_train, params[0])
        return galaxy_weighted_mae(pred, y_train, slices_train)
    
    result = minimize(objective, [np.log10(G_DAGGER)], method='Powell',
                     bounds=[(-12, -8)])
    return result.x[0]


def fit_quadratic(x_train, y_train, slices_train):
    """Fit quadratic: y = a + bx + cx²"""
    def objective(params):
        a, b, c = params
        pred = a + b*x_train + c*x_train**2
        return galaxy_weighted_mae(pred, y_train, slices_train)
    
    result = minimize(objective, [0, 1, 0], method='Powell')
    return result.x


def fit_linear(x_train, y_train, slices_train):
    """Fit linear: y = a + bx"""
    def objective(params):
        a, b = params
        pred = a + b*x_train
        return galaxy_weighted_mae(pred, y_train, slices_train)
    
    result = minimize(objective, [0, 1], method='Powell')
    return result.x


# =============================================================================
# GP DISCOVERED FORMULA
# =============================================================================

def gp_discovered(x, c1=1.46, c2=-0.00292):
    """The formula GP discovered: y = (c1 + x²*tanh(c2)) * (x + c1)"""
    return (c1 + x**2 * np.tanh(c2)) * (x + c1)


# =============================================================================
# VALIDATION TESTS
# =============================================================================

def test_1_cubic_baseline(galaxies, train_names, val_names, test_names):
    """TEST 1: Does GP beat a simple cubic polynomial?"""
    print("\n" + "="*70)
    print("TEST 1: Cubic Baseline Comparison")
    print("="*70)
    print("If GP is just rediscovering 'flexible curve', cubic should match it.")
    
    x_train, y_train, slices_train = get_data(galaxies, train_names)
    x_val, y_val, slices_val = get_data(galaxies, val_names)
    x_test, y_test, slices_test = get_data(galaxies, test_names)
    
    # Fit cubic on train
    cubic_params = fit_cubic(x_train, y_train, slices_train)
    a, b, c, d = cubic_params
    
    print(f"\nFitted cubic: y = {a:.3f} + {b:.3f}x + {c:.4f}x² + {d:.5f}x³")
    
    # Evaluate all models
    results = {}
    
    # Cubic
    pred_cubic_test = a + b*x_test + c*x_test**2 + d*x_test**3
    mae_cubic = galaxy_weighted_mae(pred_cubic_test, y_test, slices_test)
    results['cubic'] = mae_cubic
    
    # GP discovered
    pred_gp_test = gp_discovered(x_test)
    mae_gp = galaxy_weighted_mae(pred_gp_test, y_test, slices_test)
    results['gp_discovered'] = mae_gp
    
    # Also fit and test quadratic and linear
    lin_params = fit_linear(x_train, y_train, slices_train)
    pred_lin = lin_params[0] + lin_params[1]*x_test
    mae_lin = galaxy_weighted_mae(pred_lin, y_test, slices_test)
    results['linear'] = mae_lin
    
    quad_params = fit_quadratic(x_train, y_train, slices_train)
    pred_quad = quad_params[0] + quad_params[1]*x_test + quad_params[2]*x_test**2
    mae_quad = galaxy_weighted_mae(pred_quad, y_test, slices_test)
    results['quadratic'] = mae_quad
    
    print(f"\nTest MAE (galaxy-weighted, log-space):")
    print(f"  Linear (2 params):     {mae_lin:.4f}")
    print(f"  Quadratic (3 params):  {mae_quad:.4f}")
    print(f"  Cubic (4 params):      {mae_cubic:.4f}")
    print(f"  GP discovered:         {mae_gp:.4f}")
    
    if mae_cubic < mae_gp * 1.02:
        print(f"\n  [!] CUBIC MATCHES GP - GP just found a flexible polynomial!")
    elif mae_gp < mae_cubic:
        print(f"\n  [✓] GP beats cubic by {(mae_cubic/mae_gp - 1)*100:.1f}%")
    
    return results


def test_2_fitted_rar(galaxies, train_names, test_names):
    """TEST 2: Compare to RAR fitted on training data."""
    print("\n" + "="*70)
    print("TEST 2: Fitted RAR Comparison")
    print("="*70)
    print("Fair comparison: fit g† on train, evaluate on test.")
    
    x_train, y_train, slices_train = get_data(galaxies, train_names)
    x_test, y_test, slices_test = get_data(galaxies, test_names)
    
    # Fit RAR
    log_g_dagger_fit = fit_rar_log(x_train, y_train, slices_train)
    g_dagger_fit = 10**log_g_dagger_fit
    
    print(f"\nFitted g† = {g_dagger_fit:.2e} m/s² (canonical: 1.2e-10)")
    
    # Evaluate
    def rar_pred(x, log_gd):
        g_bar = 10**x
        g_dagger = 10**log_gd
        g_obs = g_bar / (1 - np.exp(-np.sqrt(g_bar / g_dagger)))
        return np.log10(np.maximum(g_obs, 1e-30))
    
    pred_rar_fitted = rar_pred(x_test, log_g_dagger_fit)
    mae_rar_fitted = galaxy_weighted_mae(pred_rar_fitted, y_test, slices_test)
    
    pred_rar_fixed = rar_pred(x_test, np.log10(G_DAGGER))
    mae_rar_fixed = galaxy_weighted_mae(pred_rar_fixed, y_test, slices_test)
    
    pred_gp = gp_discovered(x_test)
    mae_gp = galaxy_weighted_mae(pred_gp, y_test, slices_test)
    
    print(f"\nTest MAE:")
    print(f"  RAR (fixed g†=1.2e-10):  {mae_rar_fixed:.4f}")
    print(f"  RAR (fitted g†):         {mae_rar_fitted:.4f}")
    print(f"  GP discovered:           {mae_gp:.4f}")
    
    if mae_rar_fitted < mae_gp * 1.02:
        print(f"\n  [!] FITTED RAR MATCHES GP - GP didn't beat properly trained RAR!")
    elif mae_gp < mae_rar_fitted:
        print(f"\n  [✓] GP beats fitted RAR by {(mae_rar_fitted/mae_gp - 1)*100:.1f}%")
    
    return {
        'rar_fixed': mae_rar_fixed,
        'rar_fitted': mae_rar_fitted,
        'gp': mae_gp,
        'g_dagger_fit': g_dagger_fit
    }


def test_3_multi_seed(galaxies, n_seeds=10):
    """TEST 3: Multi-seed stability."""
    print("\n" + "="*70)
    print(f"TEST 3: Multi-Seed Stability ({n_seeds} seeds)")
    print("="*70)
    print("If GP result is robust, different seeds should find similar formulas.")
    
    # Import the GP module
    import sys
    sys.path.insert(0, '.')
    from grammar_discovery_v2 import (PhysicsGP, prepare_training_data, 
                                       split_galaxies as gp_split, load_sparc_log_space)
    
    results = []
    
    for seed in range(n_seeds):
        print(f"\n  Seed {seed+1}/{n_seeds}...", end=" ", flush=True)
        
        # Different split each seed
        train_names, val_names, test_names = gp_split(galaxies, seed=seed*7 + 42)
        
        x_train, y_train, slices_train = get_data(galaxies, train_names)
        x_test, y_test, slices_test = get_data(galaxies, test_names)
        
        # Run GP with this seed
        import random
        random.seed(seed)
        np.random.seed(seed)
        
        gp = PhysicsGP(pop_size=50, max_depth=4, max_complexity=10, parsimony_coef=0.01)
        
        candidates = gp.evolve(x_train, y_train, slices_train, 
                               n_generations=20, verbose=False)
        
        if candidates:
            best_expr, best_params, train_fitness = candidates[0]
            
            # Evaluate on test
            pred = best_expr.evaluate(x_test, best_params)
            test_mae = galaxy_weighted_mae(pred, y_test, slices_test)
            
            formula = best_expr.to_string(best_params)
            results.append({
                'seed': seed,
                'formula': formula,
                'train_mae': train_fitness,
                'test_mae': test_mae,
                'complexity': best_expr.complexity()
            })
            print(f"MAE={test_mae:.4f}, formula={formula[:40]}...")
    
    # Analyze stability
    test_maes = [r['test_mae'] for r in results]
    formulas = [r['formula'] for r in results]
    
    print(f"\n\nStability Analysis:")
    print(f"  Test MAE: {np.mean(test_maes):.4f} ± {np.std(test_maes):.4f}")
    print(f"  Range: [{np.min(test_maes):.4f}, {np.max(test_maes):.4f}]")
    
    # Check structural similarity
    unique_structures = len(set(formulas))
    print(f"  Unique formulas: {unique_structures}/{len(formulas)}")
    
    if unique_structures == len(formulas):
        print(f"\n  [!] ALL FORMULAS DIFFERENT - GP is unstable!")
    elif np.std(test_maes) > 0.05:
        print(f"\n  [!] HIGH VARIANCE - GP results are not stable")
    else:
        print(f"\n  [✓] GP shows reasonable stability")
    
    return results


def test_4_monotonicity(galaxies, train_names):
    """TEST 4: Check if GP formula is monotonic."""
    print("\n" + "="*70)
    print("TEST 4: Monotonicity Check")
    print("="*70)
    print("Physical requirement: g_obs should increase with g_bar.")
    
    x = np.linspace(-14, -8, 200)
    y_gp = gp_discovered(x)
    
    # Check monotonicity
    dy = np.diff(y_gp)
    n_decreasing = np.sum(dy < 0)
    
    print(f"\n  GP formula y = (1.46 + x²*tanh(-0.003)) * (x + 1.46)")
    print(f"  Points where dy/dx < 0: {n_decreasing}/{len(dy)}")
    
    if n_decreasing > 0:
        # Find where it decreases
        bad_idx = np.where(dy < 0)[0]
        print(f"  Non-monotonic at x (log g_bar) in [{x[bad_idx[0]]:.1f}, {x[bad_idx[-1]+1]:.1f}]")
        print(f"\n  [!] GP FORMULA IS NON-MONOTONIC - physically invalid!")
    else:
        print(f"\n  [✓] GP formula is monotonic")
    
    # Also check RAR
    g_bar = 10**x
    g_obs_rar = g_bar / (1 - np.exp(-np.sqrt(g_bar / G_DAGGER)))
    y_rar = np.log10(g_obs_rar)
    dy_rar = np.diff(y_rar)
    
    print(f"\n  RAR monotonicity check: {np.sum(dy_rar < 0)} violations (should be 0)")
    
    return {
        'gp_violations': int(n_decreasing),
        'rar_violations': int(np.sum(dy_rar < 0))
    }


def test_5_asymptotic_behavior(galaxies):
    """TEST 5: Check asymptotic behavior."""
    print("\n" + "="*70)
    print("TEST 5: Asymptotic Behavior")
    print("="*70)
    print("Physical requirement: dy/dx → 1 at high g_bar (Newtonian limit)")
    
    x = np.linspace(-14, -8, 200)
    y_gp = gp_discovered(x)
    
    # Numerical derivative
    dy_dx = np.gradient(y_gp, x)
    
    print(f"\n  GP formula slope dy/dx:")
    print(f"    At x=-14 (very low g_bar):  {dy_dx[0]:.3f}")
    print(f"    At x=-11 (MOND regime):     {dy_dx[50]:.3f}")
    print(f"    At x=-9 (high g_bar):       {dy_dx[150]:.3f}")
    print(f"    At x=-8 (Newtonian):        {dy_dx[-1]:.3f}")
    
    if abs(dy_dx[-1] - 1.0) > 0.2:
        print(f"\n  [!] GP doesn't approach Newtonian limit (slope should → 1)")
    
    # Compare to RAR
    g_bar = 10**x
    g_obs_rar = g_bar / (1 - np.exp(-np.sqrt(g_bar / G_DAGGER)))
    y_rar = np.log10(g_obs_rar)
    dy_dx_rar = np.gradient(y_rar, x)
    
    print(f"\n  RAR slope dy/dx:")
    print(f"    At x=-14:  {dy_dx_rar[0]:.3f}")
    print(f"    At x=-11:  {dy_dx_rar[50]:.3f}")
    print(f"    At x=-9:   {dy_dx_rar[150]:.3f}")
    print(f"    At x=-8:   {dy_dx_rar[-1]:.3f}")
    
    return {
        'gp_slope_low': float(dy_dx[0]),
        'gp_slope_high': float(dy_dx[-1]),
        'rar_slope_low': float(dy_dx_rar[0]),
        'rar_slope_high': float(dy_dx_rar[-1])
    }


def main():
    print("="*70)
    print("VALIDATION: Is GP Discovery Real or Just Flexible Polynomial?")
    print("="*70)
    
    # Load data
    galaxies = load_sparc()
    print(f"\nLoaded {len(galaxies)} galaxies")
    
    train_names, val_names, test_names = split_galaxies(galaxies)
    print(f"Split: {len(train_names)} train, {len(val_names)} val, {len(test_names)} test")
    
    all_results = {}
    
    # Run tests
    all_results['cubic_comparison'] = test_1_cubic_baseline(galaxies, train_names, val_names, test_names)
    all_results['rar_comparison'] = test_2_fitted_rar(galaxies, train_names, test_names)
    all_results['monotonicity'] = test_4_monotonicity(galaxies, train_names)
    all_results['asymptotic'] = test_5_asymptotic_behavior(galaxies)
    
    # Skip multi-seed for now (slow)
    # all_results['multi_seed'] = test_3_multi_seed(galaxies, n_seeds=5)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    cubic_mae = all_results['cubic_comparison']['cubic']
    gp_mae = all_results['cubic_comparison']['gp_discovered']
    rar_fitted_mae = all_results['rar_comparison']['rar_fitted']
    
    print(f"\nTest MAE (lower is better):")
    print(f"  Cubic (4 params):    {cubic_mae:.4f}")
    print(f"  GP discovered:       {gp_mae:.4f}")
    print(f"  RAR fitted:          {rar_fitted_mae:.4f}")
    
    print(f"\nPhysics checks:")
    print(f"  GP monotonic:        {'YES' if all_results['monotonicity']['gp_violations'] == 0 else 'NO'}")
    print(f"  GP Newtonian limit:  {'YES' if abs(all_results['asymptotic']['gp_slope_high'] - 1.0) < 0.2 else 'NO'}")
    
    if cubic_mae < gp_mae * 1.02:
        print(f"\n[CONCLUSION] GP just found a flexible polynomial - NOT a new law")
    elif rar_fitted_mae < gp_mae * 1.02:
        print(f"\n[CONCLUSION] Fitted RAR matches GP - GP rediscovered RAR-like behavior")
    elif all_results['monotonicity']['gp_violations'] > 0:
        print(f"\n[CONCLUSION] GP formula is unphysical (non-monotonic)")
    else:
        print(f"\n[CONCLUSION] GP may have found something interesting - needs more validation")
    
    # Save
    with open("validation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to validation_results.json")


if __name__ == "__main__":
    main()
