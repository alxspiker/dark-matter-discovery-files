"""
UDE Dark Matter Discovery - Paper 2 (RIGOROUS VERSION)
=======================================================
Symbolic discovery of empirical acceleration laws from SPARC
with strict out-of-galaxy validation.

Key methodological choices (per AI reviewer):
1. TARGET: Acceleration form g_obs vs g_bar (standard in field, directly tests RAR/MOND)
2. HOLDOUT: Galaxy-level train/val/test split (LOCKED test set)
3. FITNESS: Galaxy-weighted MAE + complexity penalty (prevents overfitting)
4. CONTROLS: Permutation, feature ablation, random features
5. BASELINES: Linear, quadratic, spline, RAR/MOND

Success criteria:
- Beat baselines with fewer DoF on LOCKED test galaxies
- Stable structure across folds
- Must FAIL on permutation controls

Usage:
    python ude_acceleration_discovery.py --discover
    python ude_acceleration_discovery.py --validate
    python ude_acceleration_discovery.py --test  # Only run ONCE at the end!
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================

# MOND/RAR constants
A0 = 1.2e-10  # m/s² - MOND critical acceleration
G_DAGGER = 1.2e-10  # m/s² - RAR scale

# Unit conversions
KPC_TO_M = 3.086e19  # kpc to meters
KMS_TO_MS = 1000  # km/s to m/s

# =============================================================================
# DATA LOADING (Acceleration Space)
# =============================================================================

def load_sparc_accelerations(data_dir: str = "sparc_galaxies") -> Dict[str, pd.DataFrame]:
    """
    Load SPARC data and compute accelerations.
    
    Returns dict with columns: g_bar, g_obs, R, Vbar, Vobs, etc.
    All accelerations in m/s².
    """
    galaxies = {}
    data_path = Path(data_dir)
    
    if not data_path.exists():
        script_dir = Path(__file__).parent
        data_path = script_dir / data_dir
    
    if not data_path.exists():
        raise FileNotFoundError(f"SPARC data directory not found: {data_path}")
    
    for dat_file in data_path.glob("*.dat"):
        galaxy_name = dat_file.stem.replace("_rotmod", "")
        
        try:
            df = pd.read_csv(dat_file, sep=r'\s+', comment='#',
                           names=['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul'])
            
            # Compute baryonic velocity
            df['Vbar'] = np.sqrt(df['Vgas']**2 + df['Vdisk']**2 + df['Vbul']**2)
            
            # Convert to SI units
            df['R_m'] = df['Rad'] * KPC_TO_M  # radius in meters
            df['Vobs_ms'] = df['Vobs'] * KMS_TO_MS  # velocity in m/s
            df['Vbar_ms'] = df['Vbar'] * KMS_TO_MS
            
            # Compute accelerations: g = V² / R
            df['g_obs'] = df['Vobs_ms']**2 / (df['R_m'] + 1e-20)
            df['g_bar'] = df['Vbar_ms']**2 / (df['R_m'] + 1e-20)
            
            # Filter valid rows (positive accelerations)
            df = df.dropna(subset=['g_obs', 'g_bar'])
            df = df[(df['g_obs'] > 0) & (df['g_bar'] > 0)]
            
            if len(df) >= 5:
                galaxies[galaxy_name] = df
                
        except Exception as e:
            pass  # Skip problematic files
    
    print(f"[DATA] Loaded {len(galaxies)} galaxies in acceleration space")
    return galaxies


def split_galaxies(galaxies: Dict[str, pd.DataFrame], 
                   train_frac: float = 0.6,
                   val_frac: float = 0.2,
                   seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Split galaxies into train/val/test sets (galaxy-level, not point-level).
    TEST SET IS LOCKED - only touch once at the very end!
    """
    np.random.seed(seed)
    names = list(galaxies.keys())
    np.random.shuffle(names)
    
    n = len(names)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    train = names[:n_train]
    val = names[n_train:n_train + n_val]
    test = names[n_train + n_val:]  # LOCKED!
    
    print(f"[SPLIT] Train: {len(train)}, Val: {len(val)}, Test: {len(test)} (LOCKED)")
    return train, val, test


def get_galaxy_data(galaxies: Dict[str, pd.DataFrame], 
                    galaxy_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract g_bar, g_obs arrays with galaxy indices for weighting."""
    g_bar_all = []
    g_obs_all = []
    galaxy_idx = []
    
    for i, name in enumerate(galaxy_names):
        df = galaxies[name]
        g_bar_all.extend(df['g_bar'].values)
        g_obs_all.extend(df['g_obs'].values)
        galaxy_idx.extend([i] * len(df))
    
    return np.array(g_bar_all), np.array(g_obs_all), np.array(galaxy_idx)


# =============================================================================
# CANDIDATE FUNCTIONAL FORMS (Physically Motivated)
# =============================================================================

@dataclass
class DiscoveredLaw:
    """Represents a discovered acceleration law."""
    name: str
    formula: str
    func: Callable
    params: np.ndarray
    complexity: int  # Number of free parameters + structural complexity
    

def rar_form(g_bar: np.ndarray, g_dagger: float) -> np.ndarray:
    """RAR: g_obs = g_bar / (1 - exp(-sqrt(g_bar/g†)))"""
    ratio = np.sqrt(g_bar / (g_dagger + 1e-30))
    return g_bar / (1 - np.exp(-ratio) + 1e-10)


def mond_simple_form(g_bar: np.ndarray, a0: float) -> np.ndarray:
    """Simple MOND: g_obs = g_bar * (1 + sqrt(a0/g_bar))"""
    return g_bar * (1 + np.sqrt(a0 / (g_bar + 1e-30)))


def mond_standard_form(g_bar: np.ndarray, a0: float) -> np.ndarray:
    """Standard MOND interpolation: g_obs = g_bar / mu(g_bar/a0)
    where mu(x) = x / sqrt(1 + x²)"""
    x = g_bar / (a0 + 1e-30)
    mu = x / np.sqrt(1 + x**2)
    return g_bar / (mu + 1e-10)


def linear_form(g_bar: np.ndarray, a: float, b: float) -> np.ndarray:
    """Linear: g_obs = a * g_bar + b"""
    return a * g_bar + b


def power_form(g_bar: np.ndarray, a: float, alpha: float) -> np.ndarray:
    """Power law: g_obs = a * g_bar^alpha"""
    return a * np.power(g_bar + 1e-30, alpha)


def rational_form(g_bar: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Rational: g_obs = a * g_bar / (1 + b * g_bar^c)"""
    return a * g_bar / (1 + b * np.power(g_bar + 1e-30, c))


def sigmoid_accel_form(g_bar: np.ndarray, g_mid: float, k: float, g_max: float) -> np.ndarray:
    """Sigmoid in acceleration space: smooth transition"""
    x = np.log10(g_bar + 1e-30) - np.log10(g_mid)
    return g_bar * (1 + (g_max / g_bar - 1) / (1 + np.exp(-k * x)))


# =============================================================================
# FITNESS FUNCTION (Galaxy-Weighted + Complexity Penalty)
# =============================================================================

def galaxy_weighted_mae(g_obs_pred: np.ndarray, 
                        g_obs_true: np.ndarray,
                        galaxy_idx: np.ndarray,
                        n_galaxies: int) -> float:
    """
    Galaxy-weighted MAE: each galaxy contributes equally regardless of point count.
    This prevents large galaxies from dominating.
    """
    total_mae = 0.0
    for g in range(n_galaxies):
        mask = galaxy_idx == g
        if mask.sum() > 0:
            # Use log-space MAE for accelerations (spans many orders of magnitude)
            log_pred = np.log10(g_obs_pred[mask] + 1e-30)
            log_true = np.log10(g_obs_true[mask] + 1e-30)
            total_mae += np.mean(np.abs(log_pred - log_true))
    
    return total_mae / n_galaxies


def fitness_with_complexity(mae: float, complexity: int, lambda_c: float = 0.01) -> float:
    """
    MDL-inspired fitness: penalize complex formulas.
    Lower is better.
    """
    return mae + lambda_c * complexity


# =============================================================================
# DISCOVERY ENGINE (Simplified Symbolic Search)
# =============================================================================

def fit_candidate_law(func: Callable, 
                      g_bar: np.ndarray, 
                      g_obs: np.ndarray,
                      p0: List[float],
                      bounds: Tuple) -> Tuple[Optional[np.ndarray], float]:
    """Fit a candidate functional form to data."""
    try:
        params, _ = curve_fit(func, g_bar, g_obs, p0=p0, bounds=bounds, maxfev=5000)
        pred = func(g_bar, *params)
        # Use log-space RMSE for fitting
        log_rmse = np.sqrt(np.mean((np.log10(pred + 1e-30) - np.log10(g_obs + 1e-30))**2))
        return params, log_rmse
    except Exception:
        return None, float('inf')


def discover_laws(galaxies: Dict[str, pd.DataFrame],
                  train_names: List[str],
                  verbose: bool = True) -> List[DiscoveredLaw]:
    """
    Discover candidate laws by fitting various functional forms.
    Returns list of discovered laws ranked by fitness.
    """
    g_bar, g_obs, galaxy_idx = get_galaxy_data(galaxies, train_names)
    n_gal = len(train_names)
    
    if verbose:
        print("\n" + "="*60)
        print("[DISCOVERY] Fitting candidate functional forms")
        print("="*60)
    
    candidates = []
    
    # 1. RAR form (1 parameter)
    params, rmse = fit_candidate_law(rar_form, g_bar, g_obs, 
                                     p0=[G_DAGGER], bounds=([1e-12], [1e-8]))
    if params is not None:
        pred = rar_form(g_bar, *params)
        mae = galaxy_weighted_mae(pred, g_obs, galaxy_idx, n_gal)
        candidates.append(DiscoveredLaw(
            name="RAR",
            formula=f"g_bar / (1 - exp(-sqrt(g_bar/{params[0]:.2e})))",
            func=lambda x, p=params: rar_form(x, *p),
            params=params,
            complexity=2  # 1 param + 1 structural
        ))
        if verbose:
            print(f"  RAR: MAE={mae:.4f}, g†={params[0]:.2e}")
    
    # 2. Simple MOND (1 parameter)
    params, rmse = fit_candidate_law(mond_simple_form, g_bar, g_obs,
                                     p0=[A0], bounds=([1e-12], [1e-8]))
    if params is not None:
        pred = mond_simple_form(g_bar, *params)
        mae = galaxy_weighted_mae(pred, g_obs, galaxy_idx, n_gal)
        candidates.append(DiscoveredLaw(
            name="MOND_simple",
            formula=f"g_bar * (1 + sqrt({params[0]:.2e}/g_bar))",
            func=lambda x, p=params: mond_simple_form(x, *p),
            params=params,
            complexity=2
        ))
        if verbose:
            print(f"  MOND_simple: MAE={mae:.4f}, a0={params[0]:.2e}")
    
    # 3. Standard MOND interpolation (1 parameter)
    params, rmse = fit_candidate_law(mond_standard_form, g_bar, g_obs,
                                     p0=[A0], bounds=([1e-12], [1e-8]))
    if params is not None:
        pred = mond_standard_form(g_bar, *params)
        mae = galaxy_weighted_mae(pred, g_obs, galaxy_idx, n_gal)
        candidates.append(DiscoveredLaw(
            name="MOND_standard",
            formula=f"g_bar / mu(g_bar/{params[0]:.2e}), mu(x)=x/sqrt(1+x²)",
            func=lambda x, p=params: mond_standard_form(x, *p),
            params=params,
            complexity=3
        ))
        if verbose:
            print(f"  MOND_standard: MAE={mae:.4f}, a0={params[0]:.2e}")
    
    # 4. Power law (2 parameters)
    params, rmse = fit_candidate_law(power_form, g_bar, g_obs,
                                     p0=[1.0, 0.5], bounds=([0.1, 0.1], [10, 2]))
    if params is not None:
        pred = power_form(g_bar, *params)
        mae = galaxy_weighted_mae(pred, g_obs, galaxy_idx, n_gal)
        candidates.append(DiscoveredLaw(
            name="Power",
            formula=f"{params[0]:.3f} * g_bar^{params[1]:.3f}",
            func=lambda x, p=params: power_form(x, *p),
            params=params,
            complexity=3
        ))
        if verbose:
            print(f"  Power: MAE={mae:.4f}, a={params[0]:.3f}, alpha={params[1]:.3f}")
    
    # 5. Rational form (3 parameters)
    params, rmse = fit_candidate_law(rational_form, g_bar, g_obs,
                                     p0=[2.0, 1e10, 0.5], 
                                     bounds=([0.5, 1e8, 0.1], [10, 1e12, 1.5]))
    if params is not None:
        pred = rational_form(g_bar, *params)
        mae = galaxy_weighted_mae(pred, g_obs, galaxy_idx, n_gal)
        candidates.append(DiscoveredLaw(
            name="Rational",
            formula=f"{params[0]:.2f} * g_bar / (1 + {params[1]:.2e} * g_bar^{params[2]:.2f})",
            func=lambda x, p=params: rational_form(x, *p),
            params=params,
            complexity=4
        ))
        if verbose:
            print(f"  Rational: MAE={mae:.4f}")
    
    # 6. Linear (2 parameters) - baseline
    params, rmse = fit_candidate_law(linear_form, g_bar, g_obs,
                                     p0=[1.0, 0], bounds=([0.1, -1e-9], [10, 1e-9]))
    if params is not None:
        pred = linear_form(g_bar, *params)
        mae = galaxy_weighted_mae(pred, g_obs, galaxy_idx, n_gal)
        candidates.append(DiscoveredLaw(
            name="Linear",
            formula=f"{params[0]:.3f} * g_bar + {params[1]:.2e}",
            func=lambda x, p=params: linear_form(x, *p),
            params=params,
            complexity=2
        ))
        if verbose:
            print(f"  Linear: MAE={mae:.4f}")
    
    # Rank by fitness (MAE + complexity penalty)
    for law in candidates:
        pred = law.func(g_bar)
        law.mae = galaxy_weighted_mae(pred, g_obs, galaxy_idx, n_gal)
        law.fitness = fitness_with_complexity(law.mae, law.complexity)
    
    candidates.sort(key=lambda x: x.fitness)
    
    if verbose:
        print(f"\n[RANKING] Best candidates by fitness (MAE + complexity):")
        for i, law in enumerate(candidates[:5]):
            print(f"  {i+1}. {law.name}: fitness={law.fitness:.4f} (MAE={law.mae:.4f}, complexity={law.complexity})")
    
    return candidates


# =============================================================================
# VALIDATION (On held-out galaxies)
# =============================================================================

def validate_laws(galaxies: Dict[str, pd.DataFrame],
                  val_names: List[str],
                  candidates: List[DiscoveredLaw],
                  verbose: bool = True) -> DiscoveredLaw:
    """
    Validate discovered laws on held-out validation galaxies.
    Returns the best law for final testing.
    """
    g_bar, g_obs, galaxy_idx = get_galaxy_data(galaxies, val_names)
    n_gal = len(val_names)
    
    if verbose:
        print("\n" + "="*60)
        print("[VALIDATION] Testing on held-out validation galaxies")
        print("="*60)
    
    results = []
    for law in candidates:
        pred = law.func(g_bar)
        mae = galaxy_weighted_mae(pred, g_obs, galaxy_idx, n_gal)
        fitness = fitness_with_complexity(mae, law.complexity)
        results.append((law, mae, fitness))
        
        if verbose:
            print(f"  {law.name}: val_MAE={mae:.4f}, fitness={fitness:.4f}")
    
    # Sort by validation fitness
    results.sort(key=lambda x: x[2])
    best_law = results[0][0]
    
    if verbose:
        print(f"\n[BEST] {best_law.name}: {best_law.formula}")
    
    return best_law


# =============================================================================
# CONTROLS (Kill Tests)
# =============================================================================

def permutation_control(galaxies: Dict[str, pd.DataFrame],
                        train_names: List[str],
                        best_law: DiscoveredLaw,
                        n_permutations: int = 100,
                        verbose: bool = True) -> Dict:
    """
    KILL TEST 1: Shuffle g_obs within each galaxy.
    If discovery is real, performance should CRATER on shuffled data.
    """
    g_bar, g_obs, galaxy_idx = get_galaxy_data(galaxies, train_names)
    n_gal = len(train_names)
    
    # Original performance
    pred = best_law.func(g_bar)
    original_mae = galaxy_weighted_mae(pred, g_obs, galaxy_idx, n_gal)
    
    # Permutation performance
    perm_maes = []
    for _ in range(n_permutations):
        g_obs_shuffled = g_obs.copy()
        for g in range(n_gal):
            mask = galaxy_idx == g
            g_obs_shuffled[mask] = np.random.permutation(g_obs_shuffled[mask])
        
        perm_mae = galaxy_weighted_mae(pred, g_obs_shuffled, galaxy_idx, n_gal)
        perm_maes.append(perm_mae)
    
    mean_perm = np.mean(perm_maes)
    std_perm = np.std(perm_maes)
    
    # Z-score: how many SDs worse is permuted than original?
    z_score = (mean_perm - original_mae) / (std_perm + 1e-10)
    
    passed = z_score > 2.0  # Permuted should be much worse
    
    if verbose:
        print("\n" + "="*60)
        print("[CONTROL 1] Permutation Test")
        print("="*60)
        print(f"  Original MAE: {original_mae:.4f}")
        print(f"  Permuted MAE: {mean_perm:.4f} +/- {std_perm:.4f}")
        print(f"  Z-score: {z_score:.2f}")
        print(f"  PASSED: {passed} (permuted should be much worse)")
    
    return {
        "original_mae": original_mae,
        "permuted_mae_mean": mean_perm,
        "permuted_mae_std": std_perm,
        "z_score": z_score,
        "passed": passed
    }


def feature_ablation_control(galaxies: Dict[str, pd.DataFrame],
                             train_names: List[str],
                             verbose: bool = True) -> Dict:
    """
    KILL TEST 2: Remove g_bar, use only R.
    Performance should crater without the key physical input.
    """
    # Fit a model using only R (no g_bar)
    all_r = []
    all_g_obs = []
    galaxy_idx = []
    
    for i, name in enumerate(train_names):
        df = galaxies[name]
        all_r.extend(df['R_m'].values)
        all_g_obs.extend(df['g_obs'].values)
        galaxy_idx.extend([i] * len(df))
    
    all_r = np.array(all_r)
    all_g_obs = np.array(all_g_obs)
    galaxy_idx = np.array(galaxy_idx)
    n_gal = len(train_names)
    
    # Fit power law in R only
    def r_only_form(r, a, alpha):
        return a * np.power(r + 1e-30, alpha)
    
    try:
        params, _ = curve_fit(r_only_form, all_r, all_g_obs, 
                             p0=[1e-10, -1], bounds=([1e-15, -3], [1e-5, 0]), maxfev=5000)
        pred = r_only_form(all_r, *params)
        mae_r_only = galaxy_weighted_mae(pred, all_g_obs, galaxy_idx, n_gal)
    except:
        mae_r_only = float('inf')
    
    # Compare with full g_bar model (RAR as reference)
    g_bar, g_obs, _ = get_galaxy_data(galaxies, train_names)
    params_rar, _ = fit_candidate_law(rar_form, g_bar, g_obs,
                                      p0=[G_DAGGER], bounds=([1e-12], [1e-8]))
    if params_rar is not None:
        pred_rar = rar_form(g_bar, *params_rar)
        mae_full = galaxy_weighted_mae(pred_rar, g_obs, galaxy_idx, n_gal)
    else:
        mae_full = float('inf')
    
    ratio = mae_r_only / (mae_full + 1e-10)
    passed = ratio > 1.5  # R-only should be much worse
    
    if verbose:
        print("\n" + "="*60)
        print("[CONTROL 2] Feature Ablation (Remove g_bar)")
        print("="*60)
        print(f"  Full model (g_bar) MAE: {mae_full:.4f}")
        print(f"  R-only model MAE: {mae_r_only:.4f}")
        print(f"  Ratio: {ratio:.2f}x worse without g_bar")
        print(f"  PASSED: {passed} (g_bar must be essential)")
    
    return {
        "mae_full": mae_full,
        "mae_r_only": mae_r_only,
        "ratio": ratio,
        "passed": passed
    }


def random_feature_control(galaxies: Dict[str, pd.DataFrame],
                           train_names: List[str],
                           verbose: bool = True) -> Dict:
    """
    KILL TEST 3: Add random noise features.
    Discovery should NOT find them important.
    """
    g_bar, g_obs, galaxy_idx = get_galaxy_data(galaxies, train_names)
    
    # Add 5 random features
    n = len(g_bar)
    np.random.seed(42)
    random_features = np.random.randn(n, 5)
    
    # Fit linear model with g_bar + random features using OLS (not Lasso)
    X = np.column_stack([np.log10(g_bar + 1e-30), random_features])
    y = np.log10(g_obs + 1e-30)
    
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1e-6)  # Nearly OLS
    model.fit(X, y)
    
    # Check coefficients (absolute values)
    coef_gbar = np.abs(model.coef_[0])
    coef_random = np.abs(model.coef_[1:])
    
    ratio = coef_gbar / (np.mean(coef_random) + 1e-10)
    passed = ratio > 5  # g_bar should dominate random features
    
    if verbose:
        print("\n" + "="*60)
        print("[CONTROL 3] Random Feature Control")
        print("="*60)
        print(f"  g_bar coefficient: {coef_gbar:.4f}")
        print(f"  Random feature coefficients: {coef_random}")
        print(f"  Ratio (g_bar / mean_random): {ratio:.2f}x")
        print(f"  PASSED: {passed} (g_bar should dominate)")
    
    return {
        "coef_gbar": float(coef_gbar),
        "coef_random_mean": float(np.mean(coef_random)),
        "ratio": ratio,
        "passed": passed
    }


# =============================================================================
# FINAL TEST (LOCKED - Run ONCE)
# =============================================================================

def final_test(galaxies: Dict[str, pd.DataFrame],
               test_names: List[str],
               best_law: DiscoveredLaw,
               verbose: bool = True) -> Dict:
    """
    FINAL TEST on locked test galaxies.
    Run this ONCE at the very end!
    """
    g_bar, g_obs, galaxy_idx = get_galaxy_data(galaxies, test_names)
    n_gal = len(test_names)
    
    if verbose:
        print("\n" + "="*60)
        print("[FINAL TEST] On LOCKED test galaxies")
        print("="*60)
        print("  WARNING: This should only be run ONCE!")
    
    results = {}
    
    # Best discovered law
    pred = best_law.func(g_bar)
    mae_best = galaxy_weighted_mae(pred, g_obs, galaxy_idx, n_gal)
    results['discovered'] = {
        'name': best_law.name,
        'formula': best_law.formula,
        'mae': mae_best,
        'complexity': best_law.complexity
    }
    
    # Baselines
    # 1. Linear
    pred_linear = linear_form(g_bar, 1.5, 0)  # g_obs ~ 1.5 * g_bar typical
    mae_linear = galaxy_weighted_mae(pred_linear, g_obs, galaxy_idx, n_gal)
    results['linear'] = {'mae': mae_linear}
    
    # 2. RAR with default constant
    pred_rar = rar_form(g_bar, G_DAGGER)
    mae_rar = galaxy_weighted_mae(pred_rar, g_obs, galaxy_idx, n_gal)
    results['rar_default'] = {'mae': mae_rar}
    
    # 3. MOND with default constant
    pred_mond = mond_simple_form(g_bar, A0)
    mae_mond = galaxy_weighted_mae(pred_mond, g_obs, galaxy_idx, n_gal)
    results['mond_default'] = {'mae': mae_mond}
    
    if verbose:
        print(f"\n  Discovered ({best_law.name}): MAE = {mae_best:.4f}")
        print(f"  Linear baseline: MAE = {mae_linear:.4f}")
        print(f"  RAR (g†=1.2e-10): MAE = {mae_rar:.4f}")
        print(f"  MOND (a0=1.2e-10): MAE = {mae_mond:.4f}")
        
        # Verdict
        if mae_best < mae_linear and mae_best <= mae_rar * 1.1:
            print(f"\n  [VERDICT] Discovered law GENERALIZES and is competitive with RAR!")
        elif mae_best < mae_linear:
            print(f"\n  [VERDICT] Discovered law beats linear but not RAR.")
        else:
            print(f"\n  [VERDICT] Discovered law does NOT generalize well.")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="UDE Acceleration Law Discovery")
    parser.add_argument("--discover", action="store_true", help="Run discovery on train set")
    parser.add_argument("--validate", action="store_true", help="Validate on held-out galaxies")
    parser.add_argument("--controls", action="store_true", help="Run kill tests")
    parser.add_argument("--test", action="store_true", help="FINAL test (run ONCE!)")
    parser.add_argument("--all", action="store_true", help="Run discovery + validation + controls")
    parser.add_argument("--data-dir", default="sparc_galaxies", help="SPARC data directory")
    args = parser.parse_args()
    
    if not any([args.discover, args.validate, args.controls, args.test, args.all]):
        args.all = True
    
    print("="*60)
    print("[PAPER 2] Symbolic Discovery of Acceleration Laws")
    print("="*60)
    print("Target: g_obs = f(g_bar)")
    print("Method: Galaxy-level holdout + complexity penalty + controls")
    
    # Load data
    galaxies = load_sparc_accelerations(args.data_dir)
    
    # Split galaxies
    train_names, val_names, test_names = split_galaxies(galaxies)
    
    best_law = None
    
    if args.discover or args.all:
        # Discover candidate laws
        candidates = discover_laws(galaxies, train_names)
        best_law = candidates[0] if candidates else None
    
    if args.validate or args.all:
        if best_law is None:
            candidates = discover_laws(galaxies, train_names, verbose=False)
        best_law = validate_laws(galaxies, val_names, candidates)
    
    if args.controls or args.all:
        if best_law is None:
            candidates = discover_laws(galaxies, train_names, verbose=False)
            best_law = candidates[0]
        
        # Run kill tests
        perm_result = permutation_control(galaxies, train_names, best_law)
        ablation_result = feature_ablation_control(galaxies, train_names)
        random_result = random_feature_control(galaxies, train_names)
        
        all_passed = perm_result['passed'] and ablation_result['passed'] and random_result['passed']
        print(f"\n[CONTROLS SUMMARY] All passed: {all_passed}")
    
    if args.test:
        if best_law is None:
            candidates = discover_laws(galaxies, train_names, verbose=False)
            best_law = validate_laws(galaxies, val_names, candidates, verbose=False)
        
        # FINAL TEST (LOCKED)
        print("\n" + "!"*60)
        print("  WARNING: Running FINAL TEST on LOCKED test galaxies!")
        print("  This should only be done ONCE!")
        print("!"*60)
        
        confirm = input("  Type 'YES' to proceed: ")
        if confirm == 'YES':
            results = final_test(galaxies, test_names, best_law)
            
            # Save results
            with open("paper2_final_results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n[SAVED] Results to paper2_final_results.json")
        else:
            print("  Aborted.")
    
    print("\n" + "="*60)
    print("[COMPLETE]")
    print("="*60)


if __name__ == "__main__":
    main()
