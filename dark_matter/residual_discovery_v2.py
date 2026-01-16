#!/usr/bin/env python3
"""
Paper 2b (FIXED): Residual Discovery
=====================================

FIXES from AI reviewer:
1. Fit RAR in LOG space (not linear space)
2. Compute residuals in low-acceleration regime only
3. Apply FDR correction for multiple testing
4. Control for n_points as nuisance variable

"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.optimize import minimize
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Physical constants
G_DAGGER = 1.2e-10  # m/s^2 - RAR characteristic acceleration
LOW_ACCEL_THRESHOLD = 3e-11  # Only analyze low-acceleration regime


def load_sparc_with_properties(data_dir: str = "sparc_galaxies") -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Load SPARC data + galaxy properties."""
    data_path = Path(data_dir)
    galaxies = {}
    properties = []
    
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
            df['e_V_obs'] = df['e_V_obs'].fillna(df['V_obs'] * 0.1)  # Default 10% error
            
            df = df[(df['g_bar'] > 1e-15) & (df['g_obs'] > 1e-15)].copy()
            
            if len(df) >= 5:
                galaxies[name] = df
                
                props = {
                    'name': name,
                    'n_points': len(df),
                    'R_max': df['R'].max(),
                    'V_flat': df['V_obs'].iloc[-3:].mean() if len(df) >= 3 else df['V_obs'].mean(),
                    'V_bar_mean': df['V_bar'].mean(),
                    'g_bar_median': df['g_bar'].median(),
                    'SB_mean': df['SBdisk'].mean() if 'SBdisk' in df.columns else np.nan,
                    'f_gas': (df['V_gas']**2).sum() / (V_bar_sq.sum() + 1e-10),
                    'f_bul': (df['V_bul']**2).sum() / (V_bar_sq.sum() + 1e-10),
                    # Count of points in low-acceleration regime
                    'n_low_accel': (df['g_bar'] < LOW_ACCEL_THRESHOLD).sum(),
                }
                properties.append(props)
                
        except Exception:
            continue
    
    props_df = pd.DataFrame(properties)
    print(f"[DATA] Loaded {len(galaxies)} galaxies with properties")
    return galaxies, props_df


def rar_prediction(g_bar: np.ndarray, g_dagger: float = G_DAGGER) -> np.ndarray:
    """RAR functional form."""
    with np.errstate(over='ignore', invalid='ignore'):
        x = np.sqrt(g_bar / g_dagger)
        result = g_bar / (1 - np.exp(-x))
        result = np.where(np.isfinite(result), result, g_bar)
    return result


def fit_rar_log_space(all_g_bar: np.ndarray, all_g_obs: np.ndarray) -> float:
    """
    FIX #1: Fit RAR by minimizing LOG-space MAE.
    This is the proper way since the relation has small scatter in dex.
    """
    def log_loss(theta):
        g_dagger = theta[0]
        pred = rar_prediction(all_g_bar, g_dagger)
        residuals = np.log10(all_g_obs + 1e-30) - np.log10(pred + 1e-30)
        return np.mean(np.abs(residuals))
    
    result = minimize(log_loss, x0=[G_DAGGER], bounds=[(1e-12, 1e-8)], method='L-BFGS-B')
    return result.x[0]


def compute_residuals_low_accel(galaxies: Dict[str, pd.DataFrame], 
                                 props_df: pd.DataFrame,
                                 verbose: bool = True) -> Tuple[pd.DataFrame, float]:
    """
    FIX #2: Compute residuals ONLY in low-acceleration regime.
    This is where MOND/RAR physics matters most.
    """
    # Collect all data
    all_g_bar, all_g_obs = [], []
    for df in galaxies.values():
        all_g_bar.extend(df['g_bar'].values)
        all_g_obs.extend(df['g_obs'].values)
    
    all_g_bar = np.array(all_g_bar)
    all_g_obs = np.array(all_g_obs)
    
    # FIT IN LOG SPACE
    g_dagger_fit = fit_rar_log_space(all_g_bar, all_g_obs)
    
    if verbose:
        print(f"[RAR FIT] g† = {g_dagger_fit:.2e} m/s² (fitted in log-space)")
    
    # Compute per-galaxy residuals IN LOW-ACCELERATION REGIME ONLY
    residual_stats = []
    
    for name, df in galaxies.items():
        # Filter to low-acceleration regime
        low_accel_mask = df['g_bar'] < LOW_ACCEL_THRESHOLD
        
        if low_accel_mask.sum() < 3:
            # Not enough low-acceleration points
            continue
        
        g_bar_low = df.loc[low_accel_mask, 'g_bar'].values
        g_obs_low = df.loc[low_accel_mask, 'g_obs'].values
        
        g_pred = rar_prediction(g_bar_low, g_dagger_fit)
        log_residual = np.log10(g_obs_low) - np.log10(g_pred)
        
        stats_dict = {
            'name': name,
            'residual_mean': np.mean(log_residual),
            'residual_std': np.std(log_residual),
            'residual_median': np.median(log_residual),
            'n_low_accel_used': len(g_bar_low),
        }
        residual_stats.append(stats_dict)
    
    residual_df = pd.DataFrame(residual_stats)
    result = props_df.merge(residual_df, on='name')
    
    if verbose:
        print(f"[RESIDUALS] Using {len(result)} galaxies with ≥3 low-accel points")
        print(f"[RESIDUALS] Mean residual: {result['residual_mean'].mean():.4f} dex")
        print(f"[RESIDUALS] Std of residual means: {result['residual_mean'].std():.4f} dex")
    
    return result, g_dagger_fit


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    FIX #3: Apply Benjamini-Hochberg FDR correction.
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    # BH threshold
    thresholds = alpha * np.arange(1, n + 1) / n
    
    # Find largest k where p_(k) <= threshold
    significant = np.zeros(n, dtype=bool)
    for i in range(n - 1, -1, -1):
        if sorted_p[i] <= thresholds[i]:
            significant[sorted_idx[:i+1]] = True
            break
    
    return significant


def discover_residual_predictors_fixed(result_df: pd.DataFrame, 
                                        verbose: bool = True) -> Dict:
    """
    FIX #3: Proper multiple testing correction + partial correlations.
    """
    predictors = ['f_gas', 'f_bul', 'SB_mean', 'V_flat', 'R_max', 'g_bar_median']
    target = 'residual_mean'
    nuisance = 'n_points'  # Control for sampling artifact
    
    if verbose:
        print("\n" + "="*60)
        print("[RESIDUAL DISCOVERY] What predicts RAR residuals?")
        print("="*60)
        print(f"  (Controlling for {nuisance} as nuisance variable)")
        print(f"  Low-accel regime only (g_bar < {LOW_ACCEL_THRESHOLD:.0e})")
    
    results = {}
    p_values = []
    predictor_names = []
    
    for pred in predictors:
        if pred not in result_df.columns:
            continue
            
        mask = ~(result_df[pred].isna() | result_df[target].isna() | result_df[nuisance].isna())
        df_clean = result_df.loc[mask, [pred, target, nuisance]]
        
        if len(df_clean) < 15:
            continue
        
        # Partial correlation (controlling for n_points)
        from scipy.stats import pearsonr
        
        # Residualize both predictor and target on n_points
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        
        X_nuisance = df_clean[[nuisance]].values
        pred_resid = df_clean[pred].values - lr.fit(X_nuisance, df_clean[pred]).predict(X_nuisance)
        target_resid = df_clean[target].values - lr.fit(X_nuisance, df_clean[target]).predict(X_nuisance)
        
        r_partial, p_partial = pearsonr(pred_resid, target_resid)
        
        # Also simple correlation for comparison
        r_simple, p_simple = pearsonr(df_clean[pred].values, df_clean[target].values)
        
        results[pred] = {
            'partial_r': r_partial,
            'partial_p': p_partial,
            'simple_r': r_simple,
            'simple_p': p_simple,
        }
        p_values.append(p_partial)
        predictor_names.append(pred)
    
    # Apply FDR correction
    if p_values:
        p_values = np.array(p_values)
        significant = benjamini_hochberg(p_values, alpha=0.05)
        
        for i, pred in enumerate(predictor_names):
            results[pred]['fdr_significant'] = significant[i]
        
        if verbose:
            print(f"\n  {'Predictor':<15} {'Partial r':>10} {'p-value':>10} {'FDR sig':>10}")
            print("  " + "-"*50)
            for i, pred in enumerate(predictor_names):
                r = results[pred]
                sig_mark = "***" if r['fdr_significant'] else ""
                print(f"  {pred:<15} {r['partial_r']:>+10.3f} {r['partial_p']:>10.4f} {sig_mark:>10}")
    
    # Find strongest after FDR
    fdr_sig = [p for p in predictor_names if results.get(p, {}).get('fdr_significant', False)]
    if fdr_sig:
        strongest = max(fdr_sig, key=lambda k: abs(results[k]['partial_r']))
        results['strongest_fdr_significant'] = strongest
        if verbose:
            print(f"\n[STRONGEST FDR-SIGNIFICANT] {strongest}: partial r = {results[strongest]['partial_r']:.3f}")
    else:
        results['strongest_fdr_significant'] = None
        if verbose:
            print(f"\n[NO FDR-SIGNIFICANT PREDICTORS]")
    
    return results


def multivariate_residual_model_fixed(result_df: pd.DataFrame,
                                       verbose: bool = True) -> Dict:
    """
    FIX #3: Multivariate model with n_points as nuisance.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    
    # ALWAYS include n_points as nuisance
    predictors = ['f_gas', 'f_bul', 'V_flat', 'R_max', 'g_bar_median', 'n_points']
    target = 'residual_mean'
    
    available = [p for p in predictors if p in result_df.columns]
    df_clean = result_df[available + [target]].dropna()
    
    if len(df_clean) < 20:
        print("[WARNING] Not enough data for multivariate model")
        return {}
    
    X = df_clean[available].values
    y = df_clean[target].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RidgeCV(cv=5, alphas=[0.01, 0.1, 1, 10, 100])
    model.fit(X_scaled, y)
    
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    
    if verbose:
        print("\n" + "="*60)
        print("[MULTIVARIATE MODEL] (n_points included as nuisance)")
        print("="*60)
        print(f"  Cross-validated R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  Best alpha: {model.alpha_}")
        print(f"\n  Feature coefficients (standardized):")
        for feat, coef in zip(available, model.coef_):
            nuisance_mark = " (nuisance)" if feat == 'n_points' else ""
            print(f"    {feat:<15}: {coef:+.4f}{nuisance_mark}")
        
        # Interpretation
        if cv_scores.mean() < 0.1:
            print(f"\n  [INTERPRETATION] R² < 0.1 means residuals are mostly noise,")
            print(f"                   NOT systematically predicted by galaxy properties.")
        elif cv_scores.mean() < 0.3:
            print(f"\n  [INTERPRETATION] Weak signal. May be systematics, not new physics.")
    
    return {
        'cv_r2_mean': float(cv_scores.mean()),
        'cv_r2_std': float(cv_scores.std()),
        'coefficients': dict(zip(available, model.coef_.tolist())),
        'alpha': float(model.alpha_)
    }


def residual_vs_g_bar_binned(galaxies: Dict[str, pd.DataFrame],
                              g_dagger_fit: float,
                              verbose: bool = True) -> Dict:
    """
    Check for systematic residual structure vs g_bar using bins.
    """
    # Collect all data
    all_g_bar, all_residuals = [], []
    
    for df in galaxies.values():
        g_pred = rar_prediction(df['g_bar'].values, g_dagger_fit)
        residual = np.log10(df['g_obs'].values) - np.log10(g_pred)
        all_g_bar.extend(df['g_bar'].values)
        all_residuals.extend(residual)
    
    all_g_bar = np.array(all_g_bar)
    all_residuals = np.array(all_residuals)
    
    # Bin by g_bar
    log_g_bar = np.log10(all_g_bar)
    bins = np.linspace(log_g_bar.min(), log_g_bar.max(), 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    bin_stds = []
    
    for i in range(len(bins) - 1):
        mask = (log_g_bar >= bins[i]) & (log_g_bar < bins[i+1])
        if mask.sum() > 10:
            bin_means.append(np.mean(all_residuals[mask]))
            bin_stds.append(np.std(all_residuals[mask]) / np.sqrt(mask.sum()))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
    
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    
    # Test for trend
    valid = ~np.isnan(bin_means)
    if valid.sum() >= 4:
        slope, intercept, r, p, se = stats.linregress(bin_centers[valid], bin_means[valid])
    else:
        slope, p = 0, 1
    
    if verbose:
        print("\n" + "="*60)
        print("[RESIDUAL vs g_bar] Binned analysis")
        print("="*60)
        print(f"  {'log10(g_bar)':>12} {'mean resid':>12} {'std err':>12}")
        print("  " + "-"*40)
        for i, (c, m, s) in enumerate(zip(bin_centers, bin_means, bin_stds)):
            if not np.isnan(m):
                print(f"  {c:>12.1f} {m:>+12.4f} {s:>12.4f}")
        
        print(f"\n  Linear trend: slope = {slope:.4f}, p = {p:.4f}")
        if p < 0.05:
            print(f"  [!] Significant trend detected - possible systematic")
        else:
            print(f"  [OK] No significant trend - residuals are flat")
    
    return {
        'bin_centers': bin_centers.tolist(),
        'bin_means': [float(x) if not np.isnan(x) else None for x in bin_means],
        'trend_slope': float(slope),
        'trend_p': float(p)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Residual Discovery (FIXED)")
    parser.add_argument("--data-dir", default="sparc_galaxies", help="SPARC data directory")
    args = parser.parse_args()
    
    print("="*60)
    print("[PAPER 2b FIXED] Residual Discovery")
    print("="*60)
    print("FIXES APPLIED:")
    print("  1. RAR fitted in LOG space")
    print("  2. Residuals computed in LOW-ACCELERATION regime only")
    print("  3. FDR correction for multiple testing")
    print("  4. n_points controlled as nuisance")
    
    # Load data
    galaxies, props_df = load_sparc_with_properties(args.data_dir)
    
    # Compute residuals (FIXED)
    result_df, g_dagger_fit = compute_residuals_low_accel(galaxies, props_df)
    
    # Discover predictors (FIXED)
    predictor_results = discover_residual_predictors_fixed(result_df)
    
    # Multivariate model (FIXED)
    multi_results = multivariate_residual_model_fixed(result_df)
    
    # Residual vs g_bar structure
    binned_results = residual_vs_g_bar_binned(galaxies, g_dagger_fit)
    
    # Save results
    all_results = {
        'g_dagger_fit': float(g_dagger_fit),
        'n_galaxies_analyzed': len(result_df),
        'low_accel_threshold': LOW_ACCEL_THRESHOLD,
        'predictor_correlations': predictor_results,
        'multivariate_model': multi_results,
        'binned_residuals': binned_results
    }
    
    with open("paper2b_residual_results_v2.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("[SAVED] Results to paper2b_residual_results_v2.json")
    print("="*60)
    
    # Summary
    print("\n" + "="*60)
    print("[SUMMARY]")
    print("="*60)
    if predictor_results.get('strongest_fdr_significant'):
        pred = predictor_results['strongest_fdr_significant']
        r = predictor_results[pred]['partial_r']
        print(f"  Strongest FDR-significant predictor: {pred} (r={r:.3f})")
    else:
        print("  NO predictors survive FDR correction")
    
    r2 = multi_results.get('cv_r2_mean', 0)
    print(f"  Multivariate CV R²: {r2:.3f}")
    
    if r2 < 0.1:
        print("\n  CONCLUSION: RAR residuals show NO systematic structure")
        print("              that can be predicted by galaxy properties.")
        print("              This is EXPECTED if RAR is the correct law.")
    elif r2 < 0.3:
        print("\n  CONCLUSION: Weak residual structure exists but may be")
        print("              systematics rather than new physics.")


if __name__ == "__main__":
    main()
