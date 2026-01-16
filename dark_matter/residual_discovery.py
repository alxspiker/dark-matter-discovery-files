#!/usr/bin/env python3
"""
Paper 2b: Residual Discovery
============================

After fitting RAR to all data, what predicts the RESIDUALS?
This is where new physics might hide.

Galaxy properties to test:
- Surface brightness (μ_eff)
- Gas fraction (f_gas = M_gas / M_bar)
- Morphological type
- Distance
- Inclination
- Scale length

"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.optimize import curve_fit
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Physical constants
G_DAGGER = 1.2e-10  # m/s^2 - RAR characteristic acceleration


def load_sparc_with_properties(data_dir: str = "sparc_galaxies") -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load SPARC data + galaxy properties.
    Returns rotation curves and a properties table.
    """
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
                
            # Clean data
            df = df[(df['V_obs'] > 0) & (df['R'] > 0)].copy()
            
            # Compute accelerations
            V_bar_sq = df['V_gas']**2 + df['V_disk']**2 + df['V_bul']**2
            df['V_bar'] = np.sqrt(np.maximum(V_bar_sq, 1e-10))
            df['g_bar'] = df['V_bar']**2 / (df['R'] * 3.086e19) * 1e6  # Convert to m/s^2
            df['g_obs'] = df['V_obs']**2 / (df['R'] * 3.086e19) * 1e6
            
            # Filter valid accelerations
            df = df[(df['g_bar'] > 1e-15) & (df['g_obs'] > 1e-15)].copy()
            
            if len(df) >= 5:
                galaxies[name] = df
                
                # Compute galaxy-level properties
                props = {
                    'name': name,
                    'n_points': len(df),
                    'R_max': df['R'].max(),
                    'V_flat': df['V_obs'].iloc[-3:].mean() if len(df) >= 3 else df['V_obs'].mean(),
                    'V_bar_mean': df['V_bar'].mean(),
                    'g_bar_median': df['g_bar'].median(),
                    'SB_mean': df['SBdisk'].mean() if 'SBdisk' in df.columns else np.nan,
                    # Gas fraction proxy
                    'f_gas': (df['V_gas']**2).sum() / (V_bar_sq.sum() + 1e-10),
                    # Bulge fraction proxy
                    'f_bul': (df['V_bul']**2).sum() / (V_bar_sq.sum() + 1e-10),
                }
                properties.append(props)
                
        except Exception as e:
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


def compute_residuals(galaxies: Dict[str, pd.DataFrame], 
                      props_df: pd.DataFrame,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Fit RAR globally, compute per-galaxy residuals.
    Returns a DataFrame with galaxy properties + residual statistics.
    """
    # Collect all data
    all_g_bar = []
    all_g_obs = []
    galaxy_names = []
    
    for name, df in galaxies.items():
        all_g_bar.extend(df['g_bar'].values)
        all_g_obs.extend(df['g_obs'].values)
        galaxy_names.extend([name] * len(df))
    
    all_g_bar = np.array(all_g_bar)
    all_g_obs = np.array(all_g_obs)
    galaxy_names = np.array(galaxy_names)
    
    # Fit RAR globally
    def rar_func(g_bar, g_dagger):
        x = np.sqrt(g_bar / g_dagger)
        return g_bar / (1 - np.exp(-x))
    
    try:
        popt, _ = curve_fit(rar_func, all_g_bar, all_g_obs, 
                           p0=[G_DAGGER], bounds=(1e-12, 1e-8), maxfev=5000)
        g_dagger_fit = popt[0]
    except:
        g_dagger_fit = G_DAGGER
    
    if verbose:
        print(f"[RAR FIT] g† = {g_dagger_fit:.2e} m/s²")
    
    # Compute per-galaxy residuals
    residual_stats = []
    
    for name in galaxies.keys():
        mask = galaxy_names == name
        g_bar_gal = all_g_bar[mask]
        g_obs_gal = all_g_obs[mask]
        
        # RAR prediction
        g_pred = rar_func(g_bar_gal, g_dagger_fit)
        
        # Residuals in log space
        log_residual = np.log10(g_obs_gal) - np.log10(g_pred)
        
        stats_dict = {
            'name': name,
            'residual_mean': np.mean(log_residual),
            'residual_std': np.std(log_residual),
            'residual_median': np.median(log_residual),
            'residual_iqr': np.percentile(log_residual, 75) - np.percentile(log_residual, 25),
            'residual_skew': stats.skew(log_residual) if len(log_residual) > 3 else 0,
        }
        residual_stats.append(stats_dict)
    
    residual_df = pd.DataFrame(residual_stats)
    
    # Merge with properties
    result = props_df.merge(residual_df, on='name')
    
    if verbose:
        print(f"[RESIDUALS] Mean residual: {result['residual_mean'].mean():.4f}")
        print(f"[RESIDUALS] Std of residuals: {result['residual_std'].mean():.4f}")
    
    return result, g_dagger_fit


def discover_residual_predictors(result_df: pd.DataFrame, 
                                  verbose: bool = True) -> Dict:
    """
    What galaxy properties predict RAR residuals?
    This is where new physics might hide.
    """
    # Properties to test
    predictors = ['f_gas', 'f_bul', 'SB_mean', 'V_flat', 'R_max', 'g_bar_median', 'n_points']
    target = 'residual_mean'
    
    if verbose:
        print("\n" + "="*60)
        print("[RESIDUAL DISCOVERY] What predicts RAR residuals?")
        print("="*60)
    
    results = {}
    
    for pred in predictors:
        if pred not in result_df.columns:
            continue
            
        # Remove NaN
        mask = ~(result_df[pred].isna() | result_df[target].isna())
        x = result_df.loc[mask, pred].values
        y = result_df.loc[mask, target].values
        
        if len(x) < 10:
            continue
        
        # Pearson correlation
        r, p = stats.pearsonr(x, y)
        
        # Spearman (rank) correlation
        rho, p_spearman = stats.spearmanr(x, y)
        
        results[pred] = {
            'pearson_r': r,
            'pearson_p': p,
            'spearman_rho': rho,
            'spearman_p': p_spearman,
            'significant': p < 0.05
        }
        
        if verbose:
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {pred:15s}: r={r:+.3f} (p={p:.4f}) {sig}")
    
    # Find strongest predictor
    if results:
        strongest = max(results.keys(), key=lambda k: abs(results[k]['pearson_r']))
        results['strongest_predictor'] = strongest
        results['strongest_r'] = results[strongest]['pearson_r']
        
        if verbose:
            print(f"\n[STRONGEST] {strongest}: r = {results[strongest]['pearson_r']:.3f}")
    
    return results


def multivariate_residual_model(result_df: pd.DataFrame,
                                 verbose: bool = True) -> Dict:
    """
    Multivariate regression to predict residuals.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    
    predictors = ['f_gas', 'f_bul', 'V_flat', 'R_max', 'g_bar_median']
    target = 'residual_mean'
    
    # Prepare data
    available = [p for p in predictors if p in result_df.columns]
    df_clean = result_df[available + [target]].dropna()
    
    if len(df_clean) < 20:
        print("[WARNING] Not enough data for multivariate model")
        return {}
    
    X = df_clean[available].values
    y = df_clean[target].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit with cross-validation
    model = RidgeCV(cv=5)
    model.fit(X_scaled, y)
    
    # Cross-val R²
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    
    if verbose:
        print("\n" + "="*60)
        print("[MULTIVARIATE MODEL] Predicting residuals from galaxy properties")
        print("="*60)
        print(f"  Cross-validated R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  Feature importances:")
        for feat, coef in zip(available, model.coef_):
            print(f"    {feat:15s}: {coef:+.4f}")
    
    return {
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'coefficients': dict(zip(available, model.coef_.tolist())),
        'predictors': available
    }


def residual_structure_test(galaxies: Dict[str, pd.DataFrame],
                            g_dagger_fit: float,
                            verbose: bool = True) -> Dict:
    """
    Is there systematic structure in residuals vs g_bar?
    (Beyond random scatter)
    """
    # Collect all data with residuals
    g_bar_all = []
    residual_all = []
    
    for name, df in galaxies.items():
        g_pred = rar_prediction(df['g_bar'].values, g_dagger_fit)
        residual = np.log10(df['g_obs'].values) - np.log10(g_pred)
        g_bar_all.extend(df['g_bar'].values)
        residual_all.extend(residual)
    
    g_bar_all = np.array(g_bar_all)
    residual_all = np.array(residual_all)
    
    # Bin by g_bar and check for systematic trends
    log_g_bar = np.log10(g_bar_all)
    bins = np.linspace(log_g_bar.min(), log_g_bar.max(), 10)
    bin_idx = np.digitize(log_g_bar, bins)
    
    bin_means = []
    bin_stds = []
    bin_centers = []
    
    for i in range(1, len(bins)):
        mask = bin_idx == i
        if mask.sum() > 10:
            bin_means.append(np.mean(residual_all[mask]))
            bin_stds.append(np.std(residual_all[mask]))
            bin_centers.append((bins[i-1] + bins[i]) / 2)
    
    bin_means = np.array(bin_means)
    bin_centers = np.array(bin_centers)
    
    # Test for systematic trend
    if len(bin_means) > 3:
        slope, intercept, r, p, se = stats.linregress(bin_centers, bin_means)
        has_trend = p < 0.05 and abs(slope) > 0.01
    else:
        slope, r, p = 0, 0, 1
        has_trend = False
    
    if verbose:
        print("\n" + "="*60)
        print("[RESIDUAL STRUCTURE] Is there systematic g_bar dependence?")
        print("="*60)
        print(f"  Slope of residual vs log(g_bar): {slope:.4f}")
        print(f"  Correlation: r = {r:.3f}, p = {p:.4f}")
        print(f"  Systematic trend: {has_trend}")
        
        if has_trend:
            print(f"\n  [!] RAR may not fully capture the acceleration dependence!")
        else:
            print(f"\n  [✓] No systematic residual structure - RAR fits well")
    
    return {
        'slope': slope,
        'correlation': r,
        'p_value': p,
        'has_trend': has_trend,
        'bin_centers': bin_centers.tolist(),
        'bin_means': bin_means.tolist()
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Residual Discovery Analysis")
    parser.add_argument("--data-dir", default="sparc_galaxies", help="SPARC data directory")
    args = parser.parse_args()
    
    print("="*60)
    print("[PAPER 2b] Residual Discovery: What Does RAR Miss?")
    print("="*60)
    
    # Load data with properties
    galaxies, props_df = load_sparc_with_properties(args.data_dir)
    
    # Compute residuals from RAR
    result_df, g_dagger_fit = compute_residuals(galaxies, props_df)
    
    # What predicts residuals?
    univariate_results = discover_residual_predictors(result_df)
    
    # Multivariate model
    multivariate_results = multivariate_residual_model(result_df)
    
    # Residual structure test
    structure_results = residual_structure_test(galaxies, g_dagger_fit)
    
    # Save results
    all_results = {
        'g_dagger_fit': g_dagger_fit,
        'univariate': univariate_results,
        'multivariate': multivariate_results,
        'structure': structure_results
    }
    
    with open("paper2b_residual_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("[SAVED] Results to paper2b_residual_results.json")
    print("="*60)
    
    # Summary
    print("\n[SUMMARY]")
    if univariate_results.get('strongest_r'):
        print(f"  Strongest residual predictor: {univariate_results['strongest_predictor']}")
        print(f"  Correlation: r = {univariate_results['strongest_r']:.3f}")
    
    if multivariate_results.get('cv_r2_mean'):
        print(f"  Multivariate R²: {multivariate_results['cv_r2_mean']:.3f}")
    
    if structure_results['has_trend']:
        print(f"  [!] Systematic residual structure detected!")
    else:
        print(f"  [✓] No systematic residual structure")


if __name__ == "__main__":
    main()
