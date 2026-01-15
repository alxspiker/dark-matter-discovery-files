"""
COMPREHENSIVE VISUALIZATION
==========================
Generate publication-quality figures for the research paper.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re
import glob
from scipy import stats

SPARC_DIR = os.path.join(os.path.dirname(__file__), "sparc_galaxies")

def load_all_galaxies():
    """Load all SPARC galaxy data files."""
    files = glob.glob(os.path.join(SPARC_DIR, "*_rotmod.dat"))
    galaxies = {}
    
    for f in files:
        name = os.path.basename(f).replace("_rotmod.dat", "")
        radius, vobs, vgas, vdisk, vbul = [], [], [], [], []
        
        with open(f, 'r') as fp:
            for line in fp:
                if not line.strip() or line.strip().startswith("#"):
                    continue
                try:
                    parts = re.split(r"\s+", line.strip())
                    radius.append(float(parts[0]))
                    vobs.append(float(parts[1]))
                    vgas.append(abs(float(parts[3])))
                    vdisk.append(abs(float(parts[4])))
                    vbul.append(abs(float(parts[5])) if len(parts) > 5 else 0.0)
                except:
                    continue
        
        if len(radius) < 5:
            continue
            
        vbar = np.sqrt(np.array(vgas)**2 + np.array(vdisk)**2 + np.array(vbul)**2)
        
        galaxies[name] = {
            'radius': np.array(radius),
            'vobs': np.array(vobs),
            'vbar': vbar,
        }
    
    return galaxies

def create_figure1_rotation_curves(galaxies):
    """Figure 1: Example rotation curves showing the flat curve anomaly."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    example_galaxies = ['NGC3198', 'NGC2403', 'NGC7331', 'NGC2841']
    
    for ax, name in zip(axes.flatten(), example_galaxies):
        if name not in galaxies:
            continue
        
        data = galaxies[name]
        r = data['radius']
        vobs = data['vobs']
        vbar = data['vbar']
        
        ax.plot(r, vobs, 'b-', linewidth=2, marker='o', markersize=4, label='Observed')
        ax.plot(r, vbar, 'r--', linewidth=2, marker='s', markersize=4, label='Newtonian (V_bar)')
        ax.axhline(y=np.mean(vobs[-5:]), color='green', linestyle=':', alpha=0.7, label='Flat plateau')
        
        ax.set_xlabel('Radius (kpc)', fontsize=12)
        ax.set_ylabel('Velocity (km/s)', fontsize=12)
        ax.set_title(f'{name}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Galaxy Rotation Curves: The Dark Matter Anomaly', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figure1_rotation_curves.png'), dpi=150)
    print("Saved: figure1_rotation_curves.png")

def create_figure2_overflow_detection(galaxies):
    """Figure 2: The digital overflow detection and phase transition."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Collect all normalized data
    all_sums = []
    all_vobs_norm = []
    
    for name, data in galaxies.items():
        r = data['radius']
        vbar = data['vbar']
        vobs = data['vobs']
        
        max_v = max(vobs.max(), vbar.max()) if vbar.max() > 0 else 1.0
        max_r = r.max() if r.max() > 0 else 1.0
        
        norm_vbar = vbar / max_v
        norm_r = r / max_r
        norm_vobs = vobs / max_v
        
        digital_sum = norm_vbar + norm_r
        
        all_sums.extend(digital_sum)
        all_vobs_norm.extend(norm_vobs)
    
    all_sums = np.array(all_sums)
    all_vobs_norm = np.array(all_vobs_norm)
    
    # Left plot: Scatter with threshold
    ax1 = axes[0]
    ax1.scatter(all_sums, all_vobs_norm, alpha=0.3, s=10, c='blue')
    ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Overflow Threshold')
    
    # Add binned means
    bin_edges = np.linspace(0, 2, 21)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means = []
    for i in range(len(bin_edges)-1):
        mask = (all_sums >= bin_edges[i]) & (all_sums < bin_edges[i+1])
        if np.sum(mask) > 0:
            bin_means.append(np.mean(all_vobs_norm[mask]))
        else:
            bin_means.append(np.nan)
    
    ax1.plot(bin_centers, bin_means, 'k-', linewidth=2, marker='o', markersize=6, label='Binned Mean')
    ax1.set_xlabel('Digital Sum (Norm V_bar + Norm R)', fontsize=12)
    ax1.set_ylabel('Normalized Observed Velocity', fontsize=12)
    ax1.set_title('The Gravitational Overflow: Phase Transition', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Histogram comparison
    ax2 = axes[1]
    below = all_vobs_norm[all_sums < 1.0]
    above = all_vobs_norm[all_sums >= 1.0]
    
    ax2.hist(below, bins=30, alpha=0.5, color='blue', label=f'Below Threshold (n={len(below)})', density=True)
    ax2.hist(above, bins=30, alpha=0.5, color='red', label=f'Above Threshold (n={len(above)})', density=True)
    ax2.axvline(x=np.mean(below), color='blue', linestyle='--', linewidth=2)
    ax2.axvline(x=np.mean(above), color='red', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('Normalized Observed Velocity', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Velocity Distribution: Below vs Above Threshold', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figure2_overflow_detection.png'), dpi=150)
    print("Saved: figure2_overflow_detection.png")

def create_figure3_model_comparison(galaxies):
    """Figure 3: Model comparison and RMSE distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    newton_rmse = []
    overflow_rmse = []
    
    for name, data in galaxies.items():
        r = data['radius']
        vbar = data['vbar']
        vobs = data['vobs']
        
        # Newtonian
        rmse_n = np.sqrt(np.mean((vobs - vbar)**2))
        newton_rmse.append(rmse_n)
        
        # Overflow
        max_v = max(vobs.max(), vbar.max()) if vbar.max() > 0 else 1.0
        max_r = r.max() if r.max() > 0 else 1.0
        norm_vbar = vbar / max_v
        norm_r = r / max_r
        overflow_mask = (norm_vbar + norm_r) >= 1.0
        
        pred_overflow = np.copy(vbar)
        if np.sum(overflow_mask) > 0:
            v_flat = np.mean(vobs[overflow_mask])
            pred_overflow[overflow_mask] = v_flat
        
        rmse_o = np.sqrt(np.mean((vobs - pred_overflow)**2))
        overflow_rmse.append(rmse_o)
    
    # Left: Scatter comparison
    ax1 = axes[0]
    ax1.scatter(newton_rmse, overflow_rmse, alpha=0.5, c='blue', s=30)
    max_err = max(max(newton_rmse), max(overflow_rmse))
    ax1.plot([0, max_err], [0, max_err], 'r--', linewidth=2, label='Equal Performance')
    ax1.set_xlabel('Newtonian RMSE (km/s)', fontsize=12)
    ax1.set_ylabel('Overflow Model RMSE (km/s)', fontsize=12)
    ax1.set_title('Predictive Error Comparison', fontsize=14, fontweight='bold')
    
    wins = sum(1 for n, o in zip(newton_rmse, overflow_rmse) if o < n)
    ax1.text(max_err*0.1, max_err*0.85, 
             f'Overflow wins:\n{wins}/{len(newton_rmse)} galaxies\n({100*wins/len(newton_rmse):.1f}%)', 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Bar chart of mean RMSE
    ax2 = axes[1]
    models = ['Newtonian', 'Overflow\nLatch']
    rmses = [np.mean(newton_rmse), np.mean(overflow_rmse)]
    colors = ['red', 'green']
    
    bars = ax2.bar(models, rmses, color=colors, alpha=0.7)
    ax2.set_ylabel('Mean RMSE (km/s)', fontsize=12)
    ax2.set_title('Average Prediction Error by Model', fontsize=14, fontweight='bold')
    
    for bar, rmse in zip(bars, rmses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rmse:.1f}', ha='center', fontsize=14, fontweight='bold')
    
    improvement = (rmses[0] - rmses[1]) / rmses[0] * 100
    ax2.text(0.5, max(rmses)*0.7, f'Improvement:\n{improvement:.1f}%', 
             ha='center', fontsize=14, fontweight='bold',
             transform=ax2.transAxes,
             bbox=dict(facecolor='yellow', alpha=0.3))
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figure3_model_comparison.png'), dpi=150)
    print("Saved: figure3_model_comparison.png")

def create_figure4_digital_horizon(galaxies):
    """Figure 4: The Digital Horizon - Signal vs Noise."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bit accuracy data from the original analysis
    bit_labels = ['Bit 0\n(LSB)', 'Bit 1', 'Bit 2', 'Bit 3', 'Bit 4', 'Bit 5', 'Bit 6', 'Bit 7\n(MSB)']
    # These are approximate values from the super-batch analysis
    single_formula_acc = [72.7, 74.1, 75.4, 72.7, 74.4, 79.5, 83.5, 100.0]
    
    colors = ['#FF6B6B' if acc < 80 else '#4ECDC4' if acc < 95 else '#45B7D1' for acc in single_formula_acc]
    
    bars = ax.bar(bit_labels, single_formula_acc, color=colors, alpha=0.8, edgecolor='black')
    
    ax.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Perfect Accuracy')
    ax.axhline(y=80, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='80% Threshold')
    
    ax.set_ylabel('Universal Formula Accuracy (%)', fontsize=12)
    ax.set_xlabel('Bit Position in Observed Velocity', fontsize=12)
    ax.set_title('The Digital Horizon: Signal vs Noise Cliff', fontsize=14, fontweight='bold')
    ax.set_ylim([60, 105])
    
    # Annotate the regions
    ax.annotate('Noise Floor\n(Spaghetti Logic)', xy=(2, 74), fontsize=11, ha='center',
               bbox=dict(facecolor='#FF6B6B', alpha=0.3))
    ax.annotate('Macro Law\n(Carry-Out)', xy=(7, 100), xytext=(6.5, 95), fontsize=11, ha='center',
               arrowprops=dict(arrowstyle='->', color='green'),
               bbox=dict(facecolor='#45B7D1', alpha=0.3))
    
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figure4_digital_horizon.png'), dpi=150)
    print("Saved: figure4_digital_horizon.png")

def create_all_figures():
    """Generate all figures for the paper."""
    print("Loading galaxy data...")
    galaxies = load_all_galaxies()
    print(f"Loaded {len(galaxies)} galaxies")
    
    print("\nGenerating figures...")
    create_figure1_rotation_curves(galaxies)
    create_figure2_overflow_detection(galaxies)
    create_figure3_model_comparison(galaxies)
    create_figure4_digital_horizon(galaxies)
    
    print("\nAll figures generated successfully!")

if __name__ == "__main__":
    create_all_figures()
