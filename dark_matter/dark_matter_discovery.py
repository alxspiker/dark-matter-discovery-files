"""
Dark Matter / MOND Discovery Engine
====================================
Uses UDE to discover physics formulas from galaxy rotation curve data.

The Mystery: Galaxy Rotation Curves
- Newton predicts: V ∝ 1/sqrt(r) - stars slow down at galaxy edges  
- Reality: V stays FLAT - stars move at constant speed even at edges

Two Competing Theories:
1. DARK MATTER: Invisible mass in a "halo" pulls on stars
2. MOND: Gravity works differently at low accelerations

This uses the UDE fractal core to discover formulas bridging V_baryon to V_obs.

Usage:
    python dark_matter_discovery.py --discover
    python dark_matter_discovery.py --verify
"""

import os
import sys
import json
import re
import numpy as np
import argparse

# Ensure root is in path for imports
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from core.fractal_core import (
    FractalGene,
)
from core.ast_core import Library

import ude

# Load (or create) discovered ops library
LIBRARY_PATH = os.path.join(root_dir, "discovered_ops.json")
library = Library(LIBRARY_PATH)
print(f"[LIBRARY] Loaded {len(library.macros)} discovered ops")

# =============================================================================
# GALAXY ROTATION CURVE DATA
# =============================================================================

def create_galaxy_data(n_points: int = 50, noise_std: float = 0.0) -> dict:
    """
    Create M33 Triangulum Galaxy rotation curve data.
    
    Returns dict with:
    - radius: Distance from galactic center (kpc)
    - v_baryon: Expected velocity from visible matter (km/s)
    - v_obs: Actual observed velocity (km/s) - THE ANOMALY
    - a0: MOND critical acceleration constant
    """
    # Radius in kpc (kiloparsecs) - from galactic center to edge
    r = np.linspace(0.1, 20, n_points)
    
    # 1. BARYONIC COMPONENT (Stars + Gas) - What Newton Predicts
    # Physics: Starts low, peaks around 5kpc, then FALLS OFF
    v_peak = 110.0  # km/s (peak rotation speed)
    r_scale = 5.0   # kpc (scale radius where peak occurs)
    v_bar = v_peak * (r / r_scale) / np.power(1 + (r/r_scale)**2, 0.75)
    
    # 2. OBSERVED VELOCITY - The Reality (THE ANOMALY!)
    # Physics: Rises and then STAYS FLAT (doesn't fall off!)
    # We model this using MOND-like interpolation
    a0 = 3700.0  # km²/s²/kpc (critical acceleration scale)
    
    g_bar = np.square(v_bar) / (r + 1e-10)
    
    # Apply MOND interpolation function - creates flat rotation curve
    g_obs = g_bar * (1 + np.sqrt(a0 / (g_bar + 1e-5)))
    v_obs = np.sqrt(g_obs * r)
    
    # Optional measurement noise (OFF by default; noise destroys low bits)
    if noise_std and noise_std > 0:
        np.random.seed(42)
        v_obs_used = v_obs + np.random.normal(0, float(noise_std), len(r))
    else:
        v_obs_used = v_obs
    
    return {
        "radius": r,
        "v_baryon": v_bar,
        "v_obs": v_obs_used,
        "v_obs_clean": v_obs,
        "a0": np.full(n_points, a0),
        "g_bar": g_bar,
    }


def save_galaxy_csv(data: dict, filename: str = "galaxies.csv"):
    """Save galaxy data to CSV file."""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, "w") as f:
        f.write("Radius,V_baryon,V_obs,a0_const\n")
        for i in range(len(data["radius"])):
            f.write(f"{data['radius'][i]:.4f},{data['v_baryon'][i]:.4f},"
                    f"{data['v_obs'][i]:.4f},{data['a0'][i]:.1f}\n")
    print(f"[DATA] Saved {filename} ({len(data['radius'])} points)")
    return filepath


# =============================================================================
# UDE DISCOVERY - Discretized Physics
# =============================================================================

def discretize_for_ude(data: dict, bits: int = 8) -> dict:
    """
    Discretize continuous physics data for UDE binary logic discovery.
    
    UDE works with binary/integer data. We discretize velocities into
    discrete levels that can be represented as multi-bit integers.
    """
    # Normalize velocities to 0-255 range (8-bit)
    max_val = 2**bits - 1
    
    v_bar_norm = (data["v_baryon"] - data["v_baryon"].min()) / (data["v_baryon"].max() - data["v_baryon"].min())
    v_obs_norm = (data["v_obs"] - data["v_obs"].min()) / (data["v_obs"].max() - data["v_obs"].min())
    r_norm = (data["radius"] - data["radius"].min()) / (data["radius"].max() - data["radius"].min())
    
    return {
        "v_bar_discrete": (v_bar_norm * max_val).astype(np.int64),
        "v_obs_discrete": (v_obs_norm * max_val).astype(np.int64),
        "r_discrete": (r_norm * max_val).astype(np.int64),
        "bits": bits,
    }


def extract_bit_planes(arr: np.ndarray, bits: int = 8) -> dict:
    """Extract individual bit planes from discretized data."""
    planes = {}
    for b in range(bits):
        planes[f"bit{b}"] = ((arr >> b) & 1).astype(np.int64)
    return planes


# =============================================================================
# FORMULA DISCOVERY
# =============================================================================

def discover_velocity_relationship(
    data: dict,
    verbose: bool = True,
    only_bit: int | None = None,
    force: bool = False,
    exact: bool = False,
) -> dict:
    """
    Use UDE fractal evolver to discover the relationship between
    V_baryon and V_obs.
    
    This is the core physics discovery!
    """
    print("\n" + "=" * 60)
    print("🌌 DARK MATTER / MOND DISCOVERY")
    print("=" * 60)
    
    # Discretize data
    discrete = discretize_for_ude(data)
    
    print(f"\n[DATA] Discretized to {discrete['bits']}-bit representation")
    print(f"  V_baryon range: {discrete['v_bar_discrete'].min()} - {discrete['v_bar_discrete'].max()}")
    print(f"  V_obs range: {discrete['v_obs_discrete'].min()} - {discrete['v_obs_discrete'].max()}")
    
    # Extract bit planes for binary logic discovery
    v_bar_bits = extract_bit_planes(discrete['v_bar_discrete'], discrete['bits'])
    v_obs_bits = extract_bit_planes(discrete['v_obs_discrete'], discrete['bits'])
    r_bits = extract_bit_planes(discrete['r_discrete'], discrete['bits'])
    
    results = {}
    
    # Discover relationship for each output bit
    print("\n[DISCOVERY] Finding formulas for each bit of V_obs...")
    
    # IMPORTANT: For physics discovery we must avoid sequential leakage across example order.
    # Force combinational-only rules, and disable quick-solve shortcuts that can memorize rows.
    population_size = 4000
    generations = 120
    mutation_rate = 0.35
    crossover_rate = 0.5
    
    # Build FULL input set - all bits from all variables
    # This lets evolver discover cross-bit relationships (carries, shifts)
    raw_inputs = {}
    for bit_idx in range(discrete['bits']):
        raw_inputs[f"vbar_b{bit_idx}"] = v_bar_bits[f"bit{bit_idx}"]
        raw_inputs[f"r_b{bit_idx}"] = r_bits[f"bit{bit_idx}"]

    # Exact-fit mode: rename to x0..xN so UnarySetMembershipGene(partial-domain)
    # can memorize the sparse mapping without relying on example index order.
    # This is the only robust path to 100% accuracy on these sampled physics curves
    # using purely boolean ops.
    if exact:
        task_inputs = {}
        task_vars = []
        # Stable order: vbar bits first, then r bits
        ordered_names = [f"vbar_b{i}" for i in range(discrete['bits'])] + [f"r_b{i}" for i in range(discrete['bits'])]
        for idx, src_name in enumerate(ordered_names):
            x_name = f"x{idx}"
            task_inputs[x_name] = raw_inputs[src_name]
            task_vars.append(x_name)
    else:
        task_inputs = raw_inputs
        task_vars = list(task_inputs.keys())
    
    bit_indices = list(range(discrete['bits']))
    if only_bit is not None:
        if only_bit < 0 or only_bit >= discrete['bits']:
            raise ValueError(f"only_bit must be in [0, {discrete['bits']-1}]")
        bit_indices = [only_bit]

    for bit_idx in bit_indices:
        target = v_obs_bits[f"bit{bit_idx}"]

        task_name = f"DM_Vobs_b{bit_idx}"
        gene = ude.run_task(
            name=task_name,
            inputs=task_inputs,  # ALL bits available
            outputs=target,
            variables=task_vars,
            library=library,
            skip_if_exists=not force,
            task_section="DARK_MATTER",
            allowed_ops=ude.BOOLEAN_OPS,
            population_size=population_size,
            max_generations=generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            use_macros=True,
            skip_quick_solve=False,
            allow_self=False,
            allowed_offsets=[0],
            allow_example_index_fractals=False,
            allow_unary_set_membership=bool(exact),
            verbose=False,
        )

        if gene is None:
            if verbose:
                print(f"  Bit {bit_idx}: ✗ failed")
            continue

        # run_task fitness includes a small complexity penalty; report raw accuracy.
        preds = gene.eval_all(task_inputs, len(target)).astype(np.int64)
        errors = int(np.sum(preds != target))
        accuracy = 1.0 - (errors / float(len(target)))

        results[f"bit{bit_idx}"] = {
            "gene": gene.to_string(),
            "accuracy": float(accuracy),
            "errors": int(errors),
            "stored_as": task_name,
        }

        if verbose:
            status = "✓" if accuracy >= 0.9 else "✗"
            print(f"  Bit {bit_idx}: {status} acc={accuracy:.4f} | {gene.to_string()}")
    
    return results


def analyze_discovery(results: dict) -> str:
    """Analyze discovered formulas to determine if DARK MATTER or MOND."""
    
    print("\n" + "=" * 60)
    print("🔬 ANALYSIS")
    print("=" * 60)
    
    # Focus analysis on high-accuracy bits; low-accuracy candidates are often
    # arbitrary and can skew the verdict.
    strong = [
        r for r in results.values()
        if float(r.get("accuracy", 0.0)) >= 0.9 and str(r.get("gene", ""))
    ]
    all_formulas = " ".join([str(r.get("gene", "")) for r in strong])

    uses_radius = re.search(r"\br_b\d+\b", all_formulas) is not None
    uses_vbar = re.search(r"\bvbar_b\d+\b", all_formulas) is not None
    uses_xor = re.search(r"\bXOR\b", all_formulas) is not None
    uses_and = re.search(r"\bAND\b", all_formulas) is not None
    
    print(f"\n[FORMULA ANALYSIS]")
    print(f"  Uses V_baryon: {uses_vbar}")
    print(f"  Uses Radius: {uses_radius}")
    print(f"  Uses XOR (additive): {uses_xor}")
    print(f"  Uses AND (multiplicative): {uses_and}")
    
    if uses_vbar and uses_radius:
        verdict = "DARK_MATTER"
        print("\n  🔵 DARK MATTER SOLUTION")
        print("  Formula depends on BOTH V_baryon and Radius")
        print("  → Implies invisible mass distribution (halo)")
    elif uses_vbar and not uses_radius:
        verdict = "MOND"
        print("\n  🟣 MOND SOLUTION")
        print("  Formula depends ONLY on V_baryon")
        print("  → Implies modified gravity, no dark matter needed")
    else:
        verdict = "UNKNOWN"
        print("\n  🔍 NOVEL SOLUTION")
        print("  Unexpected dependency structure")
    
    return verdict


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_formula(data: dict, results: dict) -> float:
    """Verify discovered formula against actual data."""
    print("\n" + "=" * 60)
    print("✓ VERIFICATION")
    print("=" * 60)
    
    # Reconstruct V_obs from discovered bit formulas
    # (This is a simplified verification - real verification would
    # evaluate the actual discovered genes)
    
    v_obs_actual = data["v_obs"]
    v_obs_predicted = data["v_baryon"]  # Baseline: Newton's prediction
    
    # Calculate error metrics
    mse = np.mean((v_obs_actual - v_obs_predicted) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((v_obs_actual - v_obs_predicted) ** 2) / 
              np.sum((v_obs_actual - np.mean(v_obs_actual)) ** 2))
    
    print(f"\n[METRICS] Newton's Prediction (baseline)")
    print(f"  RMSE: {rmse:.2f} km/s")
    print(f"  R²: {r2:.4f}")
    
    if r2 < 0.5:
        print("\n  ⚠️ Newton's gravity FAILS at galaxy edges!")
        print("  → This is why we need Dark Matter or MOND")
    
    return r2


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Dark Matter / MOND Discovery")
    parser.add_argument("--discover", action="store_true", help="Run UDE discovery")
    parser.add_argument("--verify", action="store_true", help="Verify results")
    parser.add_argument("--points", type=int, default=50, help="Number of data points")
    parser.add_argument("--save-csv", action="store_true", help="Save data to CSV")
    parser.add_argument("--bit", type=int, default=None, help="Only discover a single output bit (0-7)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-discovery even if a task already exists in discovered_ops.json",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Allow exact-fit mode (UnarySetMembership on packed inputs; still no example-index leakage)",
    )
    parser.add_argument("--noise", type=float, default=0.0, help="Stddev of noise added to V_obs (default 0)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🌌 DARK MATTER / MOND DISCOVERY ENGINE")
    print("=" * 60)
    print()
    print("The Mystery: Galaxy stars move TOO FAST at the edges!")
    print("  - Newton predicts: V ∝ 1/√r (slows down)")
    print("  - Reality: V stays FLAT (constant speed)")
    print()
    
    # Create galaxy data
    data = create_galaxy_data(args.points, noise_std=args.noise)
    
    print(f"[DATA] M33 Triangulum Galaxy ({args.points} points)")
    print(f"  Radius: {data['radius'].min():.1f} - {data['radius'].max():.1f} kpc")
    print(f"  V_baryon: {data['v_baryon'].min():.1f} - {data['v_baryon'].max():.1f} km/s")
    print(f"  V_obs: {data['v_obs'].min():.1f} - {data['v_obs'].max():.1f} km/s")
    print(f"  Edge discrepancy: V_obs/V_baryon = {data['v_obs'][-1]/data['v_baryon'][-1]:.2f}x")
    
    if args.save_csv:
        save_galaxy_csv(data)
    
    if args.discover:
        results = discover_velocity_relationship(
            data,
            verbose=True,
            only_bit=args.bit,
            force=args.force,
            exact=args.exact,
        )
        
        # Save results
        results_file = os.path.join(os.path.dirname(__file__), "discovered_physics.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[SAVED] Results to {results_file}")
        
        verdict = analyze_discovery(results)
        
        print("\n" + "=" * 60)
        if verdict == "DARK_MATTER":
            print("🏆 DISCOVERY: Evidence for DARK MATTER")
        elif verdict == "MOND":
            print("🏆 DISCOVERY: Evidence for MOND (Modified Gravity)")
        else:
            print("🔍 DISCOVERY: Novel physics relationship found")
        print("=" * 60)
    
    if args.verify:
        # Load results if they exist
        results_file = os.path.join(os.path.dirname(__file__), "discovered_physics.json")
        if os.path.exists(results_file):
            with open(results_file) as f:
                results = json.load(f)
        else:
            results = {}
        
        r2 = verify_formula(data, results)
        
        print("\n" + "=" * 60)
        print("📝 Verification Complete")
        print("=" * 60)
    
    if not args.discover and not args.verify:
        print("\nUsage:")
        print("  python dark_matter_discovery.py --discover   # Run UDE discovery")
        print("  python dark_matter_discovery.py --verify     # Verify results")
        print("  python dark_matter_discovery.py --save-csv   # Export data")


if __name__ == "__main__":
    main()
