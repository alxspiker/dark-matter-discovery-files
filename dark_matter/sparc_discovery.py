"""
SPARC Galaxy Discovery Engine
=============================
Uses UDE to discover physics formulas from REAL galaxy rotation curve data.

This is the critical generalization test:
- Run UDE discovery BLIND on 175 real SPARC galaxies
- See if the same bit-level patterns emerge (latch on b7, carry on b6)
- If patterns are universal → real physics discovered
- If patterns diverge → dataset-specific overfitting

Usage:
    python sparc_discovery.py --discover --dat sparc_galaxies/NGC2403_rotmod.dat
    python sparc_discovery.py --discover --batch  # All galaxies
    python sparc_discovery.py --discover --batch --exact  # With exact fit mode
    python sparc_discovery.py --analyze  # Analyze discovered patterns
"""

import os
import sys
import json
import re
import numpy as np
import argparse
import glob
from collections import Counter

# Ensure root is in path for imports
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from core.fractal_core import FractalGene
from core.ast_core import Library
import ude

# Load library
LIBRARY_PATH = os.path.join(root_dir, "discovered_ops.json")
library = Library(LIBRARY_PATH)
print(f"[LIBRARY] Loaded {len(library.macros)} discovered ops")

# =============================================================================
# DATA LOADING
# =============================================================================

def load_sparc_dat(
    path: str,
    *,
    col_radius: str = "Rad",
    col_vobs: str = "Vobs",
    col_vgas: str = "Vgas",
    col_vdisk: str = "Vdisk",
    col_vbul: str = "Vbul",
) -> dict:
    """Load a SPARC Rotmod_LTG .dat file."""
    header_cols = None
    data_lines = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                if s.startswith("# Rad"):
                    header_cols = re.split(r"\s+", s.lstrip("#").strip())
                continue
            data_lines.append(s)

    if not data_lines:
        raise ValueError(f"No data in {path}")
    if header_cols is None:
        header_cols = ["Rad", "Vobs", "errV", "Vgas", "Vdisk", "Vbul"]

    idx = {name: i for i, name in enumerate(header_cols)}

    def col(name: str) -> int:
        if name not in idx:
            raise ValueError(f"Missing column '{name}' in {path}")
        return idx[name]

    radius, vobs, vgas, vdisk, vbul = [], [], [], [], []

    for line in data_lines:
        parts = re.split(r"\s+", line)
        radius.append(float(parts[col(col_radius)]))
        vobs.append(float(parts[col(col_vobs)]))
        vgas.append(float(parts[col(col_vgas)]))
        vdisk.append(float(parts[col(col_vdisk)]))
        vbul.append(float(parts[col(col_vbul)]))

    vbar = np.sqrt(
        np.square(np.array(vgas)) +
        np.square(np.array(vdisk)) +
        np.square(np.array(vbul))
    )

    return {
        "radius": np.array(radius, dtype=np.float64),
        "v_baryon": vbar,
        "v_obs": np.array(vobs, dtype=np.float64),
    }


def create_synthetic_m33(n_points: int = 50) -> dict:
    """Create synthetic M33-like rotation curve for reference scaling."""
    r = np.linspace(0.1, 20, n_points)
    v_peak = 110.0
    r_scale = 5.0
    v_bar = v_peak * (r / r_scale) / np.power(1 + (r / r_scale) ** 2, 0.75)
    a0 = 3700.0
    g_bar = np.square(v_bar) / (r + 1e-10)
    g_obs = g_bar * (1 + np.sqrt(a0 / (g_bar + 1e-5)))
    v_obs = np.sqrt(g_obs * r)
    return {
        "radius": r,
        "v_baryon": v_bar,
        "v_obs": v_obs,
        "scaling": {
            "r_min": r.min(),
            "r_max": r.max(),
            "vbar_min": v_bar.min(),
            "vbar_max": v_bar.max(),
            "vobs_min": v_obs.min(),
            "vobs_max": v_obs.max(),
        },
    }


# =============================================================================
# DISCRETIZATION
# =============================================================================

def discretize_for_ude(data: dict, bits: int = 8, scaling: dict | None = None) -> dict:
    """Convert continuous velocity data to discrete bit planes."""
    if scaling is None:
        scaling = {
            "r_min": data["radius"].min(),
            "r_max": data["radius"].max(),
            "vbar_min": data["v_baryon"].min(),
            "vbar_max": data["v_baryon"].max(),
            "vobs_min": data["v_obs"].min(),
            "vobs_max": data["v_obs"].max(),
        }

    max_val = 2 ** bits - 1

    def norm(arr, mn, mx):
        return np.clip((arr - mn) / (mx - mn + 1e-10), 0, 1) * max_val

    return {
        "v_bar_discrete": norm(data["v_baryon"], scaling["vbar_min"], scaling["vbar_max"]).astype(np.int64),
        "v_obs_discrete": norm(data["v_obs"], scaling["vobs_min"], scaling["vobs_max"]).astype(np.int64),
        "r_discrete": norm(data["radius"], scaling["r_min"], scaling["r_max"]).astype(np.int64),
        "bits": bits,
        "scaling": scaling,
    }


def extract_bit_planes(arr: np.ndarray, bits: int = 8) -> dict:
    """Extract individual bit planes from integer array."""
    return {f"bit{b}": ((arr >> b) & 1).astype(np.int64) for b in range(bits)}


# =============================================================================
# DISCOVERY ENGINE
# =============================================================================

# Common cross-galaxy patterns to try first (from 10-galaxy analysis)
# Format: {bit_index: [(pattern_desc, accuracy_threshold), ...]}
CROSS_GALAXY_HINTS = {
    7: [("Add8_Cout", "RippleAdderCout"), ("b6 OR b7", None)],
    6: [("Add8_S7", "RippleAdderSumBit"), ("Add8_S5", "RippleAdderSumBit")],
}


def discover_velocity_relationship(
    data: dict,
    galaxy_name: str,
    verbose: bool = True,
    only_bit: int | None = None,
    force: bool = False,
    exact: bool = False,
    scaling: dict | None = None,
    max_points: int | None = None,
    quick_accept: float | None = None,
) -> dict:
    """
    Run UDE discovery to find formulas predicting v_obs bits from v_bar bits.
    
    This is the core experiment: can UDE rediscover the same bit-level
    physics patterns across different galaxies?
    
    Optimizations:
    - max_points: Subsample large galaxies (default: None = use all)
    - quick_accept: Accept quick-solve at this accuracy (default: 0.95 for >20 points)
    """
    n_points = len(data['radius'])
    
    # =========================================================================
    # OPTIMIZATION 1: Subsample large galaxies for faster exact solutions
    # Galaxies with 30+ points often can't find exact patterns; subsample to 16
    # =========================================================================
    if max_points is not None and n_points > max_points:
        print(f"[SUBSAMPLE] {n_points} points -> {max_points} (uniformly spaced)")
        indices = np.linspace(0, n_points - 1, max_points, dtype=int)
        data = {
            "radius": data["radius"][indices],
            "v_baryon": data["v_baryon"][indices],
            "v_obs": data["v_obs"][indices],
        }
        n_points = max_points
    
    # =========================================================================
    # OPTIMIZATION 2: Auto quick_accept threshold for large datasets
    # Don't waste 60+ seconds on evolution if quick-solve hits 95%+
    # =========================================================================
    if quick_accept is None and n_points > 20:
        quick_accept = 0.95
        print(f"[AUTO] Large galaxy ({n_points} points): quick_accept=95%")
    
    print("\n" + "=" * 60)
    print(f"🌌 DISCOVERY: {galaxy_name} ({n_points} points)")
    print("=" * 60)

    discrete = discretize_for_ude(data, scaling=scaling)
    v_bar_bits = extract_bit_planes(discrete["v_bar_discrete"])
    v_obs_bits = extract_bit_planes(discrete["v_obs_discrete"])
    r_bits = extract_bit_planes(discrete["r_discrete"])

    # Build input variables
    raw_inputs = {}
    for bit_idx in range(discrete["bits"]):
        raw_inputs[f"vbar_b{bit_idx}"] = v_bar_bits[f"bit{bit_idx}"]
        raw_inputs[f"r_b{bit_idx}"] = r_bits[f"bit{bit_idx}"]

    # =========================================================================
    # OPTION 1: Variable Renaming for Macro Compatibility
    # Rename vbar_b0..b7 -> a0..a7 and r_b0..b7 -> b0..b7
    # This enables learned macros like Add8_S0, Mul2_P0, etc. to match!
    # =========================================================================
    bits = discrete["bits"]
    
    if exact:
        # For exact mode, use x0..xN naming (required for UnarySetMembership)
        task_inputs = {}
        task_vars = []
        ordered = [f"vbar_b{i}" for i in range(bits)] + \
                  [f"r_b{i}" for i in range(bits)]
        for idx, name in enumerate(ordered):
            task_inputs[f"x{idx}"] = raw_inputs[name]
            task_vars.append(f"x{idx}")
    elif bits == 8:
        # Option 1: Rename to a0..a7, b0..b7 for macro compatibility
        print("[OPTION 1] Renaming vbar->a0..a7, r->b0..b7 for macro compatibility")
        task_inputs = {}
        for b in range(bits):
            task_inputs[f"a{b}"] = raw_inputs[f"vbar_b{b}"]
            task_inputs[f"b{b}"] = raw_inputs[f"r_b{b}"]
        task_vars = [f"a{i}" for i in range(bits)] + [f"b{i}" for i in range(bits)]
    else:
        task_inputs = raw_inputs
        task_vars = list(task_inputs.keys())

    results = {}
    bit_indices = list(range(discrete["bits"])) if only_bit is None else [only_bit]

    for bit_idx in bit_indices:
        target = v_obs_bits[f"bit{bit_idx}"]
        # Unique task name per galaxy to avoid collisions
        task_name = f"SPARC_{galaxy_name.replace('-', '_').replace(' ', '_')}_b{bit_idx}"

        gene = ude.run_task(
            name=task_name,
            inputs=task_inputs,
            outputs=target,
            variables=task_vars,
            library=library,
            skip_if_exists=not force,
            task_section="DARK_MATTER_SPARC",
            allowed_ops=ude.BOOLEAN_OPS,
            population_size=4000,
            max_generations=120,
            mutation_rate=0.35,
            crossover_rate=0.5,
            use_macros=True,
            skip_quick_solve=False,
            allow_self=False,
            allowed_offsets=[0],
            allow_example_index_fractals=False,
            allow_unary_set_membership=exact,
            quick_accept_threshold=quick_accept,  # OPTIMIZATION: Accept quick-solve at threshold
            verbose=False,
        )

        if gene is None:
            if verbose:
                print(f"  Bit {bit_idx}: ✗ failed")
            results[f"bit{bit_idx}"] = {"gene": None, "accuracy": 0.0, "stored_as": task_name}
            continue

        preds = gene.eval_all(task_inputs, len(target)).astype(np.int64)
        accuracy = np.mean(preds == target)

        results[f"bit{bit_idx}"] = {
            "gene": gene.to_string(),
            "accuracy": float(accuracy),
            "stored_as": task_name,
        }

        if verbose:
            status = "✓" if accuracy >= 0.9 else "~" if accuracy >= 0.7 else "?"
            gene_str = gene.to_string()
            # Highlight key patterns that would indicate universal physics
            pattern_hints = []
            if "x7" in gene_str or "vbar_b7" in gene_str:
                pattern_hints.append("MSB")
            if "OR" in gene_str:
                pattern_hints.append("latch?")
            if "AND" in gene_str and "XOR" in gene_str:
                pattern_hints.append("carry?")
            if "UnarySetMembership" in gene_str:
                pattern_hints.append("lookup")
            hint = f" [{', '.join(pattern_hints)}]" if pattern_hints else ""
            print(f"  Bit {bit_idx}: {status} acc={accuracy:.3f} | {gene_str}{hint}")

    # Save per-galaxy results
    out_dir = os.path.join(os.path.dirname(__file__), "sparc_discovered")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{galaxy_name}_discovered.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {galaxy_name} → {out_path}")

    return results


# =============================================================================
# PATTERN ANALYSIS
# =============================================================================

def analyze_discovered_patterns(results_dir: str) -> None:
    """
    Analyze all discovered patterns across galaxies for universality.
    
    KEY QUESTION: Do the same formulas emerge for each bit across different galaxies?
    - If yes → Universal physics (e.g., latch on b7 is real)
    - If no → Dataset-specific overfitting
    """
    json_files = glob.glob(os.path.join(results_dir, "*_discovered.json"))
    if not json_files:
        print(f"[ERROR] No discovered results in {results_dir}")
        return

    print("\n" + "=" * 60)
    print("📊 CROSS-GALAXY PATTERN ANALYSIS")
    print("=" * 60)

    bit_patterns = {f"bit{i}": [] for i in range(8)}
    bit_accuracies = {f"bit{i}": [] for i in range(8)}

    for path in json_files:
        galaxy = os.path.basename(path).replace("_discovered.json", "")
        with open(path, "r") as f:
            data = json.load(f)
        for bit_key in bit_patterns:
            if bit_key in data and data[bit_key]["gene"]:
                bit_patterns[bit_key].append((galaxy, data[bit_key]["gene"]))
                bit_accuracies[bit_key].append(data[bit_key]["accuracy"])

    print(f"\nAnalyzed {len(json_files)} galaxies\n")

    # Summary table
    print("BIT | AVG_ACC | GALAXIES | UNIQUE_FORMULAS | TOP_FORMULA")
    print("-" * 80)

    for bit_idx in range(7, -1, -1):  # MSB first
        bit_key = f"bit{bit_idx}"
        patterns = bit_patterns[bit_key]
        accs = bit_accuracies[bit_key]

        if not patterns:
            print(f" {bit_idx}  |   N/A   |    0     |        0        | (no data)")
            continue

        avg_acc = np.mean(accs)
        unique_genes = set(p[1] for p in patterns)
        gene_counts = Counter(p[1] for p in patterns)
        top_gene, top_count = gene_counts.most_common(1)[0]
        top_pct = top_count / len(patterns) * 100

        # Truncate formula for display
        top_gene_short = top_gene[:40] + "..." if len(top_gene) > 40 else top_gene

        print(f" {bit_idx}  | {avg_acc:6.3f} | {len(patterns):8d} | {len(unique_genes):15d} | {top_pct:5.1f}%: {top_gene_short}")

    # Detailed breakdown for high bits (most interesting)
    print("\n" + "=" * 60)
    print("🔬 DETAILED BIT 7 (MSB) FORMULAS")
    print("=" * 60)

    if bit_patterns["bit7"]:
        gene_counts = Counter(p[1] for p in bit_patterns["bit7"])
        for gene, count in gene_counts.most_common(10):
            pct = count / len(bit_patterns["bit7"]) * 100
            # List galaxies using this formula
            galaxies = [p[0] for p in bit_patterns["bit7"] if p[1] == gene][:5]
            print(f"\n{pct:5.1f}% ({count:3d} galaxies): {gene}")
            print(f"       Examples: {', '.join(galaxies)}")

    print("\n" + "=" * 60)
    print("🔬 DETAILED BIT 6 FORMULAS")
    print("=" * 60)

    if bit_patterns["bit6"]:
        gene_counts = Counter(p[1] for p in bit_patterns["bit6"])
        for gene, count in gene_counts.most_common(10):
            pct = count / len(bit_patterns["bit6"]) * 100
            galaxies = [p[0] for p in bit_patterns["bit6"] if p[1] == gene][:5]
            print(f"\n{pct:5.1f}% ({count:3d} galaxies): {gene}")
            print(f"       Examples: {', '.join(galaxies)}")


def export_summary_csv(results_dir: str, out_path: str) -> None:
    """Export analysis to CSV for further processing."""
    json_files = glob.glob(os.path.join(results_dir, "*_discovered.json"))
    if not json_files:
        print(f"[ERROR] No discovered results in {results_dir}")
        return

    rows = []
    for path in json_files:
        galaxy = os.path.basename(path).replace("_discovered.json", "")
        with open(path, "r") as f:
            data = json.load(f)
        for bit_idx in range(8):
            bit_key = f"bit{bit_idx}"
            if bit_key in data:
                rows.append({
                    "galaxy": galaxy,
                    "bit": bit_idx,
                    "accuracy": data[bit_key].get("accuracy", 0),
                    "formula": data[bit_key].get("gene", ""),
                })

    import csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["galaxy", "bit", "accuracy", "formula"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[EXPORT] Wrote {len(rows)} rows to {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SPARC Galaxy Discovery Engine")
    parser.add_argument("--discover", action="store_true", help="Run UDE discovery")
    parser.add_argument("--dat", type=str, default=None, help="Single SPARC .dat file path")
    parser.add_argument("--batch", action="store_true", help="Discover on all galaxies in sparc_galaxies/")
    parser.add_argument("--scaling", choices=["per-galaxy", "m33"], default="per-galaxy",
                        help="per-galaxy = fair blind test, m33 = use M33 normalization")
    parser.add_argument("--bit", type=int, default=None, help="Only discover for specific bit")
    parser.add_argument("--force", action="store_true", help="Force rediscovery even if cached")
    parser.add_argument("--exact", action="store_true", help="Enable exact-fit mode (UnarySetMembership)")
    parser.add_argument("--analyze", action="store_true", help="Analyze already-discovered patterns")
    parser.add_argument("--export-csv", type=str, default=None, help="Export analysis to CSV")
    parser.add_argument("--limit", type=int, default=None, help="Limit batch to N galaxies (for testing)")
    # Optimization arguments
    parser.add_argument("--max-points", type=int, default=None,
                        help="Subsample large galaxies to N points (e.g., --max-points 16)")
    parser.add_argument("--quick-accept", type=float, default=None,
                        help="Accept quick-solve at this accuracy (0.0-1.0, e.g., --quick-accept 0.95)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: --max-points 16 --quick-accept 0.95")
    args = parser.parse_args()

    # Fast mode shortcut
    if args.fast:
        if args.max_points is None:
            args.max_points = 16
        if args.quick_accept is None:
            args.quick_accept = 0.95
        print("[FAST MODE] max_points=16, quick_accept=95%")

    sparc_dir = os.path.join(os.path.dirname(__file__), "sparc_galaxies")
    discovered_dir = os.path.join(os.path.dirname(__file__), "sparc_discovered")

    # Analysis mode
    if args.analyze:
        analyze_discovered_patterns(discovered_dir)
        if args.export_csv:
            export_summary_csv(discovered_dir, args.export_csv)
        return

    # Export only
    if args.export_csv and not args.discover:
        export_summary_csv(discovered_dir, args.export_csv)
        return

    # Get M33 scaling if needed
    m33_data = create_synthetic_m33()
    m33_scaling = m33_data["scaling"] if args.scaling == "m33" else None

    # Batch mode
    if args.batch:
        dat_files = sorted(glob.glob(os.path.join(sparc_dir, "*_rotmod.dat")))
        if not dat_files:
            print(f"[ERROR] No .dat files in {sparc_dir}")
            return

        if args.limit:
            dat_files = dat_files[:args.limit]

        print(f"[BATCH] Discovering on {len(dat_files)} SPARC galaxies")
        print(f"[MODE] scaling={args.scaling}, exact={args.exact}, force={args.force}")
        if args.max_points:
            print(f"[OPTIM] max_points={args.max_points}")
        if args.quick_accept:
            print(f"[OPTIM] quick_accept={args.quick_accept*100:.0f}%")

        for i, path in enumerate(dat_files):
            name = os.path.basename(path).replace("_rotmod.dat", "")
            print(f"\n[{i+1}/{len(dat_files)}] {name}")

            try:
                data = load_sparc_dat(path)
            except Exception as e:
                print(f"[SKIP] {name}: {e}")
                continue

            discover_velocity_relationship(
                data,
                galaxy_name=name,
                verbose=True,
                only_bit=args.bit,
                force=args.force,
                exact=args.exact,
                scaling=m33_scaling,
                max_points=args.max_points,
                quick_accept=args.quick_accept,
            )

        # Auto-analyze after batch
        print("\n" + "=" * 60)
        analyze_discovered_patterns(discovered_dir)
        return

    # Single galaxy mode
    if args.dat:
        data = load_sparc_dat(args.dat)
        galaxy_name = os.path.basename(args.dat).replace("_rotmod.dat", "")
    else:
        print("Usage:")
        print("  Single galaxy:  python sparc_discovery.py --discover --dat sparc_galaxies/NGC2403_rotmod.dat")
        print("  Batch (all):    python sparc_discovery.py --discover --batch")
        print("  With exact fit: python sparc_discovery.py --discover --batch --exact")
        print("  Analyze:        python sparc_discovery.py --analyze")
        print("  Quick test:     python sparc_discovery.py --discover --batch --limit 5")
        print("  Fast mode:      python sparc_discovery.py --discover --batch --fast")
        return

    if args.discover:
        discover_velocity_relationship(
            data,
            galaxy_name=galaxy_name,
            verbose=True,
            only_bit=args.bit,
            force=args.force,
            exact=args.exact,
            scaling=m33_scaling,
            max_points=args.max_points,
            quick_accept=args.quick_accept,
        )


if __name__ == "__main__":
    main()
