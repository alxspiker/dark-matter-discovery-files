#!/usr/bin/env python3
"""
SUPER-BATCH CROSS-GALAXY DISCOVERY
===================================
Takes galaxies that agreed on Add8_Cout for bit 7 and forces them into
a cage match. Concatenates their data into one massive dataset so the AI
cannot memorize individual galaxy noise - it must find the UNIVERSAL pattern.

The hypothesis: "Spaghetti Logic" in bits 4-6 will collapse into clean
RippleAdder or Mux structures because those are the only logical gates
that generalize across arithmetic operations.
"""

import sys
import os
import json
import glob
import numpy as np
import argparse

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import UDE and library
import ude
from core.ast_core import Library

# Load library
LIBRARY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "discovered_ops.json")
library = Library(LIBRARY_PATH)
print(f"[LIBRARY] Loaded {len(library.macros)} discovered ops")

# Import from sparc_discovery - use the actual function names
from dark_matter.sparc_discovery import (
    load_sparc_dat, 
    discretize_for_ude,
    extract_bit_planes,
)

# Get SPARC directory
SPARC_DIR = os.path.join(os.path.dirname(__file__), "sparc_galaxies")

# The 31 galaxies that independently discovered Add8_Cout on bit 7
COUT_GALAXIES = [
    "D512-2", "D564-8", "DDO064", "DDO168", "ESO116-G012",
    "ESO444-G084", "F571-V1", "NGC3109", "NGC3949", "NGC4085",
    "NGC4389", "NGC5005", "NGC5585", "UGC00891", "UGC02023",
    "UGC02259", "UGC04278", "UGC04483", "UGC05005", "UGC05414",
    "UGC05721", "UGC06923", "UGC07089", "UGC07261", "UGC07323",
    "UGC07866", "UGC08837", "UGC11820", "UGCA281", "UGCA442",
    "UGCA444"
]


def load_and_concatenate_galaxies(galaxy_names, max_points_per_galaxy=16):
    """
    Load and concatenate data from multiple galaxies into one mega-dataset.
    
    Args:
        galaxy_names: List of galaxy names to include
        max_points_per_galaxy: Subsample each galaxy to this many points
        
    Returns:
        Dictionary with concatenated bit arrays and metadata
    """
    all_vbar_bits = []  # List of (n_points, 8) arrays
    all_r_bits = []
    all_vobs_bits = []
    galaxy_sources = []  # Track which points came from which galaxy
    
    # Find .dat files
    dat_files = glob.glob(os.path.join(SPARC_DIR, "*_rotmod.dat"))
    name_to_file = {}
    for f in dat_files:
        name = os.path.basename(f).replace("_rotmod.dat", "")
        name_to_file[name] = f
    
    loaded_count = 0
    for gname in galaxy_names:
        if gname not in name_to_file:
            print(f"  [WARN] Galaxy {gname} not found in {SPARC_DIR}")
            continue
            
        try:
            data = load_sparc_dat(name_to_file[gname])
            discrete = discretize_for_ude(data, bits=8)
            
            n_points = len(discrete["v_bar_discrete"])
            
            # Subsample if needed
            if n_points > max_points_per_galaxy:
                indices = np.linspace(0, n_points - 1, max_points_per_galaxy, dtype=int)
            else:
                indices = np.arange(n_points)
            
            vbar_d = discrete["v_bar_discrete"][indices]
            r_d = discrete["r_discrete"][indices]
            vobs_d = discrete["v_obs_discrete"][indices]
            
            # Extract bits for each point
            n_sel = len(indices)
            vbar_bits = np.zeros((n_sel, 8), dtype=np.int64)
            r_bits = np.zeros((n_sel, 8), dtype=np.int64)
            vobs_bits = np.zeros((n_sel, 8), dtype=np.int64)
            
            for b in range(8):
                vbar_bits[:, b] = (vbar_d >> b) & 1
                r_bits[:, b] = (r_d >> b) & 1
                vobs_bits[:, b] = (vobs_d >> b) & 1
            
            all_vbar_bits.append(vbar_bits)
            all_r_bits.append(r_bits)
            all_vobs_bits.append(vobs_bits)
            galaxy_sources.extend([gname] * n_sel)
            loaded_count += 1
            
        except Exception as e:
            print(f"  [ERROR] Failed to load {gname}: {e}")
            continue
    
    if loaded_count == 0:
        raise ValueError("No galaxies loaded!")
    
    # Concatenate all
    return {
        "vbar_bits": np.vstack(all_vbar_bits),  # (N_total, 8)
        "r_bits": np.vstack(all_r_bits),        # (N_total, 8)
        "vobs_bits": np.vstack(all_vobs_bits),  # (N_total, 8)
        "sources": galaxy_sources,
        "n_galaxies": loaded_count,
        "n_points": len(galaxy_sources),
    }


def test_known_patterns(data):
    """
    Test known patterns against the concatenated dataset.
    This is the critical test: do patterns found in individual galaxies
    actually generalize?
    """
    vbar = data["vbar_bits"]  # (N, 8) - a0..a7
    r = data["r_bits"]        # (N, 8) - b0..b7
    vobs = data["vobs_bits"]  # (N, 8) - target bits
    n_points = data["n_points"]
    
    print(f"\n{'='*60}")
    print(f"PATTERN GENERALIZATION TEST")
    print(f"{'='*60}")
    print(f"Testing on {n_points} points from {data['n_galaxies']} galaxies")
    print()
    
    results = {}
    
    # Test BIT 7: Add8_Cout (RippleAdder Carry Out)
    # Carry out of 8-bit addition: occurs when sum >= 256
    print("BIT 7: Testing Add8_Cout (RippleAdderCarryOut)")
    
    # We need to compute full addition to get carry out
    # Reconstruct 8-bit values from bits
    vbar_val = np.zeros(n_points, dtype=np.int64)
    r_val = np.zeros(n_points, dtype=np.int64)
    for b in range(8):
        vbar_val += vbar[:, b] << b
        r_val += r[:, b] << b
    
    # Carry out = 1 if (a + b) >= 256
    add_result = vbar_val + r_val
    cout_pred = (add_result >= 256).astype(np.int64)
    
    actual_bit7 = vobs[:, 7]
    acc_bit7 = np.mean(cout_pred == actual_bit7)
    print(f"  Add8_Cout accuracy: {acc_bit7:.4f} ({int(acc_bit7*n_points)}/{n_points})")
    results["bit7_Add8_Cout"] = acc_bit7
    
    # Also test simpler patterns
    # b7 OR b7
    pred_or = vbar[:, 7] | r[:, 7]
    acc_or = np.mean(pred_or == actual_bit7)
    print(f"  (a7 OR b7) accuracy: {acc_or:.4f}")
    results["bit7_OR"] = acc_or
    
    # Just a7
    acc_a7 = np.mean(vbar[:, 7] == actual_bit7)
    print(f"  Just a7 accuracy: {acc_a7:.4f}")
    results["bit7_a7"] = acc_a7
    
    print()
    
    # Test BIT 6: RippleAdderSumBit[7]
    # Sum bit 7 = a7 XOR b7 XOR carry_in_to_bit7
    print("BIT 6: Testing RippleAdderSumBit[7]")
    
    # We need to compute carry into bit 7
    # Carry chain: c_i = (a_{i-1} AND b_{i-1}) OR (c_{i-1} AND (a_{i-1} XOR b_{i-1}))
    # For sum bit 7: s7 = a7 XOR b7 XOR c7
    
    # Compute carry propagation
    carry = np.zeros(n_points, dtype=np.int64)
    for b in range(8):
        a_b = vbar[:, b]
        b_b = r[:, b]
        # Generate and propagate
        gen = a_b & b_b
        prop = a_b ^ b_b
        carry = gen | (prop & carry)
    
    # Actually need carry INTO bit 7, not out of bit 7
    carry_into_7 = np.zeros(n_points, dtype=np.int64)
    for b in range(7):  # 0 to 6
        a_b = vbar[:, b]
        b_b = r[:, b]
        gen = a_b & b_b
        prop = a_b ^ b_b
        carry_into_7 = gen | (prop & carry_into_7)
    
    sum_bit7 = vbar[:, 7] ^ r[:, 7] ^ carry_into_7
    
    actual_bit6 = vobs[:, 6]
    acc_bit6_sum7 = np.mean(sum_bit7 == actual_bit6)
    print(f"  RippleAdderSumBit[7] accuracy: {acc_bit6_sum7:.4f}")
    results["bit6_SumBit7"] = acc_bit6_sum7
    
    # Test simpler patterns for bit 6
    acc_xor = np.mean((vbar[:, 6] ^ r[:, 6]) == actual_bit6)
    print(f"  (a6 XOR b6) accuracy: {acc_xor:.4f}")
    results["bit6_XOR"] = acc_xor
    
    print()
    
    # Test BITS 4-5: These had "spaghetti logic" - let's see if simple patterns work
    for target_bit in [5, 4, 3]:
        print(f"BIT {target_bit}: Testing simple patterns")
        actual = vobs[:, target_bit]
        
        # Compute carry into this bit
        carry_into = np.zeros(n_points, dtype=np.int64)
        for b in range(target_bit):
            a_b = vbar[:, b]
            b_b = r[:, b]
            gen = a_b & b_b
            prop = a_b ^ b_b
            carry_into = gen | (prop & carry_into)
        
        sum_bit = vbar[:, target_bit] ^ r[:, target_bit] ^ carry_into
        acc_sum = np.mean(sum_bit == actual)
        print(f"  RippleAdderSumBit[{target_bit}] accuracy: {acc_sum:.4f}")
        results[f"bit{target_bit}_SumBit"] = acc_sum
        
        # Simple XOR
        acc_xor = np.mean((vbar[:, target_bit] ^ r[:, target_bit]) == actual)
        print(f"  (a{target_bit} XOR b{target_bit}) accuracy: {acc_xor:.4f}")
        
        # Just a
        acc_a = np.mean(vbar[:, target_bit] == actual)
        print(f"  Just a{target_bit} accuracy: {acc_a:.4f}")
        
        print()
    
    # Summary
    print("="*60)
    print("UNIVERSALITY VERDICT")
    print("="*60)
    
    if results["bit7_Add8_Cout"] > 0.90:
        print("✓ BIT 7: Add8_Cout GENERALIZES across galaxies!")
        print(f"  This is REAL PHYSICS - the carry-out pattern holds at {results['bit7_Add8_Cout']:.1%}")
    else:
        print(f"✗ BIT 7: Add8_Cout does NOT generalize ({results['bit7_Add8_Cout']:.1%})")
        print("  Individual galaxy fits were overfit.")
    
    if results.get("bit6_SumBit7", 0) > 0.80:
        print("✓ BIT 6: RippleAdderSumBit[7] GENERALIZES!")
    else:
        print(f"✗ BIT 6: RippleAdderSumBit[7] does not generalize ({results.get('bit6_SumBit7', 0):.1%})")
    
    return results


def run_superbatch_discovery(data, quick_accept=0.95):
    """
    Run full UDE discovery on the concatenated multi-galaxy dataset.
    This forces the AI to find patterns that generalize across ALL galaxies
    in the subset - no per-galaxy overfitting possible.
    """
    vbar = data["vbar_bits"]  # (N, 8) - a0..a7
    r = data["r_bits"]        # (N, 8) - b0..b7
    vobs = data["vobs_bits"]  # (N, 8) - target bits
    n_points = data["n_points"]
    
    print(f"\n{'='*60}")
    print(f"SUPER-BATCH UDE DISCOVERY")
    print(f"{'='*60}")
    print(f"Running on {n_points} points from {data['n_galaxies']} galaxies")
    print(f"Quick accept threshold: {quick_accept}")
    print()
    
    # Build input variables: a0..a7 (vbar bits), b0..b7 (radius bits)
    task_inputs = {}
    for b in range(8):
        task_inputs[f"a{b}"] = vbar[:, b]
        task_inputs[f"b{b}"] = r[:, b]
    task_vars = [f"a{i}" for i in range(8)] + [f"b{i}" for i in range(8)]
    
    results = {}
    
    # Discover patterns for each bit, starting from MSB (most likely to generalize)
    for bit_idx in [7, 6, 5, 4, 3, 2, 1, 0]:
        target = vobs[:, bit_idx]
        task_name = f"SUPERBATCH_{data['n_galaxies']}gal_b{bit_idx}"
        
        print(f"\n[BIT {bit_idx}] Discovering universal pattern...")
        
        gene = ude.run_task(
            name=task_name,
            inputs=task_inputs,
            outputs=target,
            variables=task_vars,
            library=library,
            skip_if_exists=False,  # Always rerun for fresh discovery
            task_section="DARK_MATTER_SUPERBATCH",
            allowed_ops=ude.BOOLEAN_OPS,
            population_size=4000,
            max_generations=150,  # More generations for harder problem
            mutation_rate=0.35,
            crossover_rate=0.5,
            use_macros=True,
            skip_quick_solve=False,
            allow_self=False,
            allowed_offsets=[0],
            allow_example_index_fractals=False,
            allow_unary_set_membership=False,  # No lookup tables - force generalization
            quick_accept_threshold=quick_accept,
            verbose=False,
        )
        
        if gene is None:
            print(f"  Bit {bit_idx}: ✗ FAILED - no pattern found")
            results[f"bit{bit_idx}"] = {"gene": None, "accuracy": 0.0}
            continue
        
        preds = gene.eval_all(task_inputs, n_points).astype(np.int64)
        accuracy = np.mean(preds == target)
        
        results[f"bit{bit_idx}"] = {
            "gene": gene.to_string(),
            "accuracy": float(accuracy),
        }
        
        status = "✓" if accuracy >= 0.95 else "~" if accuracy >= 0.80 else "?"
        gene_str = gene.to_string()
        print(f"  Bit {bit_idx}: {status} acc={accuracy:.4f} | {gene_str}")
    
    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), "sparc_discovered")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"SUPERBATCH_{data['n_galaxies']}gal_discovered.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Super-batch results → {out_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUPER-BATCH DISCOVERY SUMMARY")
    print("="*60)
    for bit_idx in [7, 6, 5, 4, 3, 2, 1, 0]:
        r = results.get(f"bit{bit_idx}", {})
        acc = r.get("accuracy", 0)
        gene = r.get("gene", "N/A")
        status = "✓" if acc >= 0.95 else "~" if acc >= 0.80 else "✗"
        print(f"  Bit {bit_idx}: {status} {acc:.1%} | {gene}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Super-batch cross-galaxy discovery")
    parser.add_argument("--test-only", action="store_true",
                       help="Only test known patterns, don't run discovery")
    parser.add_argument("--max-points", type=int, default=16,
                       help="Max points per galaxy (default: 16)")
    parser.add_argument("--all-galaxies", action="store_true",
                       help="Use all 175 galaxies, not just the 31 Add8_Cout ones")
    parser.add_argument("--quick-accept", type=float, default=0.95,
                       help="Quick accept threshold (default: 0.95)")
    args = parser.parse_args()
    
    print("="*60)
    print("SUPER-BATCH CROSS-GALAXY DISCOVERY")
    print("="*60)
    
    if args.all_galaxies:
        # Load all galaxies
        dat_files = glob.glob(os.path.join(SPARC_DIR, "*_rotmod.dat"))
        galaxy_names = [os.path.basename(f).replace("_rotmod.dat", "") for f in dat_files]
        print(f"Loading ALL {len(galaxy_names)} galaxies...")
    else:
        galaxy_names = COUT_GALAXIES
        print(f"Loading {len(COUT_GALAXIES)} galaxies that agreed on Add8_Cout...")
    
    print(f"Max points per galaxy: {args.max_points}")
    print()
    
    # Load and concatenate
    data = load_and_concatenate_galaxies(galaxy_names, max_points_per_galaxy=args.max_points)
    
    print(f"\nLoaded {data['n_points']} total points from {data['n_galaxies']} galaxies")
    print(f"Shape: vbar_bits={data['vbar_bits'].shape}, vobs_bits={data['vobs_bits'].shape}")
    
    if args.test_only:
        # Just test known patterns
        results = test_known_patterns(data)
    else:
        # Run full UDE discovery
        results = run_superbatch_discovery(data, quick_accept=args.quick_accept)


if __name__ == "__main__":
    main()
