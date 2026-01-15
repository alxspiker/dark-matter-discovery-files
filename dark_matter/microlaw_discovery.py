"""
MICRO-LAW DISCOVERY ENGINE
==========================
Hypothesis: The "noise" in lower bits (4-5) isn't random. 
It is fine-structure physics determined by the ratio of Gas vs Stars.

This script loads the 31 "Golden Set" galaxies but splits the input 
into 4 components:
1. V_gas (Gas Velocity contribution)
2. V_disk (Stellar Disk contribution)
3. V_bul (Bulge contribution)
4. Radius

Total Inputs: 32 bits (4 vars * 8 bits).
Target: V_obs (Observed Velocity).
"""

import sys
import os
import json
import glob
import numpy as np
import argparse
import re

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ude
from core.ast_core import Library

# Load library
LIBRARY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "discovered_ops.json")
library = Library(LIBRARY_PATH)

# The 31 "Golden Set" galaxies
COUT_GALAXIES = [
    "D512-2", "D564-8", "DDO064", "DDO168", "ESO116-G012",
    "ESO444-G084", "F571-V1", "NGC3109", "NGC3949", "NGC4085",
    "NGC4389", "NGC5005", "NGC5585", "UGC00891", "UGC02023",
    "UGC02259", "UGC04278", "UGC04483", "UGC05005", "UGC05414",
    "UGC05721", "UGC06923", "UGC07089", "UGC07261", "UGC07323",
    "UGC07866", "UGC08837", "UGC11820", "UGCA281", "UGCA442",
    "UGCA444"
]

SPARC_DIR = os.path.join(os.path.dirname(__file__), "sparc_galaxies")

def load_sparc_components(path):
    """Load raw component velocities."""
    data_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.strip().startswith("#"): continue
            data_lines.append(line.strip())

    radius, vobs, vgas, vdisk, vbul = [], [], [], [], []

    for line in data_lines:
        parts = re.split(r"\s+", line)
        # Standard SPARC format: Rad, Vobs, err, Vgas, Vdisk, Vbul
        # Note: Index might vary slightly if err is missing, but standard SPARC has it.
        # Assuming: Rad(0), Vobs(1), err(2), Vgas(3), Vdisk(4), Vbul(5)
        # We handle Vbul carefully as it might be missing in some files? 
        # (SPARC format is usually consistent, putting 0 if no bulge)
        
        try:
            r_val = float(parts[0])
            obs_val = float(parts[1])
            gas_val = float(parts[3])
            disk_val = float(parts[4])
            bul_val = float(parts[5]) if len(parts) > 5 else 0.0
            
            radius.append(r_val)
            vobs.append(obs_val)
            vgas.append(abs(gas_val))   # Use magnitude
            vdisk.append(abs(disk_val)) # Use magnitude
            vbul.append(abs(bul_val))   # Use magnitude
        except:
            continue

    return {
        "radius": np.array(radius),
        "v_obs": np.array(vobs),
        "v_gas": np.array(vgas),
        "v_disk": np.array(vdisk),
        "v_bul": np.array(vbul)
    }

def get_superbatch_data(max_points=16):
    """Load and discretize all components."""
    all_bits = {k: [] for k in ["gas", "dsk", "bul", "rad", "obs"]}
    
    # 1. Determine Global Scaling (Crucial for relative component weights)
    # We scan all galaxies first to find the global max velocity to normalize against
    # This ensures that if Gas is small compared to Disk, its integer value is smaller.
    
    # Actually, let's do per-galaxy scaling but consistent across components
    # max_v = max(max(v_gas), max(v_disk), max(v_bul), max(v_obs))
    
    print(f"Loading {len(COUT_GALAXIES)} galaxies...")
    
    for gname in COUT_GALAXIES:
        fpath = os.path.join(SPARC_DIR, f"{gname}_rotmod.dat")
        if not os.path.exists(fpath): continue
        
        raw = load_sparc_components(fpath)
        n = len(raw["radius"])
        
        # Subsample
        if n > max_points:
            idx = np.linspace(0, n-1, max_points, dtype=int)
        else:
            idx = np.arange(n)
            
        # Determine local max for normalization
        # We normalize everything to the max observed velocity or component in this galaxy
        # to keep 8-bit resolution relevant to the galaxy's scale.
        local_max = max(
            raw["v_obs"].max(), 
            raw["v_gas"].max(), 
            raw["v_disk"].max(), 
            raw["v_bul"].max()
        )
        r_max = raw["radius"].max()
        
        # Discretize to 8 bits (0-255)
        def to_bits(arr, mx):
            ints = np.clip((arr[idx] / (mx + 1e-9)) * 255, 0, 255).astype(np.int64)
            # Expand to (N, 8) bits
            return np.array([[(val >> b) & 1 for b in range(8)] for val in ints])

        all_bits["gas"].append(to_bits(raw["v_gas"], local_max))
        all_bits["dsk"].append(to_bits(raw["v_disk"], local_max))
        all_bits["bul"].append(to_bits(raw["v_bul"], local_max))
        all_bits["obs"].append(to_bits(raw["v_obs"], local_max))
        all_bits["rad"].append(to_bits(raw["radius"], r_max))

    # Stack
    return {k: np.vstack(v) for k, v in all_bits.items()}

def main():
    print("============================================================")
    print("🌌 MICRO-LAW DISCOVERY: GAS vs DISK vs BULGE")
    print("============================================================")
    
    data = get_superbatch_data()
    n_points = len(data["obs"])
    print(f"Loaded {n_points} total data points.")
    print("Inputs: Gas(8) + Disk(8) + Bulge(8) + Radius(8) = 32 bits")
    
    # Construct task inputs
    task_inputs = {}
    for b in range(8):
        task_inputs[f"g{b}"] = data["gas"][:, b] # Gas
        task_inputs[f"d{b}"] = data["dsk"][:, b] # Disk
        task_inputs[f"u{b}"] = data["bul"][:, b] # Bulge (u for bulge to avoid b collision)
        task_inputs[f"r{b}"] = data["rad"][:, b] # Radius
        
    task_vars = list(task_inputs.keys())
    
    # We focus on the "Noise Floor" bits from the previous run
    # Previous run: Bit 5 (~79%), Bit 4 (~74%)
    # If components matter, these should rise.
    
    for bit_idx in [6, 5, 4]:
        target = data["obs"][:, bit_idx]
        task_name = f"MICROLAW_b{bit_idx}"
        
        print(f"\n[BIT {bit_idx}] Hunting for Component Physics...")
        
        gene = ude.run_task(
            name=task_name,
            inputs=task_inputs,
            outputs=target,
            variables=task_vars,
            library=library,
            task_section="DARK_MATTER_MICROLAW",
            allowed_ops=ude.BOOLEAN_OPS,
            population_size=5000,     # Increased for larger search space
            max_generations=200,      # Increased for larger search space
            mutation_rate=0.4,
            crossover_rate=0.5,
            use_macros=True,
            allow_unary_set_membership=False, # No lookups! Pure logic only.
            quick_accept_threshold=0.90,
            skip_quick_solve=True,    # SKIP exhaustive search - 32 vars is too many!
            verbose=False
        )
        
        if gene:
            preds = gene.eval_all(task_inputs, n_points).astype(np.int64)
            acc = np.mean(preds == target)
            print(f"  RESULT: Bit {bit_idx} Accuracy: {acc:.4f}")
            print(f"  Formula: {gene.to_string()}")
            
            # Simple check: Does it use the components?
            g_str = gene.to_string()
            used_gas = "g" in g_str
            used_dsk = "d" in g_str
            used_bul = "u" in g_str
            print(f"  Components used: Gas={used_gas}, Disk={used_dsk}, Bulge={used_bul}")
        else:
            print("  No pattern found.")

if __name__ == "__main__":
    main()
