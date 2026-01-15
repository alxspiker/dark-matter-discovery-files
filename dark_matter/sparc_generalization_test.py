"""\
SPARC Generalization Test
========================

Evaluates previously discovered dark-matter bit-macros (e.g. DM_Vobs_b0..b7)
on galaxy rotation curve data that is NOT the original training sequence.

Supports:
- SPARC Rotmod_LTG .dat files (tab-separated, comment headers)
- Synthetic toy galaxies for quick smoke tests

Key idea:
- Exact-fit membership macros should not generalize.
- High-bit structural patterns may still show signal.

Examples
--------
# Test on SPARC NGC 2403:
python dark_matter/sparc_generalization_test.py --dat dark_matter/sparc_galaxies/NGC2403_rotmod.dat --scaling m33

# Compare to simple mechanism baselines:
python dark_matter/sparc_generalization_test.py --dat dark_matter/sparc_galaxies/NGC2403_rotmod.dat --scaling m33 --mode hybrid
python dark_matter/sparc_generalization_test.py --dat dark_matter/sparc_galaxies/NGC2403_rotmod.dat --scaling m33 --mode max-latch
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.ast_core import Library
import ude


@dataclass(frozen=True)
class Scaling:
    r_min: float
    r_max: float
    vbar_min: float
    vbar_max: float
    vobs_min: float
    vobs_max: float


def create_synthetic_galaxy(name: str, n_points: int = 50) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    name_l = name.lower()

    if name_l in {"m33", "m33_synth"}:
        r = np.linspace(0.1, 20.0, n_points)
        v_peak = 110.0
        r_scale = 5.0
        v_bar = v_peak * (r / r_scale) / np.power(1 + (r / r_scale) ** 2, 0.75)

        a0 = 3700.0
        g_bar = np.square(v_bar) / (r + 1e-10)
        g_obs = g_bar * (1 + np.sqrt(a0 / (g_bar + 1e-5)))
        v_obs = np.sqrt(g_obs * r)

    elif name_l in {"ngc2403", "ngc2403_synth", "ngc_2403_synth"}:
        r = np.linspace(0.2, 15.0, n_points)
        v_peak = 95.0
        r_scale = 3.5
        v_bar = v_peak * (r / r_scale) / np.power(1 + (r / r_scale) ** 2, 0.78)

        a0 = 3300.0
        g_bar = np.square(v_bar) / (r + 1e-10)
        g_obs = g_bar * (1 + np.sqrt(a0 / (g_bar + 1e-5)))
        v_obs = np.sqrt(g_obs * r)
        v_obs = v_obs + 0.8 * np.log1p(r)

    else:
        raise ValueError(f"Unknown synthetic galaxy '{name}'")

    # Tiny noise to avoid degenerate scaling edge cases
    v_obs = v_obs + rng.normal(0.0, 1e-9, size=v_obs.shape)

    return {
        "radius": r.astype(np.float64),
        "v_baryon": v_bar.astype(np.float64),
        "v_obs": v_obs.astype(np.float64),
    }


def load_sparc_dat(
    path: str,
    *,
    col_radius: str = "Rad",
    col_vobs: str = "Vobs",
    col_vgas: str = "Vgas",
    col_vdisk: str = "Vdisk",
    col_vbul: str = "Vbul",
) -> Dict[str, np.ndarray]:
    header_cols: Optional[List[str]] = None
    data_lines: List[str] = []

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
        raise ValueError(f"No data rows found in {path}")
    if header_cols is None:
        header_cols = ["Rad", "Vobs", "errV", "Vgas", "Vdisk", "Vbul"]

    idx = {name: i for i, name in enumerate(header_cols)}

    def col(name: str) -> int:
        if name not in idx:
            raise ValueError(f"Column '{name}' not found in {path} header: {header_cols}")
        return idx[name]

    radius: List[float] = []
    vobs: List[float] = []
    vgas: List[float] = []
    vdisk: List[float] = []
    vbul: List[float] = []

    for line in data_lines:
        parts = re.split(r"\s+", line)
        radius.append(float(parts[col(col_radius)]))
        vobs.append(float(parts[col(col_vobs)]))
        vgas.append(float(parts[col(col_vgas)]))
        vdisk.append(float(parts[col(col_vdisk)]))
        vbul.append(float(parts[col(col_vbul)]))

    vbar = np.sqrt(np.square(vgas) + np.square(vdisk) + np.square(vbul))
    return {
        "radius": np.asarray(radius, dtype=np.float64),
        "v_baryon": np.asarray(vbar, dtype=np.float64),
        "v_obs": np.asarray(vobs, dtype=np.float64),
    }


def compute_scaling(data: Dict[str, np.ndarray]) -> Scaling:
    return Scaling(
        r_min=float(np.min(data["radius"])),
        r_max=float(np.max(data["radius"])),
        vbar_min=float(np.min(data["v_baryon"])),
        vbar_max=float(np.max(data["v_baryon"])),
        vobs_min=float(np.min(data["v_obs"])),
        vobs_max=float(np.max(data["v_obs"])),
    )


def _normalize_to_u8(x: np.ndarray, lo: float, hi: float, bits: int) -> np.ndarray:
    if hi <= lo:
        return np.zeros_like(x, dtype=np.int64)
    max_val = (1 << bits) - 1
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    # Match `dark_matter_discovery.discretize_for_ude`: multiply then truncate.
    return (y * max_val).astype(np.int64)


def discretize(data: Dict[str, np.ndarray], *, bits: int, scaling: Optional[Scaling]) -> Dict[str, np.ndarray]:
    s = scaling or compute_scaling(data)
    return {
        "r": _normalize_to_u8(data["radius"], s.r_min, s.r_max, bits),
        "vbar": _normalize_to_u8(data["v_baryon"], s.vbar_min, s.vbar_max, bits),
        "vobs": _normalize_to_u8(data["v_obs"], s.vobs_min, s.vobs_max, bits),
        "bits": np.int64(bits),
        "scaling": s,
    }


def bit_planes(arr: np.ndarray, bits: int) -> Dict[str, np.ndarray]:
    return {f"b{b}": ((arr >> b) & 1).astype(np.int64) for b in range(bits)}


def build_inputs_x(discrete: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    bits = int(discrete["bits"])
    vbar_bits = bit_planes(discrete["vbar"], bits)
    r_bits = bit_planes(discrete["r"], bits)

    task_inputs: Dict[str, np.ndarray] = {}
    idx = 0
    for b in range(bits):
        task_inputs[f"x{idx}"] = vbar_bits[f"b{b}"]
        idx += 1
    for b in range(bits):
        task_inputs[f"x{idx}"] = r_bits[f"b{b}"]
        idx += 1
    return task_inputs


def eval_macro_bits(
    lib: Library,
    *,
    macro_prefix: str,
    bits: int,
    inputs: Dict[str, np.ndarray],
    n: int,
) -> np.ndarray:
    pred_bits: List[np.ndarray] = []
    for b in range(bits):
        name = f"{macro_prefix}{b}"
        gene = ude.load_fractal_gene(name, lib)
        if gene is None:
            raise RuntimeError(f"Missing macro '{name}' in library")
        pred_bits.append(gene.eval_all(inputs, n).astype(np.int64))

    packed = np.zeros(n, dtype=np.int64)
    for b, pb in enumerate(pred_bits):
        packed |= (pb & 1) << b
    return packed


def bit_accuracy(y_true_u8: np.ndarray, y_pred_u8: np.ndarray, bits: int) -> Dict[int, float]:
    return {b: float(np.mean(((y_true_u8 >> b) & 1) == ((y_pred_u8 >> b) & 1))) for b in range(bits)}


def inv_scale_u8_to_float(y_u8: np.ndarray, lo: float, hi: float, bits: int) -> np.ndarray:
    max_val = (1 << bits) - 1
    return lo + (y_u8.astype(np.float64) / float(max_val)) * (hi - lo)


def cumulative_max_latch_u8(vbar_u8: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(vbar_u8.astype(np.int64))


def hybrid_highbits_from_macros_lowbits_from_latch(
    *,
    macros_u8: np.ndarray,
    latch_u8: np.ndarray,
    bits: int,
    high_bits: int = 2,
) -> np.ndarray:
    if high_bits <= 0:
        return latch_u8.astype(np.int64)
    if high_bits >= bits:
        return macros_u8.astype(np.int64)
    high_mask = ((1 << high_bits) - 1) << (bits - high_bits)
    low_mask = (1 << (bits - high_bits)) - 1
    return (macros_u8 & high_mask) | (latch_u8 & low_mask)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dat", type=str, default=None, help="SPARC Rotmod_LTG .dat path")
    ap.add_argument("--galaxy", type=str, default="m33", help="Synthetic galaxy name (m33, ngc2403_synth)")
    ap.add_argument("--batch", action="store_true", help="Run all modes on all galaxies in sparc_galaxies/")
    ap.add_argument("--out-csv", type=str, default="sparc_results.csv", help="Output CSV for batch mode")

    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--scaling", choices=["per-galaxy", "m33"], default="per-galaxy")
    ap.add_argument(
        "--mode",
        choices=["macros", "highbits", "max-latch", "hybrid"],
        default="macros",
    )

    ap.add_argument(
        "--library",
        type=str,
        default=os.path.join(REPO_ROOT, "discovered_ops.json"),
    )
    ap.add_argument("--macro-prefix", type=str, default="DM_Vobs_b")

    args = ap.parse_args()

    if args.batch:
        run_batch(args)
        return

    if args.dat:
        data = load_sparc_dat(args.dat)
        galaxy_name = os.path.basename(args.dat)
    else:
        data = create_synthetic_galaxy(args.galaxy)
        galaxy_name = args.galaxy

    bits = int(args.bits)

    if args.scaling == "m33":
        scaling = compute_scaling(create_synthetic_galaxy("m33"))
        scaling_label = "m33"
    else:
        scaling = None
        scaling_label = "per-galaxy"

    disc = discretize(data, bits=bits, scaling=scaling)
    inputs_x = build_inputs_x(disc)

    y_true_u8 = disc["vobs"].astype(np.int64)
    vbar_u8 = disc["vbar"].astype(np.int64)
    n = int(len(y_true_u8))

    lib = Library(args.library)
    print(f"[LIB] Loaded {len(lib.macros)} macros from {args.library}")
    print(f"[GALAXY] {galaxy_name} | n={n} | bits={bits} | scaling={scaling_label} | mode={args.mode}")

    y_macros_u8 = eval_macro_bits(lib, macro_prefix=args.macro_prefix, bits=bits, inputs=inputs_x, n=n)

    if args.mode == "macros":
        y_pred_u8 = y_macros_u8
    elif args.mode == "highbits":
        y_pred_u8 = (y_macros_u8 & 0xC0) | (vbar_u8 & 0x3F)
    elif args.mode == "max-latch":
        y_pred_u8 = cumulative_max_latch_u8(vbar_u8)
    elif args.mode == "hybrid":
        y_pred_u8 = hybrid_highbits_from_macros_lowbits_from_latch(
            macros_u8=y_macros_u8,
            latch_u8=cumulative_max_latch_u8(vbar_u8),
            bits=bits,
            high_bits=2,
        )
    else:
        raise RuntimeError(f"Unknown mode: {args.mode}")

    per_bit = bit_accuracy(y_true_u8, y_pred_u8, bits)
    exact_match = float(np.mean(y_true_u8 == y_pred_u8))

    test_scale = compute_scaling(data)
    v_true = data["v_obs"]
    v_pred = inv_scale_u8_to_float(y_pred_u8, test_scale.vobs_min, test_scale.vobs_max, bits)
    rmse = float(np.sqrt(np.mean((v_true - v_pred) ** 2)))

    print(f"[RESULT] exact_u8_match={exact_match*100:.1f}% | RMSE(v_obs)={rmse:.2f} km/s")
    for b in range(bits - 1, -1, -1):
        print(f"  bit{b}: acc={per_bit[b]*100:.1f}%")

    rmse_newton = float(np.sqrt(np.mean((v_true - data["v_baryon"]) ** 2)))
    print(f"[BASELINE] RMSE(Newton v_baryon vs v_obs)={rmse_newton:.2f} km/s")


def run_batch(args) -> None:
    """Run all modes on all galaxies in sparc_galaxies/, output CSV."""
    import csv as csv_mod
    import glob

    sparc_dir = os.path.join(os.path.dirname(__file__), "sparc_galaxies")
    dat_files = sorted(glob.glob(os.path.join(sparc_dir, "*.dat")))

    if not dat_files:
        print(f"[ERROR] No .dat files found in {sparc_dir}")
        return

    bits = int(args.bits)
    modes = ["macros", "highbits", "max-latch", "hybrid"]

    if args.scaling == "m33":
        scaling = compute_scaling(create_synthetic_galaxy("m33"))
        scaling_label = "m33"
    else:
        scaling = None
        scaling_label = "per-galaxy"

    lib = Library(args.library)
    print(f"[LIB] Loaded {len(lib.macros)} macros")
    print(f"[BATCH] {len(dat_files)} galaxies x {len(modes)} modes = {len(dat_files)*len(modes)} evaluations")

    results = []

    for dat_path in dat_files:
        galaxy_name = os.path.basename(dat_path).replace("_rotmod.dat", "")
        try:
            data = load_sparc_dat(dat_path)
        except Exception as e:
            print(f"[SKIP] {galaxy_name}: {e}")
            continue

        disc = discretize(data, bits=bits, scaling=scaling)
        inputs_x = build_inputs_x(disc)

        y_true_u8 = disc["vobs"].astype(np.int64)
        vbar_u8 = disc["vbar"].astype(np.int64)
        n = int(len(y_true_u8))

        test_scale = compute_scaling(data)
        v_true = data["v_obs"]
        rmse_newton = float(np.sqrt(np.mean((v_true - data["v_baryon"]) ** 2)))

        y_macros_u8 = eval_macro_bits(lib, macro_prefix=args.macro_prefix, bits=bits, inputs=inputs_x, n=n)

        for mode in modes:
            if mode == "macros":
                y_pred_u8 = y_macros_u8
            elif mode == "highbits":
                y_pred_u8 = (y_macros_u8 & 0xC0) | (vbar_u8 & 0x3F)
            elif mode == "max-latch":
                y_pred_u8 = cumulative_max_latch_u8(vbar_u8)
            elif mode == "hybrid":
                y_pred_u8 = hybrid_highbits_from_macros_lowbits_from_latch(
                    macros_u8=y_macros_u8,
                    latch_u8=cumulative_max_latch_u8(vbar_u8),
                    bits=bits,
                    high_bits=2,
                )
            else:
                continue

            per_bit = bit_accuracy(y_true_u8, y_pred_u8, bits)
            exact_match = float(np.mean(y_true_u8 == y_pred_u8))
            v_pred = inv_scale_u8_to_float(y_pred_u8, test_scale.vobs_min, test_scale.vobs_max, bits)
            rmse = float(np.sqrt(np.mean((v_true - v_pred) ** 2)))

            row = {
                "galaxy": galaxy_name,
                "n_points": n,
                "mode": mode,
                "scaling": scaling_label,
                "exact_match_pct": round(exact_match * 100, 2),
                "rmse_km_s": round(rmse, 2),
                "rmse_newton_km_s": round(rmse_newton, 2),
            }
            for b in range(bits):
                row[f"bit{b}_acc"] = round(per_bit[b] * 100, 2)
            results.append(row)

        print(f"  {galaxy_name}: done ({n} pts)")

    out_path = os.path.join(os.path.dirname(__file__), args.out_csv)
    fieldnames = ["galaxy", "n_points", "mode", "scaling", "exact_match_pct", "rmse_km_s", "rmse_newton_km_s"] + [f"bit{b}_acc" for b in range(bits)]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"[DONE] Wrote {len(results)} rows to {out_path}")



if __name__ == "__main__":
    main()
