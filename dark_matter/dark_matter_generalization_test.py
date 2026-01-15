"""\
Dark Matter Generalization Test
==============================

Goal
----
Evaluate the *already discovered* dark-matter bit macros (DM_Vobs_b0..b7)
on data that is *not* the training sequence.

This is a truth test:
- Exact-fit membership lookups should NOT generalize.
- Compact structural behaviors (high bits) might.

Supports:
- Synthetic toy galaxies (m33, ngc2403_synth)
- SPARC Rotmod_LTG .dat files (tab-separated, comment headers)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.ast_core import Library
import ude


        "--scaling",
        choices=["per-galaxy", "m33"],
        default="per-galaxy",
        help="Discretization scaling policy",
    )

    ap.add_argument(
        "--mode",
        choices=["macros", "highbits", "max-latch", "hybrid"],
        default="macros",
        help=(
            "macros=full DM_Vobs_b* macros; "
            "highbits=top-2 bits from macros + low bits from vbar; "
            "max-latch=cumulative max of vbar; "
            "hybrid=top-2 bits from macros + low bits from max-latch"
        ),
    )

    ap.add_argument(
        "--library",
        type=str,
        default=os.path.join(REPO_ROOT, "discovered_ops.json"),
        help="Path to discovered_ops.json",
    )
    ap.add_argument(
        "--macro-prefix",
        type=str,
        default="DM_Vobs_b",
        help="Macro name prefix (default: DM_Vobs_b -> DM_Vobs_b0..DM_Vobs_b7)",
    )

    ap.add_argument("--col-radius", type=str, default="radius")
    ap.add_argument("--col-vbar", type=str, default="v_baryon")
    ap.add_argument("--col-vobs", type=str, default="v_obs")

    args = ap.parse_args()

    if args.csv:
        if args.csv.lower().endswith(".dat"):
            data = load_sparc_dat(args.csv)
        else:
            data = load_galaxy_csv(
                args.csv,
                col_radius=args.col_radius,
                col_vbar=args.col_vbar,
                col_vobs=args.col_vobs,
            )
        galaxy_name = os.path.basename(args.csv)
    else:
        data = create_synthetic_galaxy(args.galaxy, n_points=args.points, noise_std=args.noise)
        galaxy_name = args.galaxy

    bits = int(args.bits)

    # Choose scaling
    if args.scaling == "m33":
        train = create_synthetic_galaxy("m33", n_points=50, noise_std=0.0)
        scaling = compute_scaling(train)
        scaling_label = "m33"
    else:
        scaling = None
        scaling_label = "per-galaxy"

    disc = discretize(data, bits=bits, scaling=scaling)
    inputs_x, _vars_x = build_inputs_x(disc)

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
        # High bits from macros; low bits from instantaneous vbar.
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

    # Convert to velocity units using the *test* galaxy's v_obs scaling for readability.
    test_scale = compute_scaling(data)
    v_true = data["v_obs"]
    v_pred = inv_scale_u8_to_float(y_pred_u8, test_scale.vobs_min, test_scale.vobs_max, bits)
    rmse = float(np.sqrt(np.mean((v_true - v_pred) ** 2)))

    print(f"[RESULT] exact_u8_match={exact_match*100:.1f}% | RMSE(v_obs)={rmse:.2f} km/s")
    for b in range(bits - 1, -1, -1):
        print(f"  bit{b}: acc={per_bit[b]*100:.1f}%")

    rmse_newton = float(np.sqrt(np.mean((v_true - data["v_baryon"]) ** 2)))
    print(f"[BASELINE] RMSE(Newton v_baryon vs v_obs)={rmse_newton:.2f} km/s")


if __name__ == "__main__":
    main()


def create_synthetic_galaxy(
    name: str,
    n_points: int = 50,
    noise_std: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Return synthetic galaxy data (radius, v_baryon, v_obs).

    We provide a few parameterized toy galaxies. This is not astrophysics-grade,
    just a controlled regression test harness.
    """

    rng = np.random.default_rng(42)

    if name.lower() in {"m33", "m33_synth"}:
        r = np.linspace(0.1, 20.0, n_points)
        v_peak = 110.0
        r_scale = 5.0
        v_bar = v_peak * (r / r_scale) / np.power(1 + (r / r_scale) ** 2, 0.75)

        a0 = 3700.0
        g_bar = np.square(v_bar) / (r + 1e-10)
        g_obs = g_bar * (1 + np.sqrt(a0 / (g_bar + 1e-5)))
        v_obs = np.sqrt(g_obs * r)

    elif name.lower() in {"ngc2403", "ngc2403_synth", "ngc_2403_synth"}:
        # Roughly: rising curve to ~130-140, mild continued rise in outer disk.
        r = np.linspace(0.2, 15.0, n_points)
        v_peak = 95.0
        r_scale = 3.5
        v_bar = v_peak * (r / r_scale) / np.power(1 + (r / r_scale) ** 2, 0.78)

        # A different "MOND-like" interpolation + a mild outer boost
        a0 = 3300.0
        g_bar = np.square(v_bar) / (r + 1e-10)
        g_obs = g_bar * (1 + np.sqrt(a0 / (g_bar + 1e-5)))
        v_obs = np.sqrt(g_obs * r)
        v_obs = v_obs + 0.8 * np.log1p(r)  # gentle continued rise

    else:
        raise ValueError(f"Unknown synthetic galaxy '{name}'.")

    if noise_std and noise_std > 0:
        v_obs = v_obs + rng.normal(0.0, float(noise_std), size=v_obs.shape)

    return {
        "radius": r.astype(np.float64),
        "v_baryon": v_bar.astype(np.float64),
        "v_obs": v_obs.astype(np.float64),
    }


def load_galaxy_csv(
    path: str,
    *,
    col_radius: str = "radius",
    col_vbar: str = "v_baryon",
    col_vobs: str = "v_obs",
) -> Dict[str, np.ndarray]:
    """Load galaxy data from a CSV file."""

    def norm(s: str) -> str:
        return s.strip().lower()

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")

        fields = {norm(x): x for x in reader.fieldnames}
        r_key = fields.get(norm(col_radius))
        vb_key = fields.get(norm(col_vbar))
        vo_key = fields.get(norm(col_vobs))
        if not (r_key and vb_key and vo_key):
            raise ValueError(
                f"CSV missing required columns. Found: {reader.fieldnames}. "
                f"Need: {col_radius}, {col_vbar}, {col_vobs} (case-insensitive)."
            )

        radius: List[float] = []
        vbar: List[float] = []
        vobs: List[float] = []

        for row in reader:
            radius.append(float(row[r_key]))
            vbar.append(float(row[vb_key]))
            vobs.append(float(row[vo_key]))

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


def discretize(
    data: Dict[str, np.ndarray],
    *,
    bits: int = 8,
    scaling: Optional[Scaling] = None,
) -> Dict[str, np.ndarray]:
    """Discretize radius/v_baryon/v_obs to integer bitvectors.

    If scaling is provided, use it (train-scaling). Otherwise per-galaxy scaling.
    """

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


def build_inputs_x(discrete: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Build x0..x15 inputs: vbar bits then r bits (matches exact-fit discovery)."""

    bits = int(discrete["bits"])
    vbar_bits = bit_planes(discrete["vbar"], bits)
    r_bits = bit_planes(discrete["r"], bits)

    task_inputs: Dict[str, np.ndarray] = {}
    task_vars: List[str] = []

    idx = 0
    for b in range(bits):
        nm = f"x{idx}"
        task_inputs[nm] = vbar_bits[f"b{b}"]
        task_vars.append(nm)
        idx += 1
    for b in range(bits):
        nm = f"x{idx}"
        task_inputs[nm] = r_bits[f"b{b}"]
        task_vars.append(nm)
        idx += 1

    return task_inputs, task_vars


def eval_macro_bits(
    lib: Library,
    *,
    macro_prefix: str,
    bits: int,
    inputs: Dict[str, np.ndarray],
    n: int,
) -> Tuple[np.ndarray, Dict[int, float]]:
    """Evaluate DM_Vobs_b{bit} macros on provided inputs; return packed u8 predictions."""

    pred_bits: List[np.ndarray] = []
    acc_by_bit: Dict[int, float] = {}

    for b in range(bits):
        name = f"{macro_prefix}{b}"
        gene = ude.load_fractal_gene(name, lib)
        if gene is None:
            raise RuntimeError(f"Missing macro '{name}' in library")

        preds = gene.eval_all(inputs, n).astype(np.int64)
        pred_bits.append(preds)

    packed = np.zeros(n, dtype=np.int64)
    for b, pb in enumerate(pred_bits):
        packed |= (pb & 1) << b

    return packed, acc_by_bit


def bit_accuracy(y_true_u8: np.ndarray, y_pred_u8: np.ndarray, bits: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for b in range(bits):
        t = ((y_true_u8 >> b) & 1).astype(np.int64)
        p = ((y_pred_u8 >> b) & 1).astype(np.int64)
        out[b] = float(np.mean(t == p))
    return out


def inv_scale_u8_to_float(y_u8: np.ndarray, lo: float, hi: float, bits: int) -> np.ndarray:
    max_val = (1 << bits) - 1
    return lo + (y_u8.astype(np.float64) / float(max_val)) * (hi - lo)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generalization test for discovered dark-matter bit macros")
    ap.add_argument("--galaxy", type=str, default="m33", help="Synthetic galaxy name (m33, ngc2403_synth)")
    ap.add_argument("--csv", type=str, default=None, help="Load galaxy data from CSV instead of synthetic")
    ap.add_argument("--noise", type=float, default=0.0, help="Stddev noise on v_obs for synthetic")
    ap.add_argument("--points", type=int, default=50, help="Number of points for synthetic")

    ap.add_argument("--bits", type=int, default=8, help="Bit width for discretization")
    ap.add_argument(
        "--scaling",
        choices=["per-galaxy", "m33"],
        default="per-galaxy",
        help="Discretization scaling policy",
    )

    ap.add_argument(
        "--library",
        type=str,
        default=os.path.join(REPO_ROOT, "discovered_ops.json"),
        help="Path to discovered_ops.json",
    )
    ap.add_argument(
        "--macro-prefix",
        type=str,
        default="DM_Vobs_b",
        help="Macro name prefix (default: DM_Vobs_b -> DM_Vobs_b0..DM_Vobs_b7)",
    )

    ap.add_argument("--col-radius", type=str, default="radius")
    ap.add_argument("--col-vbar", type=str, default="v_baryon")
    ap.add_argument("--col-vobs", type=str, default="v_obs")

    args = ap.parse_args()

    if args.csv:
        if args.csv.lower().endswith(".dat"):
            data = load_sparc_dat(args.csv)
        else:
            data = load_galaxy_csv(args.csv, col_radius=args.col_radius, col_vbar=args.col_vbar, col_vobs=args.col_vobs)
        galaxy_name = os.path.basename(args.csv)
    else:
        data = create_synthetic_galaxy(args.galaxy, n_points=args.points, noise_std=args.noise)
        galaxy_name = args.galaxy

    bits = int(args.bits)

    # Choose scaling
    if args.scaling == "m33":
        train = create_synthetic_galaxy("m33", n_points=50, noise_std=0.0)
        scaling = compute_scaling(train)
        scaling_label = "m33"
    else:
        scaling = None
        scaling_label = "per-galaxy"

    disc = discretize(data, bits=bits, scaling=scaling)
    inputs_x, vars_x = build_inputs_x(disc)

    y_true_u8 = disc["vobs"].astype(np.int64)
    n = int(len(y_true_u8))

    lib = Library(args.library)
    print(f"[LIB] Loaded {len(lib.macros)} macros from {args.library}")
    print(f"[GALAXY] {galaxy_name} | n={n} | bits={bits} | scaling={scaling_label}")

    # Evaluate discovered macros
    y_pred_u8, _ = eval_macro_bits(lib, macro_prefix=args.macro_prefix, bits=bits, inputs=inputs_x, n=n)

    # Metrics
    per_bit = bit_accuracy(y_true_u8, y_pred_u8, bits)
    exact_match = float(np.mean(y_true_u8 == y_pred_u8))

    # Convert to velocity units using the *test* galaxy's v_obs scaling for readability
    # (we still discretize by scaling policy above; this is just for printing RMSE)
    test_scale = compute_scaling(data)
    v_true = data["v_obs"]
    v_pred = inv_scale_u8_to_float(y_pred_u8, test_scale.vobs_min, test_scale.vobs_max, bits)

    rmse = float(np.sqrt(np.mean((v_true - v_pred) ** 2)))

    print(f"[RESULT] exact_u8_match={exact_match*100:.1f}% | RMSE(v_obs)={rmse:.2f} km/s")
    for b in range(bits - 1, -1, -1):
        print(f"  bit{b}: acc={per_bit[b]*100:.1f}%")

    # Baselines
    vbar = data["v_baryon"]
    rmse_newton = float(np.sqrt(np.mean((v_true - vbar) ** 2)))
    print(f"[BASELINE] RMSE(Newton v_baryon vs v_obs)={rmse_newton:.2f} km/s")


if __name__ == "__main__":
    main()
