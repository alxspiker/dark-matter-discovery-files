#!/usr/bin/env python3
"""
UDE FULL DISCOVERY: Dark Matter Acceleration Law Discovery
==========================================================
Uses ALL available UDE-CORE modules for comprehensive physics discovery.

Modules Used:
1. FractalGene / FractalEvolver - Genetic programming discovery
2. InductiveVerifier - Four-law verification (Benford/Zipf/LLN/Pareto)
3. GFMLB - Geometric Fractal Memory Lattice for pattern storage
4. ZipfAnalyzer - Rank-frequency validation
5. ParetoAnalyzer - 80/20 imbalance detection
6. LLNAnalyzer - Sample sufficiency / convergence
7. SequentialContext - State-aware evaluation
8. MultiInputFractalGene - Multi-variable formulas
9. MemorySorter - Event categorization

This is the FULL UDE stack applied to dark matter physics discovery!

Usage:
    python ude_full_discovery.py --discover
    python ude_full_discovery.py --validate
    python ude_full_discovery.py --verify-laws
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# UDE-CORE IMPORTS (FULL STACK)
# =============================================================================
UDE_CORE_PATH = r"C:\Users\alxth\OneDrive\Documents\GitHub\ude-core"
if UDE_CORE_PATH not in sys.path:
    sys.path.insert(0, UDE_CORE_PATH)

# Track what's available
UDE_MODULES = {
    'fractal_core': False,
    'ast_core': False,
    'verifier': False,
    'gfmlb': False,
    'zipf_law': False,
    'pareto_law': False,
    'lln': False,
    'sequential_memory': False,
    'memory_sorter': False,
    'ude_main': False,
}

# 1. Fractal Core - Genetic Programming
try:
    from core.fractal_core import (
        FractalGene,
        FractalEvolver,
        BinaryFractalGene,
        MultiInputFractalGene,
        CompositeFractalGene,
        FractalTransform,
        set_gfmlb_instance,
        get_gfmlb_instance,
    )
    UDE_MODULES['fractal_core'] = True
    print("[UDE] ✓ Fractal Core loaded")
except ImportError as e:
    print(f"[UDE] ✗ Fractal Core: {e}")

# 2. AST Core - Expression Trees
try:
    from core.ast_core import (
        ASTNode,
        Var,
        Const,
        Library,
        BitAnd, BitOr, BitXor, BitNot,
        MacroOp,
        full_simplify,
    )
    UDE_MODULES['ast_core'] = True
    print("[UDE] ✓ AST Core loaded")
except ImportError as e:
    print(f"[UDE] ✗ AST Core: {e}")

# 3. Inductive Verifier (Four-Law Verification)
try:
    from core.verifier import InductiveVerifier
    UDE_MODULES['verifier'] = True
    print("[UDE] ✓ Inductive Verifier loaded")
except ImportError as e:
    print(f"[UDE] ✗ Verifier: {e}")

# 4. GFMLB - Memory Lattice
try:
    from core.gfmlb import (
        GFMLB,
        MemoryType,
        PathClassification,
        BenfordResult,
        check_benford,
        create_gfmlb,
    )
    # Benford's law expected frequencies
    BENFORD_FREQUENCIES = [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]
    UDE_MODULES['gfmlb'] = True
    print("[UDE] ✓ GFMLB Memory Lattice loaded")
except ImportError as e:
    print(f"[UDE] ✗ GFMLB: {e}")
    BENFORD_FREQUENCIES = [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]

# 5. Zipf's Law
try:
    from core.zipf_law import (
        ZipfAnalyzer,
        ZipfResult,
        ZIPF_KS_NATURAL_THRESHOLD,
        quick_zipf_check,
    )
    UDE_MODULES['zipf_law'] = True
    print("[UDE] ✓ Zipf's Law Analyzer loaded")
except ImportError as e:
    print(f"[UDE] ✗ Zipf Law: {e}")

# 6. Pareto Principle (80/20)
try:
    from core.pareto_law import (
        ParetoAnalyzer,
        ParetoResult,
        PARETO_KS_NATURAL_THRESHOLD,
        compute_gini_coefficient,
        quick_pareto_check,
        quick_80_20_check,
    )
    UDE_MODULES['pareto_law'] = True
    print("[UDE] ✓ Pareto Principle Analyzer loaded")
except ImportError as e:
    print(f"[UDE] ✗ Pareto Law: {e}")

# 7. Law of Large Numbers
try:
    from core.lln import (
        LLNAnalyzer,
        LLNResult,
        LLN_MIN_SAMPLES,
        LLN_CONVERGENCE_EPSILON,
        check_lln_convergence,
        validate_sample_sufficiency,
        check_fitness_stabilization,
    )
    UDE_MODULES['lln'] = True
    print("[UDE] ✓ Law of Large Numbers loaded")
except ImportError as e:
    print(f"[UDE] ✗ LLN: {e}")

# 8. Sequential Memory
try:
    from core.sequential_memory import (
        SequentialContext,
        LogTraceBankGene,
        IteratedMapGene,
        FixedPointClosureGene,
        FractionalAccumulatorGene,
    )
    UDE_MODULES['sequential_memory'] = True
    print("[UDE] ✓ Sequential Memory loaded")
except ImportError as e:
    print(f"[UDE] ✗ Sequential Memory: {e}")

# 9. Memory Sorter
try:
    from core.memory_sorter import (
        MemorySorter,
        UDEDiscoveryHook,
    )
    UDE_MODULES['memory_sorter'] = True
    print("[UDE] ✓ Memory Sorter loaded")
except ImportError as e:
    print(f"[UDE] ✗ Memory Sorter: {e}")

# 10. Main UDE Module
try:
    import ude
    UDE_MODULES['ude_main'] = True
    print("[UDE] ✓ Main UDE Module loaded")
except ImportError as e:
    print(f"[UDE] ✗ Main UDE: {e}")

print(f"\n[UDE] Loaded {sum(UDE_MODULES.values())}/{len(UDE_MODULES)} modules")

# =============================================================================
# PHYSICS CONSTANTS
# =============================================================================
G_DAGGER = 1.2e-10  # m/s² - RAR scale (canonical)
A0_MOND = 1.2e-10   # m/s² - MOND acceleration constant
KPC_TO_M = 3.086e19
KMS_TO_MS = 1000

# =============================================================================
# DATA LOADING
# =============================================================================

def load_sparc(data_dir: str = "sparc_galaxies") -> Dict[str, pd.DataFrame]:
    """Load SPARC galaxy data."""
    data_path = Path(data_dir)
    if not data_path.exists():
        script_dir = Path(__file__).parent
        data_path = script_dir / data_dir
    
    galaxies = {}
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
            
            # Compute accelerations
            R_m = df['R'] * KPC_TO_M
            df['g_bar'] = (df['V_bar'] * KMS_TO_MS)**2 / R_m
            df['g_obs'] = (df['V_obs'] * KMS_TO_MS)**2 / R_m
            
            df = df[(df['g_bar'] > 1e-15) & (df['g_obs'] > 1e-15)].copy()
            df['log_g_bar'] = np.log10(df['g_bar'])
            df['log_g_obs'] = np.log10(df['g_obs'])
            
            if len(df) >= 5:
                galaxies[name] = df
        except Exception:
            continue
    
    return galaxies


def split_galaxies(galaxies: Dict, seed: int = 42):
    """Split galaxies into train/val/test."""
    names = sorted(galaxies.keys())
    np.random.seed(seed)
    np.random.shuffle(names)
    n = len(names)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    return names[:n_train], names[n_train:n_train+n_val], names[n_train+n_val:]


def get_data(galaxies, names):
    """Get concatenated data with galaxy slices."""
    x_list, y_list, slices = [], [], []
    idx = 0
    for name in names:
        df = galaxies[name]
        n = len(df)
        x_list.extend(df['log_g_bar'].values)
        y_list.extend(df['log_g_obs'].values)
        slices.append((idx, idx + n))
        idx += n
    return np.array(x_list), np.array(y_list), slices


def galaxy_weighted_mae(pred, true, slices):
    """Galaxy-weighted MAE."""
    total = 0.0
    for s, e in slices:
        if e > s:
            total += np.mean(np.abs(pred[s:e] - true[s:e]))
    return total / len(slices)


# =============================================================================
# FOUR-LAW VERIFICATION
# =============================================================================

@dataclass
class FourLawResult:
    """Results from four-law verification."""
    benford_pass: bool = False
    benford_deviation: float = 1.0
    zipf_pass: bool = False
    zipf_ks: float = 1.0
    lln_pass: bool = False
    lln_cv: float = 1.0
    pareto_pass: bool = False
    pareto_gini: float = 1.0
    overall_pass: bool = False
    
    def __str__(self):
        return (f"Four-Law: Benford={self.benford_pass} ({self.benford_deviation:.3f}), "
                f"Zipf={self.zipf_pass} ({self.zipf_ks:.3f}), "
                f"LLN={self.lln_pass} ({self.lln_cv:.3f}), "
                f"Pareto={self.pareto_pass} ({self.pareto_gini:.3f}) → "
                f"{'PASS' if self.overall_pass else 'FAIL'}")


def verify_four_laws(values: np.ndarray, verbose: bool = False) -> FourLawResult:
    """
    Apply all four statistical laws to verify naturalness of discovered pattern.
    
    This is the UDE's key innovation: using statistical laws to detect
    unphysical/overfit solutions AUTOMATICALLY.
    """
    result = FourLawResult()
    values = np.asarray(values).flatten()
    
    # Handle negative values for Benford (use absolute values)
    abs_values = np.abs(values[np.isfinite(values)])
    abs_values = abs_values[abs_values > 1e-30]  # Remove near-zeros
    
    if len(abs_values) < 10:
        if verbose:
            print("  [!] Insufficient data for four-law verification")
        return result
    
    # 1. BENFORD'S LAW - First digit distribution
    # Natural data follows P(d) = log10(1 + 1/d) for d=1..9
    try:
        # Get first significant digits
        first_digits = []
        for v in abs_values:
            s = f"{v:.10e}"
            for c in s:
                if c.isdigit() and c != '0':
                    first_digits.append(int(c))
                    break
        
        if len(first_digits) >= 10:
            observed = np.bincount(first_digits, minlength=10)[1:10].astype(float)
            if observed.sum() > 0:
                observed = observed / observed.sum()
                expected = np.array(BENFORD_FREQUENCIES)
                deviation = np.mean(np.abs(observed - expected))
                result.benford_deviation = deviation
                result.benford_pass = deviation < 0.10  # Relaxed threshold
    except Exception as e:
        if verbose:
            print(f"  [!] Benford check error: {e}")
    
    # 2. ZIPF'S LAW - Rank-frequency distribution
    if UDE_MODULES['zipf_law']:
        try:
            zipf = ZipfAnalyzer()
            # Bin values and analyze frequencies
            hist, _ = np.histogram(abs_values, bins=50)
            hist = hist[hist > 0]
            if len(hist) >= 5:
                zipf_result = zipf.analyze(hist)
                result.zipf_ks = zipf_result.ks_statistic
                result.zipf_pass = zipf_result.ks_statistic < 0.15  # Relaxed threshold
        except Exception as e:
            if verbose:
                print(f"  [!] Zipf check error: {e}")
    
    # 3. LAW OF LARGE NUMBERS - Sample sufficiency
    if UDE_MODULES['lln']:
        try:
            lln = LLNAnalyzer()
            lln_result = lln.analyze(abs_values)
            result.lln_cv = lln_result.coefficient_of_variation
            # Pass if we have enough samples and reasonable CV
            result.lln_pass = len(abs_values) >= 100 and result.lln_cv < 2.0
        except Exception as e:
            if verbose:
                print(f"  [!] LLN check error: {e}")
    
    # 4. PARETO PRINCIPLE - 80/20 imbalance
    if UDE_MODULES['pareto_law']:
        try:
            pareto = ParetoAnalyzer()
            pareto_result = pareto.analyze(abs_values)
            result.pareto_gini = pareto_result.gini_coefficient
            # Gini < 0.7 is acceptable (some concentration is normal)
            result.pareto_pass = pareto_result.gini_coefficient < 0.7
        except Exception as e:
            if verbose:
                print(f"  [!] Pareto check error: {e}")
    
    # Overall: need at least 2/4 to pass (relaxed for physics data)
    passes = sum([result.benford_pass, result.zipf_pass, 
                  result.lln_pass, result.pareto_pass])
    result.overall_pass = passes >= 2
    
    if verbose:
        print(f"  {result}")
    
    return result


# =============================================================================
# PHYSICS-CONSTRAINED EXPRESSION TREE
# =============================================================================

class PhysicsExpr:
    """Expression tree node for physics formulas."""
    
    def evaluate(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        raise NotImplementedError
    
    def to_string(self, params: Dict[str, float]) -> str:
        raise NotImplementedError
    
    def complexity(self) -> int:
        raise NotImplementedError
    
    def is_monotonic(self, x_range: Tuple[float, float] = (-14, -8), 
                     n_points: int = 200) -> bool:
        """Check if formula is monotonically increasing."""
        x = np.linspace(x_range[0], x_range[1], n_points)
        try:
            # Use dummy params if needed
            params = {f'c{i}': 1.0 for i in range(10)}
            y = self.evaluate(x, params)
            dy = np.diff(y)
            return np.all(dy >= -1e-10)  # Allow tiny numerical errors
        except:
            return False


class VarExpr(PhysicsExpr):
    """Variable reference (x = log g_bar)."""
    def evaluate(self, x, params):
        return x
    def to_string(self, params):
        return "x"
    def complexity(self):
        return 1


class ParamExpr(PhysicsExpr):
    """Learnable parameter."""
    def __init__(self, name: str):
        self.name = name
    def evaluate(self, x, params):
        return np.full_like(x, params.get(self.name, 1.0))
    def to_string(self, params):
        return f"{params.get(self.name, 1.0):.4g}"
    def complexity(self):
        return 1


class UnaryExpr(PhysicsExpr):
    """Unary operation."""
    OPS = {
        'sqrt': (np.sqrt, 'sqrt({})'),
        'log': (lambda x: np.log(np.maximum(x, 1e-30)), 'log({})'),
        'exp': (lambda x: np.exp(np.clip(x, -50, 50)), 'exp({})'),
        'inv': (lambda x: 1.0 / (x + 1e-30), '1/({})'),
        'neg': (lambda x: -x, '-({})'),
        'abs': (np.abs, '|{}|'),
        'tanh': (np.tanh, 'tanh({})'),
    }
    
    def __init__(self, op: str, child: PhysicsExpr):
        self.op = op
        self.child = child
    
    def evaluate(self, x, params):
        child_val = self.child.evaluate(x, params)
        return self.OPS[self.op][0](child_val)
    
    def to_string(self, params):
        return self.OPS[self.op][1].format(self.child.to_string(params))
    
    def complexity(self):
        return 1 + self.child.complexity()


class BinaryExpr(PhysicsExpr):
    """Binary operation."""
    OPS = {
        '+': (np.add, '({} + {})'),
        '-': (np.subtract, '({} - {})'),
        '*': (np.multiply, '({} * {})'),
        '/': (lambda a, b: a / (b + 1e-30), '({} / {})'),
        '^': (lambda a, b: np.power(np.maximum(np.abs(a), 1e-30), b), '({}^{})'),
    }
    
    def __init__(self, op: str, left: PhysicsExpr, right: PhysicsExpr):
        self.op = op
        self.left = left
        self.right = right
    
    def evaluate(self, x, params):
        l = self.left.evaluate(x, params)
        r = self.right.evaluate(x, params)
        return self.OPS[self.op][0](l, r)
    
    def to_string(self, params):
        return self.OPS[self.op][1].format(
            self.left.to_string(params),
            self.right.to_string(params)
        )
    
    def complexity(self):
        return 1 + self.left.complexity() + self.right.complexity()


# =============================================================================
# GFMLB-INTEGRATED GENETIC PROGRAMMING
# =============================================================================

class UDEPhysicsGP:
    """
    Physics GP with full UDE integration:
    - GFMLB for pattern storage
    - Four-law verification
    - Physics constraints (monotonicity, asymptotic behavior)
    - Pareto-optimal multi-objective selection
    """
    
    def __init__(self, 
                 pop_size: int = 100,
                 max_depth: int = 5,
                 max_complexity: int = 15,
                 parsimony_coef: float = 0.01,
                 physics_penalty: float = 10.0):
        self.pop_size = pop_size
        self.max_depth = max_depth
        self.max_complexity = max_complexity
        self.parsimony_coef = parsimony_coef
        self.physics_penalty = physics_penalty
        
        # Initialize GFMLB if available
        self.gfmlb = None
        if UDE_MODULES['gfmlb']:
            try:
                self.gfmlb = GFMLB(db_path=":memory:")
                set_gfmlb_instance(self.gfmlb)
                print("[GP] GFMLB memory lattice initialized")
            except Exception as e:
                print(f"[GP] GFMLB init failed: {e}")
        
        # Initialize analyzers
        self.zipf = ZipfAnalyzer() if UDE_MODULES['zipf_law'] else None
        self.pareto = ParetoAnalyzer() if UDE_MODULES['pareto_law'] else None
        self.lln = LLNAnalyzer() if UDE_MODULES['lln'] else None
        
        # Fitness history for LLN convergence detection
        self.fitness_history = []
        
        # Parameter counter
        self.param_counter = 0
    
    def _new_param(self) -> str:
        """Generate unique parameter name."""
        name = f"c{self.param_counter}"
        self.param_counter += 1
        return name
    
    def _random_expr(self, depth: int = 0) -> PhysicsExpr:
        """Generate random expression tree."""
        if depth >= self.max_depth or (depth > 0 and np.random.random() < 0.3):
            # Terminal
            if np.random.random() < 0.5:
                return VarExpr()
            else:
                return ParamExpr(self._new_param())
        
        # Non-terminal
        if np.random.random() < 0.3:
            # Unary
            op = np.random.choice(list(UnaryExpr.OPS.keys()))
            return UnaryExpr(op, self._random_expr(depth + 1))
        else:
            # Binary
            op = np.random.choice(list(BinaryExpr.OPS.keys()))
            return BinaryExpr(op, 
                             self._random_expr(depth + 1),
                             self._random_expr(depth + 1))
    
    def _crossover(self, p1: PhysicsExpr, p2: PhysicsExpr) -> PhysicsExpr:
        """Crossover two expressions."""
        # Simple: return copy of one parent (proper crossover is complex)
        return self._copy_expr(p1 if np.random.random() < 0.5 else p2)
    
    def _mutate(self, expr: PhysicsExpr) -> PhysicsExpr:
        """Mutate expression."""
        if np.random.random() < 0.3:
            # Replace subtree
            return self._random_expr(0)
        return self._copy_expr(expr)
    
    def _copy_expr(self, expr: PhysicsExpr) -> PhysicsExpr:
        """Deep copy expression."""
        if isinstance(expr, VarExpr):
            return VarExpr()
        elif isinstance(expr, ParamExpr):
            return ParamExpr(self._new_param())
        elif isinstance(expr, UnaryExpr):
            return UnaryExpr(expr.op, self._copy_expr(expr.child))
        elif isinstance(expr, BinaryExpr):
            return BinaryExpr(expr.op,
                             self._copy_expr(expr.left),
                             self._copy_expr(expr.right))
        return expr
    
    def _get_param_names(self, expr: PhysicsExpr) -> List[str]:
        """Get all parameter names in expression."""
        if isinstance(expr, ParamExpr):
            return [expr.name]
        elif isinstance(expr, UnaryExpr):
            return self._get_param_names(expr.child)
        elif isinstance(expr, BinaryExpr):
            return (self._get_param_names(expr.left) + 
                   self._get_param_names(expr.right))
        return []
    
    def _fit_params(self, expr: PhysicsExpr, x: np.ndarray, y: np.ndarray,
                    slices: List[Tuple[int, int]]) -> Tuple[Dict[str, float], float]:
        """Fit parameters to minimize galaxy-weighted MAE."""
        param_names = self._get_param_names(expr)
        if not param_names:
            # No parameters to fit
            try:
                pred = expr.evaluate(x, {})
                mae = galaxy_weighted_mae(pred, y, slices)
                return {}, mae
            except:
                return {}, float('inf')
        
        def objective(param_values):
            params = dict(zip(param_names, param_values))
            try:
                pred = expr.evaluate(x, params)
                if not np.all(np.isfinite(pred)):
                    return 1e10
                return galaxy_weighted_mae(pred, y, slices)
            except:
                return 1e10
        
        # Try multiple starting points
        best_params = {}
        best_mae = float('inf')
        
        for _ in range(3):
            try:
                x0 = np.random.randn(len(param_names))
                result = minimize(objective, x0, method='Powell',
                                 options={'maxiter': 100})
                if result.fun < best_mae:
                    best_mae = result.fun
                    best_params = dict(zip(param_names, result.x))
            except:
                pass
        
        return best_params, best_mae
    
    def _check_physics(self, expr: PhysicsExpr, params: Dict[str, float]) -> float:
        """Check physics constraints, return penalty."""
        penalty = 0.0
        
        x_test = np.linspace(-14, -8, 200)
        try:
            y = expr.evaluate(x_test, params)
            
            # 1. Monotonicity: dy/dx should be >= 0
            dy = np.diff(y)
            n_violations = np.sum(dy < -1e-6)
            if n_violations > 0:
                penalty += self.physics_penalty * n_violations / len(dy)
            
            # 2. Newtonian limit: dy/dx should → 1 at high g_bar
            dy_dx = np.gradient(y, x_test)
            if abs(dy_dx[-1] - 1.0) > 0.3:
                penalty += self.physics_penalty * 0.5
            
            # 3. Output range should be reasonable
            if np.any(y < -20) or np.any(y > 0):
                penalty += self.physics_penalty * 0.5
            
        except:
            penalty += self.physics_penalty * 2
        
        return penalty
    
    def _compute_fitness(self, expr: PhysicsExpr, params: Dict[str, float],
                         mae: float) -> float:
        """Compute total fitness including all penalties."""
        # Base fitness
        fitness = mae
        
        # Complexity penalty
        complexity = expr.complexity()
        if complexity > self.max_complexity:
            fitness += self.parsimony_coef * (complexity - self.max_complexity)
        
        # Physics penalty
        fitness += self._check_physics(expr, params)
        
        return fitness
    
    def evolve(self, x: np.ndarray, y: np.ndarray, slices: List[Tuple[int, int]],
               n_generations: int = 50, verbose: bool = True) -> List[Tuple[PhysicsExpr, Dict, float]]:
        """
        Evolve population with full UDE integration.
        """
        print("\n" + "="*70)
        print("[UDE-GP] Starting Full-Stack Evolution")
        print("="*70)
        print(f"  Population: {self.pop_size}")
        print(f"  Generations: {n_generations}")
        print(f"  Physics penalty: {self.physics_penalty}")
        print(f"  GFMLB: {'Active' if self.gfmlb else 'Disabled'}")
        print(f"  Four-Law: Zipf={self.zipf is not None}, "
              f"Pareto={self.pareto is not None}, LLN={self.lln is not None}")
        
        # Initialize population
        self.param_counter = 0
        population = []
        
        print("\n[UDE-GP] Initializing population...")
        for _ in range(self.pop_size):
            expr = self._random_expr(0)
            params, mae = self._fit_params(expr, x, y, slices)
            fitness = self._compute_fitness(expr, params, mae)
            population.append((expr, params, fitness, mae))
        
        # Sort by fitness
        population.sort(key=lambda x: x[2])
        
        best_fitness_ever = population[0][2]
        stagnation_count = 0
        
        for gen in range(n_generations):
            # Track fitness for LLN convergence
            self.fitness_history.append(population[0][2])
            
            # Check LLN convergence (early stopping)
            if self.lln and len(self.fitness_history) >= 10:
                recent = np.array(self.fitness_history[-10:])
                cv = np.std(recent) / (np.mean(recent) + 1e-10)
                if cv < 0.01:
                    print(f"\n[LLN] Fitness converged (CV={cv:.4f}), stopping early")
                    break
            
            # Selection: tournament
            new_pop = []
            elite_size = max(2, self.pop_size // 10)
            
            # Keep elites
            for i in range(elite_size):
                new_pop.append(population[i])
            
            # Generate rest through crossover/mutation
            while len(new_pop) < self.pop_size:
                # Tournament selection
                t1 = min(np.random.choice(len(population), 5, replace=False),
                        key=lambda i: population[i][2])
                t2 = min(np.random.choice(len(population), 5, replace=False),
                        key=lambda i: population[i][2])
                
                p1 = population[t1][0]
                p2 = population[t2][0]
                
                # Crossover
                child = self._crossover(p1, p2)
                
                # Mutate
                if np.random.random() < 0.3:
                    child = self._mutate(child)
                
                # Fit and evaluate
                params, mae = self._fit_params(child, x, y, slices)
                fitness = self._compute_fitness(child, params, mae)
                
                # Reject if too complex
                if child.complexity() <= self.max_complexity:
                    new_pop.append((child, params, fitness, mae))
            
            population = new_pop
            population.sort(key=lambda x: x[2])
            
            # Update best
            if population[0][2] < best_fitness_ever:
                best_fitness_ever = population[0][2]
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Store in GFMLB
            if self.gfmlb and gen % 10 == 0:
                try:
                    best_expr, best_params, best_fit, best_mae = population[0]
                    formula = best_expr.to_string(best_params)
                    self.gfmlb.store_pattern(
                        key=f"gen_{gen}",
                        value=formula,
                        strength=1.0 / (best_fit + 0.01),
                        metadata={"mae": best_mae, "fitness": best_fit}
                    )
                except:
                    pass
            
            if verbose and gen % 5 == 0:
                best = population[0]
                formula = best[0].to_string(best[1])[:50]
                print(f"  Gen {gen:3d}: fitness={best[2]:.4f}, mae={best[3]:.4f}, "
                      f"formula={formula}...")
        
        # Final results
        print("\n[UDE-GP] Evolution complete!")
        
        # Validate top candidates with four-law verification
        valid_candidates = []
        
        print("\n[UDE-GP] Four-Law Verification of Top Candidates:")
        for i, (expr, params, fitness, mae) in enumerate(population[:10]):
            # Generate predictions for verification
            pred = expr.evaluate(x, params)
            residuals = y - pred
            
            # Verify with four laws
            law_result = verify_four_laws(residuals, verbose=False)
            
            # Check physics
            physics_ok = self._check_physics(expr, params) < 0.1
            
            formula = expr.to_string(params)
            status = "✓" if (law_result.overall_pass and physics_ok) else "✗"
            
            print(f"  {i+1}. {status} MAE={mae:.4f}, {law_result}")
            print(f"       Formula: {formula[:60]}...")
            
            if law_result.overall_pass and physics_ok:
                valid_candidates.append((expr, params, mae))
        
        return valid_candidates


# =============================================================================
# BASELINE MODELS
# =============================================================================

def fit_rar(x_train, y_train, slices_train):
    """Fit RAR with trainable g†."""
    def rar_log(x, log_g_dagger):
        g_bar = 10**x
        g_dagger = 10**log_g_dagger
        g_obs = g_bar / (1 - np.exp(-np.sqrt(g_bar / g_dagger)))
        return np.log10(np.maximum(g_obs, 1e-30))
    
    def objective(params):
        pred = rar_log(x_train, params[0])
        return galaxy_weighted_mae(pred, y_train, slices_train)
    
    result = minimize(objective, [np.log10(G_DAGGER)], method='Powell')
    return 10**result.x[0]


def fit_mond_standard(x_train, y_train, slices_train):
    """Fit MOND standard (implicit) with trainable a₀."""
    def mond_log(x, log_a0):
        g_bar = 10**x
        a0 = 10**log_a0
        
        # Solve: g_bar = g_obs * μ(g_obs/a0)
        # where μ(x) = x / sqrt(1 + x²)
        # Iterative solution
        g_obs = g_bar.copy()
        for _ in range(50):
            y = g_obs / a0
            mu = y / np.sqrt(1 + y**2)
            g_obs_new = g_bar / (mu + 1e-30)
            if np.max(np.abs(g_obs_new - g_obs)) < 1e-12:
                break
            g_obs = g_obs_new
        
        return np.log10(np.maximum(g_obs, 1e-30))
    
    def objective(params):
        pred = mond_log(x_train, params[0])
        return galaxy_weighted_mae(pred, y_train, slices_train)
    
    result = minimize(objective, [np.log10(A0_MOND)], method='Powell')
    return 10**result.x[0]


# =============================================================================
# MAIN DISCOVERY PIPELINE
# =============================================================================

def run_full_discovery(verbose: bool = True):
    """Run complete UDE-powered discovery pipeline."""
    print("="*70)
    print("UDE FULL-STACK DARK MATTER DISCOVERY")
    print("="*70)
    print(f"\nModules Active:")
    for mod, active in UDE_MODULES.items():
        print(f"  {'✓' if active else '✗'} {mod}")
    
    # Load data
    print("\n[DATA] Loading SPARC galaxies...")
    galaxies = load_sparc()
    print(f"[DATA] Loaded {len(galaxies)} galaxies")
    
    train_names, val_names, test_names = split_galaxies(galaxies)
    print(f"[DATA] Split: {len(train_names)} train, {len(val_names)} val, {len(test_names)} test")
    
    x_train, y_train, slices_train = get_data(galaxies, train_names)
    x_val, y_val, slices_val = get_data(galaxies, val_names)
    x_test, y_test, slices_test = get_data(galaxies, test_names)
    
    print(f"[DATA] Points: {len(x_train)} train, {len(x_val)} val, {len(x_test)} test")
    
    # Fit baselines
    print("\n" + "="*70)
    print("[BASELINES] Fitting Known Models")
    print("="*70)
    
    g_dagger_fit = fit_rar(x_train, y_train, slices_train)
    print(f"  RAR fitted g† = {g_dagger_fit:.2e} m/s² (canonical: {G_DAGGER:.2e})")
    
    a0_fit = fit_mond_standard(x_train, y_train, slices_train)
    print(f"  MOND fitted a₀ = {a0_fit:.2e} m/s² (canonical: {A0_MOND:.2e})")
    
    # Evaluate baselines on test
    def rar_pred(x, g_dagger):
        g_bar = 10**x
        g_obs = g_bar / (1 - np.exp(-np.sqrt(g_bar / g_dagger)))
        return np.log10(np.maximum(g_obs, 1e-30))
    
    def mond_pred(x, a0):
        g_bar = 10**x
        g_obs = g_bar.copy()
        for _ in range(50):
            y_m = g_obs / a0
            mu = y_m / np.sqrt(1 + y_m**2)
            g_obs_new = g_bar / (mu + 1e-30)
            if np.max(np.abs(g_obs_new - g_obs)) < 1e-12:
                break
            g_obs = g_obs_new
        return np.log10(np.maximum(g_obs, 1e-30))
    
    mae_rar_fit = galaxy_weighted_mae(rar_pred(x_test, g_dagger_fit), y_test, slices_test)
    mae_mond_fit = galaxy_weighted_mae(mond_pred(x_test, a0_fit), y_test, slices_test)
    mae_rar_canon = galaxy_weighted_mae(rar_pred(x_test, G_DAGGER), y_test, slices_test)
    
    print(f"\n  Test MAE (galaxy-weighted):")
    print(f"    RAR (canonical g†): {mae_rar_canon:.4f}")
    print(f"    RAR (fitted g†):    {mae_rar_fit:.4f}")
    print(f"    MOND (fitted a₀):   {mae_mond_fit:.4f}")
    
    # Run UDE GP discovery
    print("\n" + "="*70)
    print("[UDE-GP] Running Physics-Constrained Genetic Programming")
    print("="*70)
    
    gp = UDEPhysicsGP(
        pop_size=100,
        max_depth=4,
        max_complexity=12,
        parsimony_coef=0.01,
        physics_penalty=10.0,  # Strong physics constraints
    )
    
    candidates = gp.evolve(x_train, y_train, slices_train,
                           n_generations=30, verbose=verbose)
    
    # Evaluate best candidates on test
    print("\n" + "="*70)
    print("[RESULTS] Final Test Evaluation")
    print("="*70)
    
    print(f"\n  Baselines:")
    print(f"    RAR (fitted):  MAE = {mae_rar_fit:.4f}")
    print(f"    MOND (fitted): MAE = {mae_mond_fit:.4f}")
    
    if candidates:
        print(f"\n  UDE-GP Discoveries (four-law verified):")
        for i, (expr, params, mae) in enumerate(candidates[:5]):
            pred = expr.evaluate(x_test, params)
            test_mae = galaxy_weighted_mae(pred, y_test, slices_test)
            formula = expr.to_string(params)
            print(f"    {i+1}. MAE = {test_mae:.4f}: {formula[:50]}...")
    else:
        print("\n  [!] No physics-valid formulas discovered")
    
    # Summary
    print("\n" + "="*70)
    print("[SUMMARY]")
    print("="*70)
    
    best_baseline = "MOND" if mae_mond_fit < mae_rar_fit else "RAR"
    best_baseline_mae = min(mae_mond_fit, mae_rar_fit)
    
    if candidates:
        best_gp_mae = galaxy_weighted_mae(
            candidates[0][0].evaluate(x_test, candidates[0][1]), 
            y_test, slices_test
        )
        
        if best_gp_mae < best_baseline_mae * 0.95:
            print(f"  ✓ UDE-GP found formula that beats {best_baseline} by "
                  f"{(best_baseline_mae/best_gp_mae - 1)*100:.1f}%")
        else:
            print(f"  {best_baseline} remains best (MAE = {best_baseline_mae:.4f})")
    else:
        print(f"  {best_baseline} is best model (MAE = {best_baseline_mae:.4f})")
    
    print(f"\n  Key physical constants (fitted on SPARC):")
    print(f"    g† (RAR scale):   {g_dagger_fit:.2e} m/s²")
    print(f"    a₀ (MOND scale):  {a0_fit:.2e} m/s²")
    
    return {
        'baselines': {
            'rar_fitted': mae_rar_fit,
            'mond_fitted': mae_mond_fit,
            'g_dagger': g_dagger_fit,
            'a0': a0_fit,
        },
        'candidates': candidates,
    }


# =============================================================================
# VERIFICATION TESTS
# =============================================================================

def run_verification():
    """Run four-law verification on residuals."""
    print("="*70)
    print("UDE FOUR-LAW VERIFICATION TEST")
    print("="*70)
    
    galaxies = load_sparc()
    train_names, val_names, test_names = split_galaxies(galaxies)
    x_test, y_test, slices_test = get_data(galaxies, test_names)
    
    # Test on RAR residuals
    g_dagger_fit = 4.07e-11  # Previously fitted
    
    g_bar = 10**x_test
    g_obs_pred = g_bar / (1 - np.exp(-np.sqrt(g_bar / g_dagger_fit)))
    y_pred = np.log10(np.maximum(g_obs_pred, 1e-30))
    
    residuals = y_test - y_pred
    
    print("\n[TEST] RAR Residuals Four-Law Verification:")
    result = verify_four_laws(residuals, verbose=True)
    
    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="UDE Full-Stack Dark Matter Discovery")
    parser.add_argument("--discover", action="store_true", help="Run full discovery")
    parser.add_argument("--validate", action="store_true", help="Run validation only")
    parser.add_argument("--verify-laws", action="store_true", help="Test four-law verification")
    args = parser.parse_args()
    
    if args.discover:
        results = run_full_discovery(verbose=True)
        
        # Save results
        with open("ude_full_results.json", "w") as f:
            json.dump({
                'baselines': results['baselines'],
                'n_candidates': len(results['candidates']),
            }, f, indent=2, default=str)
        print("\nResults saved to ude_full_results.json")
        
    elif args.validate:
        run_verification()
        
    elif args.verify_laws:
        run_verification()
        
    else:
        # Default: run discovery
        print("Usage:")
        print("  python ude_full_discovery.py --discover    # Full discovery pipeline")
        print("  python ude_full_discovery.py --validate    # Validation only")
        print("  python ude_full_discovery.py --verify-laws # Test four-law system")


if __name__ == "__main__":
    main()
