#!/usr/bin/env python3
"""
Paper 2c (FIXED): Grammar-Based Symbolic Discovery
===================================================

FIXES from AI reviewer:
1. REAL crossover (actually swap subtrees)
2. Remove differential evolution (too slow)
3. Add physics penalties (monotonicity, Newtonian limit)
4. Work in LOG space (x = log10(g_bar), y = log10(g_obs))
5. Vectorize galaxy-weighted MAE
6. Penalize unphysical formulas

"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
import json
import warnings
import copy
import random
warnings.filterwarnings('ignore')

# Physical constants (for comparison only)
G_DAGGER = 1.2e-10
LOG_G_DAGGER = np.log10(G_DAGGER)  # ~ -9.92


# =============================================================================
# SYMBOLIC EXPRESSION TREE (LOG SPACE)
# =============================================================================

class Expr:
    """Base class for symbolic expressions in LOG space."""
    def evaluate(self, log_g_bar: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Input and output are in LOG10 space."""
        raise NotImplementedError
    
    def complexity(self) -> int:
        raise NotImplementedError
    
    def to_string(self, params: Dict[str, float]) -> str:
        raise NotImplementedError
    
    def get_param_names(self) -> List[str]:
        raise NotImplementedError
    
    def copy(self) -> 'Expr':
        raise NotImplementedError
    
    def get_all_nodes(self) -> List['Expr']:
        """Get all nodes in tree (for crossover)."""
        raise NotImplementedError


class Var(Expr):
    """Input variable (log10(g_bar))."""
    def __init__(self, name: str = "x"):
        self.name = name
    
    def evaluate(self, log_g_bar: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        return log_g_bar
    
    def complexity(self) -> int:
        return 1
    
    def to_string(self, params: Dict[str, float]) -> str:
        return self.name
    
    def get_param_names(self) -> List[str]:
        return []
    
    def copy(self) -> 'Expr':
        return Var(self.name)
    
    def get_all_nodes(self) -> List['Expr']:
        return [self]


class Const(Expr):
    """Learnable constant."""
    _counter = 0
    
    def __init__(self, name: str = None, default: float = 0.0):
        if name is None:
            Const._counter += 1
            name = f"c{Const._counter}"
        self.name = name
        self.default = default
    
    def evaluate(self, log_g_bar: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        return np.full_like(log_g_bar, params.get(self.name, self.default))
    
    def complexity(self) -> int:
        return 1
    
    def to_string(self, params: Dict[str, float]) -> str:
        val = params.get(self.name, self.default)
        return f"{val:.3g}"
    
    def get_param_names(self) -> List[str]:
        return [self.name]
    
    def copy(self) -> 'Expr':
        return Const(self.name, self.default)
    
    def get_all_nodes(self) -> List['Expr']:
        return [self]


class UnaryOp(Expr):
    """Unary operations safe for log-space."""
    SAFE_OPS = {
        'neg': lambda x: -x,
        'abs': lambda x: np.abs(x),
        'sq': lambda x: x ** 2,
        'sqrt_abs': lambda x: np.sqrt(np.abs(x)),
        'tanh': lambda x: np.tanh(x),  # Bounded, good for transitions
    }
    
    def __init__(self, op: str, child: Expr):
        self.op = op
        self.child = child
    
    def evaluate(self, log_g_bar: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        child_val = self.child.evaluate(log_g_bar, params)
        result = self.SAFE_OPS[self.op](child_val)
        return np.clip(result, -30, 30)
    
    def complexity(self) -> int:
        return 1 + self.child.complexity()
    
    def to_string(self, params: Dict[str, float]) -> str:
        child_str = self.child.to_string(params)
        if self.op == 'neg':
            return f"(-{child_str})"
        elif self.op == 'sq':
            return f"({child_str})²"
        return f"{self.op}({child_str})"
    
    def get_param_names(self) -> List[str]:
        return self.child.get_param_names()
    
    def copy(self) -> 'Expr':
        return UnaryOp(self.op, self.child.copy())
    
    def get_all_nodes(self) -> List['Expr']:
        return [self] + self.child.get_all_nodes()


class BinaryOp(Expr):
    """Binary operations."""
    SAFE_OPS = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: np.where(np.abs(b) > 0.01, a / b, a * np.sign(b) * 100),
        'max': lambda a, b: np.maximum(a, b),
        'min': lambda a, b: np.minimum(a, b),
    }
    
    def __init__(self, op: str, left: Expr, right: Expr):
        self.op = op
        self.left = left
        self.right = right
    
    def evaluate(self, log_g_bar: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        left_val = self.left.evaluate(log_g_bar, params)
        right_val = self.right.evaluate(log_g_bar, params)
        result = self.SAFE_OPS[self.op](left_val, right_val)
        return np.clip(result, -30, 30)
    
    def complexity(self) -> int:
        return 1 + self.left.complexity() + self.right.complexity()
    
    def to_string(self, params: Dict[str, float]) -> str:
        left_str = self.left.to_string(params)
        right_str = self.right.to_string(params)
        if self.op in ['max', 'min']:
            return f"{self.op}({left_str}, {right_str})"
        return f"({left_str} {self.op} {right_str})"
    
    def get_param_names(self) -> List[str]:
        return self.left.get_param_names() + self.right.get_param_names()
    
    def copy(self) -> 'Expr':
        return BinaryOp(self.op, self.left.copy(), self.right.copy())
    
    def get_all_nodes(self) -> List['Expr']:
        return [self] + self.left.get_all_nodes() + self.right.get_all_nodes()


# =============================================================================
# PHYSICS-AWARE GENETIC PROGRAMMING
# =============================================================================

class PhysicsGP:
    """GP with physics constraints for log-space acceleration relations."""
    
    UNARY_OPS = ['neg', 'abs', 'sq', 'sqrt_abs', 'tanh']
    BINARY_OPS = ['+', '-', '*', '/', 'max', 'min']
    
    def __init__(self, 
                 pop_size: int = 100,
                 max_depth: int = 4,
                 max_complexity: int = 12,
                 tournament_size: int = 5,
                 crossover_prob: float = 0.7,
                 mutation_prob: float = 0.3,
                 parsimony_coef: float = 0.005,
                 physics_penalty_coef: float = 0.1):
        self.pop_size = pop_size
        self.max_depth = max_depth
        self.max_complexity = max_complexity
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.parsimony_coef = parsimony_coef
        self.physics_penalty_coef = physics_penalty_coef
        
        # Precompute physics test grid
        self.physics_grid = np.linspace(-14, -9, 100)  # log10(g_bar)
    
    def random_expr(self, depth: int = 0) -> Expr:
        """Generate random expression."""
        if depth >= self.max_depth:
            return Var() if random.random() < 0.7 else Const(default=random.gauss(0, 1))
        
        choice = random.random()
        
        if choice < 0.25:
            return Var()
        elif choice < 0.4:
            return Const(default=random.gauss(0, 1))
        elif choice < 0.6:
            op = random.choice(self.UNARY_OPS)
            return UnaryOp(op, self.random_expr(depth + 1))
        else:
            op = random.choice(self.BINARY_OPS)
            return BinaryOp(op, self.random_expr(depth + 1), self.random_expr(depth + 1))
    
    def mutate(self, expr: Expr, depth: int = 0) -> Expr:
        """Mutate expression."""
        if random.random() < 0.2:
            return self.random_expr(depth)
        
        expr = expr.copy()
        
        if isinstance(expr, (Var, Const)):
            if random.random() < 0.5:
                return Const(default=random.gauss(0, 1)) if isinstance(expr, Var) else Var()
            elif isinstance(expr, Const):
                expr.default += random.gauss(0, 0.5)
        elif isinstance(expr, UnaryOp):
            if random.random() < 0.4:
                expr.op = random.choice(self.UNARY_OPS)
            else:
                expr.child = self.mutate(expr.child, depth + 1)
        elif isinstance(expr, BinaryOp):
            r = random.random()
            if r < 0.3:
                expr.op = random.choice(self.BINARY_OPS)
            elif r < 0.6:
                expr.left = self.mutate(expr.left, depth + 1)
            else:
                expr.right = self.mutate(expr.right, depth + 1)
        
        return expr
    
    def crossover(self, parent1: Expr, parent2: Expr) -> Tuple[Expr, Expr]:
        """
        FIX #1: REAL crossover - actually swap subtrees.
        """
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Get all nodes
        nodes1 = child1.get_all_nodes()
        nodes2 = child2.get_all_nodes()
        
        if len(nodes1) < 2 or len(nodes2) < 2:
            return child1, child2
        
        # Pick random non-root nodes
        idx1 = random.randint(1, len(nodes1) - 1) if len(nodes1) > 1 else 0
        idx2 = random.randint(1, len(nodes2) - 1) if len(nodes2) > 1 else 0
        
        # Find parents and swap
        def replace_child(parent: Expr, old: Expr, new: Expr) -> bool:
            if isinstance(parent, UnaryOp) and parent.child is old:
                parent.child = new
                return True
            elif isinstance(parent, BinaryOp):
                if parent.left is old:
                    parent.left = new
                    return True
                elif parent.right is old:
                    parent.right = new
                    return True
            return False
        
        # Simple swap: just swap the subtrees by reconstructing
        # For simplicity, we'll do subtree replacement
        subtree1 = nodes1[idx1].copy()
        subtree2 = nodes2[idx2].copy()
        
        # Try to find and replace in both trees
        for node in child1.get_all_nodes():
            if replace_child(node, nodes1[idx1], subtree2):
                break
        
        for node in child2.get_all_nodes():
            if replace_child(node, nodes2[idx2], subtree1):
                break
        
        return child1, child2
    
    def physics_penalty(self, expr: Expr, params: Dict[str, float]) -> float:
        """
        FIX #3: Penalize unphysical behavior.
        
        Physical constraints:
        1. Monotonically increasing (more baryons -> more observed gravity)
        2. Approaches x at high g_bar (Newtonian limit: g_obs ~ g_bar)
        3. Finite and positive everywhere
        """
        try:
            pred = expr.evaluate(self.physics_grid, params)
            
            penalty = 0.0
            
            # Check finite
            if not np.all(np.isfinite(pred)):
                return 10.0
            
            # Monotonicity: derivative should be positive
            diff = np.diff(pred)
            n_decreasing = np.sum(diff < -0.01)
            penalty += 0.5 * n_decreasing / len(diff)
            
            # Newtonian limit: at high g_bar, log(g_obs) ~ log(g_bar)
            # i.e., pred ~ x when x > -10
            high_g_mask = self.physics_grid > -10
            if high_g_mask.sum() > 5:
                deviation = np.abs(pred[high_g_mask] - self.physics_grid[high_g_mask])
                penalty += 0.3 * np.mean(deviation)
            
            # Reasonable range
            if np.any(pred > 0) or np.any(pred < -20):
                penalty += 1.0
            
            return penalty
            
        except Exception:
            return 10.0
    
    def optimize_params(self, expr: Expr, log_g_bar: np.ndarray, log_g_obs: np.ndarray,
                        galaxy_slices: List[Tuple[int, int]]) -> Tuple[Dict[str, float], float]:
        """
        FIX #2: Removed differential evolution, just use Powell.
        FIX #5: Use precomputed galaxy slices for speed.
        """
        param_names = list(set(expr.get_param_names()))
        
        if not param_names:
            pred = expr.evaluate(log_g_bar, {})
            mae = self._fast_galaxy_mae(pred, log_g_obs, galaxy_slices)
            physics = self.physics_penalty(expr, {})
            return {}, mae + self.physics_penalty_coef * physics
        
        def objective(param_values):
            params = dict(zip(param_names, param_values))
            pred = expr.evaluate(log_g_bar, params)
            mae = self._fast_galaxy_mae(pred, log_g_obs, galaxy_slices)
            physics = self.physics_penalty(expr, params)
            return mae + self.physics_penalty_coef * physics
        
        # Initial guess from defaults
        x0 = [expr.copy().get_all_nodes()[i].default 
              if hasattr(expr.copy().get_all_nodes()[i], 'default') else 0.0 
              for i, name in enumerate(param_names)]
        x0 = [0.0] * len(param_names)  # Start at 0
        
        try:
            # Quick Powell optimization (FIX #2: no DE)
            result = minimize(objective, x0, method='Powell',
                            options={'maxiter': 100, 'ftol': 1e-4})
            best_params = dict(zip(param_names, result.x))
            best_fitness = result.fun
            
            return best_params, best_fitness
        except Exception:
            return dict(zip(param_names, x0)), 100.0
    
    def _fast_galaxy_mae(self, pred: np.ndarray, true: np.ndarray,
                         galaxy_slices: List[Tuple[int, int]]) -> float:
        """
        FIX #5: Vectorized galaxy-weighted MAE using precomputed slices.
        """
        total_mae = 0.0
        for start, end in galaxy_slices:
            if end > start:
                total_mae += np.mean(np.abs(pred[start:end] - true[start:end]))
        return total_mae / len(galaxy_slices)
    
    def tournament_select(self, population: List[Tuple[Expr, Dict, float]]) -> Tuple[Expr, Dict, float]:
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return min(tournament, key=lambda x: x[2] + self.parsimony_coef * x[0].complexity())
    
    def evolve(self, log_g_bar: np.ndarray, log_g_obs: np.ndarray,
               galaxy_slices: List[Tuple[int, int]],
               n_generations: int = 50, verbose: bool = True) -> List[Tuple[Expr, Dict, float]]:
        """Run genetic programming."""
        
        # Reset counter
        Const._counter = 0
        
        # Initialize population
        if verbose:
            print("  Initializing population...")
        
        population = []
        for i in range(self.pop_size):
            expr = self.random_expr()
            if expr.complexity() <= self.max_complexity:
                params, fitness = self.optimize_params(expr, log_g_bar, log_g_obs, galaxy_slices)
                population.append((expr, params, fitness))
            
            if verbose and (i + 1) % 20 == 0:
                print(f"    {i+1}/{self.pop_size} individuals initialized")
        
        population.sort(key=lambda x: x[2])
        best_ever = population[0]
        
        if verbose:
            print(f"\n[GEN 0] Best fitness: {best_ever[2]:.4f}, Complexity: {best_ever[0].complexity()}")
            print(f"        Formula: {best_ever[0].to_string(best_ever[1])}")
        
        for gen in range(1, n_generations + 1):
            new_pop = []
            
            # Elitism
            elite_size = max(2, self.pop_size // 10)
            new_pop.extend(population[:elite_size])
            
            while len(new_pop) < self.pop_size:
                if random.random() < self.crossover_prob:
                    p1 = self.tournament_select(population)[0]
                    p2 = self.tournament_select(population)[0]
                    c1, c2 = self.crossover(p1, p2)
                    children = [c1, c2]
                else:
                    parent = self.tournament_select(population)[0]
                    children = [self.mutate(parent)]
                
                for child in children:
                    if child.complexity() <= self.max_complexity:
                        params, fitness = self.optimize_params(child, log_g_bar, log_g_obs, galaxy_slices)
                        new_pop.append((child, params, fitness))
                        if len(new_pop) >= self.pop_size:
                            break
            
            population = new_pop
            population.sort(key=lambda x: x[2])
            
            if population[0][2] < best_ever[2]:
                best_ever = population[0]
            
            if verbose and gen % 5 == 0:
                print(f"[GEN {gen}] Best fitness: {population[0][2]:.4f}, Complexity: {population[0][0].complexity()}")
                print(f"        Formula: {population[0][0].to_string(population[0][1])}")
        
        # Return unique top formulas
        seen = set()
        unique = []
        for expr, params, fitness in population:
            formula_str = expr.to_string(params)
            if formula_str not in seen:
                seen.add(formula_str)
                unique.append((expr, params, fitness))
            if len(unique) >= 10:
                break
        
        return unique


# =============================================================================
# DATA LOADING (LOG SPACE)
# =============================================================================

def load_sparc_log_space(data_dir: str = "sparc_galaxies") -> Tuple[Dict[str, pd.DataFrame], np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    FIX #4: Load data directly in LOG space.
    Returns log10(g_bar), log10(g_obs), and galaxy slices for fast indexing.
    """
    data_path = Path(data_dir)
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
            df['g_bar'] = df['V_bar']**2 / (df['R'] * 3.086e19) * 1e6
            df['g_obs'] = df['V_obs']**2 / (df['R'] * 3.086e19) * 1e6
            
            df = df[(df['g_bar'] > 1e-15) & (df['g_obs'] > 1e-15)].copy()
            
            # Store log values
            df['log_g_bar'] = np.log10(df['g_bar'])
            df['log_g_obs'] = np.log10(df['g_obs'])
            
            if len(df) >= 5:
                galaxies[name] = df
        except Exception:
            continue
    
    print(f"[DATA] Loaded {len(galaxies)} galaxies")
    return galaxies


def prepare_training_data(galaxies: Dict[str, pd.DataFrame], 
                          names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Prepare concatenated log-space data with galaxy slices."""
    log_g_bar_list = []
    log_g_obs_list = []
    slices = []
    
    idx = 0
    for name in names:
        df = galaxies[name]
        n = len(df)
        log_g_bar_list.extend(df['log_g_bar'].values)
        log_g_obs_list.extend(df['log_g_obs'].values)
        slices.append((idx, idx + n))
        idx += n
    
    return np.array(log_g_bar_list), np.array(log_g_obs_list), slices


def split_galaxies(galaxies: Dict[str, pd.DataFrame], 
                   seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    names = sorted(galaxies.keys())
    np.random.seed(seed)
    np.random.shuffle(names)
    
    n = len(names)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    
    return names[:n_train], names[n_train:n_train+n_val], names[n_train+n_val:]


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def grammar_discovery_v2(galaxies: Dict[str, pd.DataFrame],
                         train_names: List[str],
                         n_generations: int = 30,
                         verbose: bool = True) -> List[Tuple[Expr, Dict, float]]:
    """Fixed grammar discovery."""
    log_g_bar, log_g_obs, slices = prepare_training_data(galaxies, train_names)
    
    if verbose:
        print("="*60)
        print("[GRAMMAR DISCOVERY v2] Physics-constrained log-space search")
        print("="*60)
        print(f"  Working in LOG10 space: x = log10(g_bar), y = log10(g_obs)")
        print(f"  Operators: +, -, *, /, max, min, neg, sq, sqrt_abs, tanh")
        print(f"  Physics penalties: monotonicity + Newtonian limit")
        print(f"  Population: 100, Generations: {n_generations}")
        print(f"  Training: {len(train_names)} galaxies, {len(log_g_bar)} points")
    
    gp = PhysicsGP(
        pop_size=100,
        max_depth=4,
        max_complexity=12,
        parsimony_coef=0.005,
        physics_penalty_coef=0.1
    )
    
    return gp.evolve(log_g_bar, log_g_obs, slices, 
                     n_generations=n_generations, verbose=verbose)


def validate_discovered_v2(galaxies: Dict[str, pd.DataFrame],
                           val_names: List[str],
                           candidates: List[Tuple[Expr, Dict, float]],
                           verbose: bool = True) -> Tuple[Expr, Dict, float]:
    """Validate on held-out galaxies."""
    log_g_bar, log_g_obs, slices = prepare_training_data(galaxies, val_names)
    
    if verbose:
        print("\n" + "="*60)
        print("[VALIDATION]")
        print("="*60)
    
    results = []
    for expr, params, train_fitness in candidates:
        pred = expr.evaluate(log_g_bar, params)
        
        val_mae = 0.0
        for start, end in slices:
            val_mae += np.mean(np.abs(pred[start:end] - log_g_obs[start:end]))
        val_mae /= len(slices)
        
        results.append((expr, params, train_fitness, val_mae))
        
        if verbose:
            formula = expr.to_string(params)
            print(f"  train={train_fitness:.4f}, val={val_mae:.4f}: {formula[:50]}...")
    
    results.sort(key=lambda x: x[3])
    best = results[0]
    
    if verbose:
        print(f"\n[BEST] val_MAE = {best[3]:.4f}")
        print(f"       {best[0].to_string(best[1])}")
    
    return best[0], best[1], best[3]


def compare_to_rar_v2(galaxies: Dict[str, pd.DataFrame],
                      test_names: List[str],
                      discovered: Tuple[Expr, Dict],
                      verbose: bool = True) -> Dict:
    """Compare to RAR in log space."""
    log_g_bar, log_g_obs, slices = prepare_training_data(galaxies, test_names)
    
    expr, params = discovered
    
    # Discovered
    pred_discovered = expr.evaluate(log_g_bar, params)
    mae_discovered = sum(np.mean(np.abs(pred_discovered[s:e] - log_g_obs[s:e])) 
                         for s, e in slices) / len(slices)
    
    # RAR in log space: log(g_obs) = log(g_bar) - log(1 - exp(-sqrt(g_bar/g†)))
    g_bar = 10**log_g_bar
    x = np.sqrt(g_bar / G_DAGGER)
    rar_pred = g_bar / (1 - np.exp(-x))
    log_rar_pred = np.log10(np.maximum(rar_pred, 1e-30))
    mae_rar = sum(np.mean(np.abs(log_rar_pred[s:e] - log_g_obs[s:e])) 
                  for s, e in slices) / len(slices)
    
    # Linear: log(g_obs) = log(g_bar) (i.e., g_obs = g_bar)
    mae_linear = sum(np.mean(np.abs(log_g_bar[s:e] - log_g_obs[s:e])) 
                     for s, e in slices) / len(slices)
    
    if verbose:
        print("\n" + "="*60)
        print("[COMPARISON] Test set (log-space MAE)")
        print("="*60)
        print(f"  Discovered:     {mae_discovered:.4f}")
        print(f"  RAR (g†=1.2e-10): {mae_rar:.4f}")
        print(f"  Linear (y=x):   {mae_linear:.4f}")
        
        if mae_discovered < mae_rar * 0.95:
            print(f"\n  [!] Discovered formula BEATS RAR!")
        elif mae_discovered < mae_rar * 1.05:
            print(f"\n  [=] Discovered ~ RAR (may have rediscovered it)")
        else:
            print(f"\n  [<] RAR is better")
    
    return {
        'discovered_mae': float(mae_discovered),
        'rar_mae': float(mae_rar),
        'linear_mae': float(mae_linear),
        'discovered_formula': expr.to_string(params),
        'discovered_complexity': expr.complexity()
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Grammar Discovery (FIXED)")
    parser.add_argument("--generations", type=int, default=30, help="Number of generations")
    parser.add_argument("--data-dir", default="sparc_galaxies", help="SPARC data directory")
    args = parser.parse_args()
    
    print("="*60)
    print("[PAPER 2c FIXED] Grammar-Based Symbolic Discovery")
    print("="*60)
    print("FIXES APPLIED:")
    print("  1. Real crossover (subtree swap)")
    print("  2. No differential evolution (faster)")
    print("  3. Physics penalties (monotonic + Newtonian limit)")
    print("  4. Log-space representation")
    print("  5. Vectorized galaxy-weighted MAE")
    
    galaxies = load_sparc_log_space(args.data_dir)
    train_names, val_names, test_names = split_galaxies(galaxies)
    
    print(f"[SPLIT] Train: {len(train_names)}, Val: {len(val_names)}, Test: {len(test_names)}")
    
    # Discover
    candidates = grammar_discovery_v2(galaxies, train_names, 
                                      n_generations=args.generations)
    
    # Validate
    best_expr, best_params, best_val_mae = validate_discovered_v2(
        galaxies, val_names, candidates)
    
    # Compare
    comparison = compare_to_rar_v2(galaxies, test_names, (best_expr, best_params))
    
    # Save
    results = {
        'discovered_formula': best_expr.to_string(best_params),
        'discovered_complexity': best_expr.complexity(),
        'validation_mae': float(best_val_mae),
        **comparison
    }
    
    with open("paper2c_grammar_results_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("[SAVED] Results to paper2c_grammar_results_v2.json")
    print("="*60)


if __name__ == "__main__":
    main()
