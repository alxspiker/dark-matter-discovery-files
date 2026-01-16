#!/usr/bin/env python3
"""
Paper 2c: Grammar-Based Symbolic Discovery
==========================================

GOAL: Discover acceleration laws WITHOUT pre-specifying RAR/MOND.
Use genetic programming to construct formulas from primitives.

Grammar (operators):
- Unary: sqrt, log, exp, neg, inv (1/x)
- Binary: +, -, *, /, ^
- Constants: learnable parameters

This is TRUE symbolic discovery: the algorithm must CONSTRUCT
the formula, not just select from a menu.

"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import minimize, differential_evolution
from scipy import stats
import json
import warnings
import copy
import random
warnings.filterwarnings('ignore')

# Physical constants (for comparison only - NOT used in discovery)
G_DAGGER = 1.2e-10  # m/s^2 - RAR characteristic acceleration


# =============================================================================
# SYMBOLIC EXPRESSION TREE
# =============================================================================

class Expr:
    """Base class for symbolic expressions."""
    def evaluate(self, g_bar: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        raise NotImplementedError
    
    def complexity(self) -> int:
        raise NotImplementedError
    
    def to_string(self, params: Dict[str, float]) -> str:
        raise NotImplementedError
    
    def get_param_names(self) -> List[str]:
        raise NotImplementedError
    
    def copy(self) -> 'Expr':
        raise NotImplementedError


class Var(Expr):
    """Input variable (g_bar)."""
    def __init__(self, name: str = "g_bar"):
        self.name = name
    
    def evaluate(self, g_bar: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        return g_bar
    
    def complexity(self) -> int:
        return 1
    
    def to_string(self, params: Dict[str, float]) -> str:
        return self.name
    
    def get_param_names(self) -> List[str]:
        return []
    
    def copy(self) -> 'Expr':
        return Var(self.name)


class Param(Expr):
    """Learnable parameter."""
    _counter = 0
    
    def __init__(self, name: str = None):
        if name is None:
            Param._counter += 1
            name = f"p{Param._counter}"
        self.name = name
    
    def evaluate(self, g_bar: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        return np.full_like(g_bar, params.get(self.name, 1.0))
    
    def complexity(self) -> int:
        return 1
    
    def to_string(self, params: Dict[str, float]) -> str:
        val = params.get(self.name, 1.0)
        if abs(val) < 1e-6:
            return f"{val:.2e}"
        return f"{val:.4g}"
    
    def get_param_names(self) -> List[str]:
        return [self.name]
    
    def copy(self) -> 'Expr':
        return Param(self.name)


class UnaryOp(Expr):
    """Unary operation: sqrt, log, exp, neg, inv."""
    SAFE_OPS = {
        'sqrt': lambda x: np.sqrt(np.maximum(x, 1e-30)),
        'log': lambda x: np.log(np.maximum(x, 1e-30)),
        'exp': lambda x: np.clip(np.exp(np.clip(x, -50, 50)), 1e-30, 1e30),
        'neg': lambda x: -x,
        'inv': lambda x: 1.0 / np.maximum(np.abs(x), 1e-30) * np.sign(x + 1e-30),
    }
    
    def __init__(self, op: str, child: Expr):
        self.op = op
        self.child = child
    
    def evaluate(self, g_bar: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        child_val = self.child.evaluate(g_bar, params)
        result = self.SAFE_OPS[self.op](child_val)
        return np.where(np.isfinite(result), result, 0.0)
    
    def complexity(self) -> int:
        return 1 + self.child.complexity()
    
    def to_string(self, params: Dict[str, float]) -> str:
        child_str = self.child.to_string(params)
        if self.op == 'neg':
            return f"(-{child_str})"
        elif self.op == 'inv':
            return f"(1/{child_str})"
        return f"{self.op}({child_str})"
    
    def get_param_names(self) -> List[str]:
        return self.child.get_param_names()
    
    def copy(self) -> 'Expr':
        return UnaryOp(self.op, self.child.copy())


class BinaryOp(Expr):
    """Binary operation: +, -, *, /, ^."""
    SAFE_OPS = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: a / np.maximum(np.abs(b), 1e-30) * np.sign(b + 1e-30),
        '^': lambda a, b: np.power(np.maximum(np.abs(a), 1e-30), np.clip(b, -10, 10)) * np.sign(a + 1e-30),
    }
    
    def __init__(self, op: str, left: Expr, right: Expr):
        self.op = op
        self.left = left
        self.right = right
    
    def evaluate(self, g_bar: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        left_val = self.left.evaluate(g_bar, params)
        right_val = self.right.evaluate(g_bar, params)
        result = self.SAFE_OPS[self.op](left_val, right_val)
        return np.where(np.isfinite(result), result, 0.0)
    
    def complexity(self) -> int:
        return 1 + self.left.complexity() + self.right.complexity()
    
    def to_string(self, params: Dict[str, float]) -> str:
        left_str = self.left.to_string(params)
        right_str = self.right.to_string(params)
        if self.op == '^':
            return f"({left_str}^{right_str})"
        return f"({left_str} {self.op} {right_str})"
    
    def get_param_names(self) -> List[str]:
        return self.left.get_param_names() + self.right.get_param_names()
    
    def copy(self) -> 'Expr':
        return BinaryOp(self.op, self.left.copy(), self.right.copy())


# =============================================================================
# GENETIC PROGRAMMING ENGINE
# =============================================================================

class GrammarGP:
    """Genetic Programming for symbolic discovery."""
    
    UNARY_OPS = ['sqrt', 'log', 'exp', 'inv']
    BINARY_OPS = ['+', '-', '*', '/']
    
    def __init__(self, 
                 pop_size: int = 100,
                 max_depth: int = 5,
                 max_complexity: int = 15,
                 tournament_size: int = 5,
                 crossover_prob: float = 0.7,
                 mutation_prob: float = 0.3,
                 parsimony_coef: float = 0.01):
        self.pop_size = pop_size
        self.max_depth = max_depth
        self.max_complexity = max_complexity
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.parsimony_coef = parsimony_coef
    
    def random_expr(self, depth: int = 0, complexity_budget: int = 10) -> Expr:
        """Generate random expression within constraints."""
        if depth >= self.max_depth or complexity_budget <= 2:
            # Terminal: Var or Param
            return Var() if random.random() < 0.7 else Param()
        
        choice = random.random()
        
        if choice < 0.3:
            # Terminal
            return Var() if random.random() < 0.7 else Param()
        elif choice < 0.5:
            # Unary
            op = random.choice(self.UNARY_OPS)
            child = self.random_expr(depth + 1, complexity_budget - 1)
            return UnaryOp(op, child)
        else:
            # Binary
            op = random.choice(self.BINARY_OPS)
            remaining = max(2, complexity_budget - 2)
            budget_left = random.randint(1, remaining)
            left = self.random_expr(depth + 1, budget_left)
            right = self.random_expr(depth + 1, max(1, complexity_budget - 1 - budget_left))
            return BinaryOp(op, left, right)
    
    def mutate(self, expr: Expr, depth: int = 0) -> Expr:
        """Mutate expression tree."""
        if random.random() < 0.3:
            # Replace subtree
            return self.random_expr(depth, self.max_complexity - depth * 2)
        
        expr = expr.copy()
        
        if isinstance(expr, (Var, Param)):
            # Swap Var <-> Param
            return Param() if isinstance(expr, Var) else Var()
        elif isinstance(expr, UnaryOp):
            if random.random() < 0.5:
                # Change operator
                expr.op = random.choice(self.UNARY_OPS)
            else:
                # Mutate child
                expr.child = self.mutate(expr.child, depth + 1)
        elif isinstance(expr, BinaryOp):
            if random.random() < 0.3:
                # Change operator
                expr.op = random.choice(self.BINARY_OPS)
            elif random.random() < 0.5:
                # Mutate left
                expr.left = self.mutate(expr.left, depth + 1)
            else:
                # Mutate right
                expr.right = self.mutate(expr.right, depth + 1)
        
        return expr
    
    def crossover(self, parent1: Expr, parent2: Expr) -> Tuple[Expr, Expr]:
        """Crossover two expression trees."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Simple swap at random depth
        def get_subtree(expr: Expr, depth: int = 0) -> Tuple[Expr, str]:
            if depth > 2 or isinstance(expr, (Var, Param)):
                return expr, ""
            
            if isinstance(expr, UnaryOp):
                if random.random() < 0.5:
                    return expr.child, "child"
                return get_subtree(expr.child, depth + 1)
            elif isinstance(expr, BinaryOp):
                r = random.random()
                if r < 0.33:
                    return expr.left, "left"
                elif r < 0.66:
                    return expr.right, "right"
                else:
                    if random.random() < 0.5:
                        return get_subtree(expr.left, depth + 1)
                    return get_subtree(expr.right, depth + 1)
            return expr, ""
        
        sub1, _ = get_subtree(child1)
        sub2, _ = get_subtree(child2)
        
        # For simplicity, just return mutated versions
        return self.mutate(child1), self.mutate(child2)
    
    def optimize_params(self, expr: Expr, g_bar: np.ndarray, g_obs: np.ndarray,
                        galaxy_idx: np.ndarray, n_galaxies: int) -> Tuple[Dict[str, float], float]:
        """Optimize parameters for given expression."""
        param_names = list(set(expr.get_param_names()))
        
        if not param_names:
            # No parameters to optimize
            pred = expr.evaluate(g_bar, {})
            mae = self._galaxy_weighted_mae(pred, g_obs, galaxy_idx, n_galaxies)
            return {}, mae
        
        def objective(param_values):
            params = dict(zip(param_names, param_values))
            pred = expr.evaluate(g_bar, params)
            mae = self._galaxy_weighted_mae(pred, g_obs, galaxy_idx, n_galaxies)
            return mae if np.isfinite(mae) else 1e10
        
        # Initial guess: scale to match g_obs magnitude
        x0 = [1e-10 if 'p' in name else 1.0 for name in param_names]
        
        try:
            # Quick local optimization
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 1e-8})
            best_params = dict(zip(param_names, result.x))
            best_mae = result.fun
            
            # Try differential evolution for better global search
            bounds = [(1e-15, 1e-5) if abs(x) < 1e-6 else (x * 0.01, x * 100) for x in result.x]
            try:
                de_result = differential_evolution(objective, bounds, maxiter=50, seed=42, polish=False)
                if de_result.fun < best_mae:
                    best_params = dict(zip(param_names, de_result.x))
                    best_mae = de_result.fun
            except:
                pass
                
            return best_params, best_mae
        except:
            return dict(zip(param_names, x0)), 1e10
    
    def _galaxy_weighted_mae(self, pred: np.ndarray, true: np.ndarray,
                              galaxy_idx: np.ndarray, n_galaxies: int) -> float:
        """Galaxy-weighted log-space MAE."""
        total_mae = 0.0
        for g in range(n_galaxies):
            mask = galaxy_idx == g
            if mask.sum() > 0:
                log_pred = np.log10(np.maximum(pred[mask], 1e-30))
                log_true = np.log10(np.maximum(true[mask], 1e-30))
                total_mae += np.mean(np.abs(log_pred - log_true))
        return total_mae / n_galaxies
    
    def tournament_select(self, population: List[Tuple[Expr, Dict, float]]) -> Tuple[Expr, Dict, float]:
        """Tournament selection."""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return min(tournament, key=lambda x: x[2] + self.parsimony_coef * x[0].complexity())
    
    def evolve(self, g_bar: np.ndarray, g_obs: np.ndarray,
               galaxy_idx: np.ndarray, n_galaxies: int,
               n_generations: int = 50, verbose: bool = True) -> List[Tuple[Expr, Dict, float]]:
        """Run genetic programming."""
        
        # Initialize population
        Param._counter = 0
        population = []
        for _ in range(self.pop_size):
            expr = self.random_expr(complexity_budget=self.max_complexity)
            params, mae = self.optimize_params(expr, g_bar, g_obs, galaxy_idx, n_galaxies)
            population.append((expr, params, mae))
        
        # Sort by fitness
        population.sort(key=lambda x: x[2] + self.parsimony_coef * x[0].complexity())
        
        best_ever = population[0]
        
        if verbose:
            print(f"\n[GEN 0] Best MAE: {best_ever[2]:.4f}, Complexity: {best_ever[0].complexity()}")
            print(f"        Formula: {best_ever[0].to_string(best_ever[1])}")
        
        for gen in range(1, n_generations + 1):
            new_pop = []
            
            # Elitism: keep best 10%
            elite_size = max(1, self.pop_size // 10)
            new_pop.extend(population[:elite_size])
            
            # Generate rest via crossover/mutation
            while len(new_pop) < self.pop_size:
                if random.random() < self.crossover_prob:
                    parent1 = self.tournament_select(population)[0]
                    parent2 = self.tournament_select(population)[0]
                    child1, child2 = self.crossover(parent1, parent2)
                    children = [child1, child2]
                else:
                    parent = self.tournament_select(population)[0]
                    children = [self.mutate(parent)]
                
                for child in children:
                    if child.complexity() <= self.max_complexity:
                        params, mae = self.optimize_params(child, g_bar, g_obs, galaxy_idx, n_galaxies)
                        new_pop.append((child, params, mae))
                        if len(new_pop) >= self.pop_size:
                            break
            
            population = new_pop
            population.sort(key=lambda x: x[2] + self.parsimony_coef * x[0].complexity())
            
            if population[0][2] < best_ever[2]:
                best_ever = population[0]
            
            if verbose and gen % 10 == 0:
                print(f"[GEN {gen}] Best MAE: {population[0][2]:.4f}, Complexity: {population[0][0].complexity()}")
                print(f"        Formula: {population[0][0].to_string(population[0][1])}")
        
        # Return top unique formulas
        seen = set()
        unique = []
        for expr, params, mae in population:
            formula_str = expr.to_string(params)
            if formula_str not in seen:
                seen.add(formula_str)
                unique.append((expr, params, mae))
            if len(unique) >= 10:
                break
        
        return unique


# =============================================================================
# DATA LOADING
# =============================================================================

def load_sparc_accelerations(data_dir: str = "sparc_galaxies") -> Dict[str, pd.DataFrame]:
    """Load SPARC data in acceleration space."""
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
            
            if len(df) >= 5:
                galaxies[name] = df
        except Exception:
            continue
    
    print(f"[DATA] Loaded {len(galaxies)} galaxies")
    return galaxies


def split_galaxies(galaxies: Dict[str, pd.DataFrame], 
                   train_frac: float = 0.6,
                   val_frac: float = 0.2,
                   seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """Split galaxies into train/val/test."""
    names = sorted(galaxies.keys())
    np.random.seed(seed)
    np.random.shuffle(names)
    
    n = len(names)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    return names[:n_train], names[n_train:n_train+n_val], names[n_train+n_val:]


def get_galaxy_data(galaxies: Dict[str, pd.DataFrame], 
                    names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract concatenated data with galaxy indices."""
    g_bar_list, g_obs_list, idx_list = [], [], []
    
    for i, name in enumerate(names):
        df = galaxies[name]
        g_bar_list.extend(df['g_bar'].values)
        g_obs_list.extend(df['g_obs'].values)
        idx_list.extend([i] * len(df))
    
    return np.array(g_bar_list), np.array(g_obs_list), np.array(idx_list)


# =============================================================================
# DISCOVERY PIPELINE
# =============================================================================

def grammar_discovery(galaxies: Dict[str, pd.DataFrame],
                      train_names: List[str],
                      n_generations: int = 100,
                      verbose: bool = True) -> List[Tuple[Expr, Dict, float]]:
    """
    Discover acceleration laws using grammar-based GP.
    NO pre-specified formulas - algorithm must CONSTRUCT them.
    """
    g_bar, g_obs, galaxy_idx = get_galaxy_data(galaxies, train_names)
    n_galaxies = len(train_names)
    
    if verbose:
        print("="*60)
        print("[GRAMMAR DISCOVERY] Constructing formulas from primitives")
        print("="*60)
        print(f"  Operators: sqrt, log, exp, inv, +, -, *, /")
        print(f"  Terminals: g_bar, learnable parameters")
        print(f"  Population: 100, Generations: {n_generations}")
        print(f"  Training galaxies: {n_galaxies}")
    
    gp = GrammarGP(
        pop_size=100,
        max_depth=5,
        max_complexity=15,
        parsimony_coef=0.01
    )
    
    results = gp.evolve(g_bar, g_obs, galaxy_idx, n_galaxies, 
                        n_generations=n_generations, verbose=verbose)
    
    return results


def validate_discovered(galaxies: Dict[str, pd.DataFrame],
                        val_names: List[str],
                        candidates: List[Tuple[Expr, Dict, float]],
                        verbose: bool = True) -> Tuple[Expr, Dict, float]:
    """Validate discovered formulas on held-out galaxies."""
    g_bar, g_obs, galaxy_idx = get_galaxy_data(galaxies, val_names)
    n_galaxies = len(val_names)
    
    if verbose:
        print("\n" + "="*60)
        print("[VALIDATION] Testing on held-out galaxies")
        print("="*60)
    
    results = []
    for expr, params, train_mae in candidates:
        pred = expr.evaluate(g_bar, params)
        log_pred = np.log10(np.maximum(pred, 1e-30))
        log_true = np.log10(np.maximum(g_obs, 1e-30))
        
        val_mae = 0.0
        for g in range(n_galaxies):
            mask = galaxy_idx == g
            if mask.sum() > 0:
                val_mae += np.mean(np.abs(log_pred[mask] - log_true[mask]))
        val_mae /= n_galaxies
        
        results.append((expr, params, train_mae, val_mae))
        
        if verbose:
            formula = expr.to_string(params)
            print(f"  MAE: train={train_mae:.4f}, val={val_mae:.4f}")
            print(f"      {formula[:60]}...")
    
    # Sort by validation MAE
    results.sort(key=lambda x: x[3])
    best = results[0]
    
    if verbose:
        print(f"\n[BEST] val_MAE={best[3]:.4f}")
        print(f"       {best[0].to_string(best[1])}")
    
    return best[0], best[1], best[3]


def compare_to_rar(galaxies: Dict[str, pd.DataFrame],
                   test_names: List[str],
                   discovered: Tuple[Expr, Dict],
                   verbose: bool = True) -> Dict:
    """Compare discovered formula to known RAR."""
    g_bar, g_obs, galaxy_idx = get_galaxy_data(galaxies, test_names)
    n_galaxies = len(test_names)
    
    expr, params = discovered
    
    # Discovered formula
    pred_discovered = expr.evaluate(g_bar, params)
    log_pred_d = np.log10(np.maximum(pred_discovered, 1e-30))
    log_true = np.log10(np.maximum(g_obs, 1e-30))
    
    mae_discovered = 0.0
    for g in range(n_galaxies):
        mask = galaxy_idx == g
        if mask.sum() > 0:
            mae_discovered += np.mean(np.abs(log_pred_d[mask] - log_true[mask]))
    mae_discovered /= n_galaxies
    
    # RAR baseline
    x = np.sqrt(g_bar / G_DAGGER)
    pred_rar = g_bar / (1 - np.exp(-x))
    log_pred_r = np.log10(np.maximum(pred_rar, 1e-30))
    
    mae_rar = 0.0
    for g in range(n_galaxies):
        mask = galaxy_idx == g
        if mask.sum() > 0:
            mae_rar += np.mean(np.abs(log_pred_r[mask] - log_true[mask]))
    mae_rar /= n_galaxies
    
    # Linear baseline
    pred_linear = 1.5 * g_bar
    log_pred_l = np.log10(np.maximum(pred_linear, 1e-30))
    
    mae_linear = 0.0
    for g in range(n_galaxies):
        mask = galaxy_idx == g
        if mask.sum() > 0:
            mae_linear += np.mean(np.abs(log_pred_l[mask] - log_true[mask]))
    mae_linear /= n_galaxies
    
    if verbose:
        print("\n" + "="*60)
        print("[COMPARISON] Discovered vs Known Physics")
        print("="*60)
        print(f"  Discovered formula: MAE = {mae_discovered:.4f}")
        print(f"  RAR (g†=1.2e-10):   MAE = {mae_rar:.4f}")
        print(f"  Linear (1.5*g_bar): MAE = {mae_linear:.4f}")
        
        if mae_discovered < mae_rar * 0.95:
            print(f"\n  [!] Discovered formula BEATS RAR!")
        elif mae_discovered < mae_rar * 1.05:
            print(f"\n  [=] Discovered formula is COMPARABLE to RAR")
        else:
            print(f"\n  [<] RAR is better (as expected)")
    
    return {
        'discovered_mae': mae_discovered,
        'rar_mae': mae_rar,
        'linear_mae': mae_linear,
        'discovered_formula': expr.to_string(params),
        'discovered_complexity': expr.complexity()
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Grammar-Based Symbolic Discovery")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations")
    parser.add_argument("--data-dir", default="sparc_galaxies", help="SPARC data directory")
    args = parser.parse_args()
    
    print("="*60)
    print("[PAPER 2c] Grammar-Based Symbolic Discovery")
    print("="*60)
    print("NO pre-specified formulas - algorithm constructs from primitives")
    
    # Load data
    galaxies = load_sparc_accelerations(args.data_dir)
    train_names, val_names, test_names = split_galaxies(galaxies)
    
    print(f"[SPLIT] Train: {len(train_names)}, Val: {len(val_names)}, Test: {len(test_names)}")
    
    # Discover
    candidates = grammar_discovery(galaxies, train_names, 
                                   n_generations=args.generations)
    
    # Validate
    best_expr, best_params, best_val_mae = validate_discovered(
        galaxies, val_names, candidates)
    
    # Compare to RAR
    comparison = compare_to_rar(galaxies, test_names, (best_expr, best_params))
    
    # Save results
    results = {
        'discovered_formula': best_expr.to_string(best_params),
        'discovered_complexity': best_expr.complexity(),
        'validation_mae': best_val_mae,
        **comparison
    }
    
    with open("paper2c_grammar_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("[SAVED] Results to paper2c_grammar_results.json")
    print("="*60)


if __name__ == "__main__":
    main()
