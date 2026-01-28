"""
git_segregation.py

Utilities for two-group segregation measures, compositional corrections,
a stochastic Schelling segregation simulator, batch experiment utilities,
neighborhood partitioning and lightweight plotting helpers.

This file is intended to be shared alongside notebooks that show usage.

Author: Boris B
Version: 2026-01-27
"""

# Standard libraries
import os
import re
import time
import math
from datetime import datetime
from collections import defaultdict
from typing import Tuple, Optional, List, Dict, Union, Set

# Third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection

from scipy.optimize import root
from scipy.special import logsumexp
from scipy.ndimage import label as connected_components

# -----------------------------
# Module metadata
# -----------------------------
__version__ = "2026.01.27"
__author__ = "Boris B"

# Default color palette (used only by plotting helpers)
colors = [
    '#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161',
    '#fbafe4', '#949494', '#ece133', '#56b4e9'
]


# -----------------------------
# Helper functions
# -----------------------------
def _to_numpy(x):
    """Convert pandas objects to numpy arrays, leave numpy arrays unchanged."""
    try:
        return x.to_numpy()
    except AttributeError:
        return np.asarray(x)


def _prepare_data(x: Union[np.ndarray, pd.DataFrame]):
    """
    Prepare input counts.

    Input: x shape (n_areas, 2) with counts [group1, group2].
    Returns: (pi, t, w, pi_avg, T)
      - pi: proportion group1 in each area
      - t: total per area
      - w: group1 counts
      - pi_avg: overall group1 proportion
      - T: total population
    Areas with total 0 are removed.
    """
    x = _to_numpy(x)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("x must be (n_areas, 2) with group counts in first two columns")

    w = x[:, 0].astype(float)
    t = np.sum(x[:, :2], axis=1).astype(float)

    valid = (t != 0)
    if not np.all(valid):
        w = w[valid]
        t = t[valid]

    T = np.sum(t)
    if T == 0:
        raise ValueError("Total population is zero after removing empty areas")

    pi = w / t
    pi_avg = np.sum(w) / T
    return pi, t, w, pi_avg, T


# -----------------------------
# Raw index computations (operate on prepared arrays)
# -----------------------------
def D_raw(pi: np.ndarray, t: np.ndarray, pi_avg: float) -> float:
    """Dissimilarity index."""
    T = np.sum(t)
    denom = 2 * T * pi_avg * (1 - pi_avg)
    if denom == 0:
        return 0.0
    return np.sum(t * np.abs(pi - pi_avg)) / denom


def S_raw(pi: np.ndarray, t: np.ndarray, pi_avg: float) -> float:
    """Separation index (variance-style)."""
    T = np.sum(t)
    denom = T * pi_avg * (1 - pi_avg)
    if denom == 0:
        return 0.0
    return np.sum(t * (pi - pi_avg) ** 2) / denom


def Iso_raw(pi: np.ndarray, t: np.ndarray, pi_avg: float) -> float:
    """
    Isolation/exposure for the first column (group1).
    Swap columns before calling if you want the other group's isolation.
    """
    T = np.sum(t)
    denom = T * (1 - pi_avg)
    if denom == 0:
        return 0.0
    numer = np.sum(t * pi * (1 - pi))
    return 1.0 - numer / denom


def Ent_raw(pi: np.ndarray, t: np.ndarray, pi_avg: float) -> float:
    """
    Entropy (Theil H) index. Handles pi==0 or pi==1 safely by using 0*log(0)->0.
    """
    T = np.sum(t)
    if pi_avg == 0 or pi_avg == 1:
        return 0.0
    E = -(pi_avg * math.log(pi_avg) + (1 - pi_avg) * math.log(1 - pi_avg))
    if E == 0:
        return 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = np.where(pi > 0, pi * np.log(pi / pi_avg), 0.0)
        term2 = np.where((1 - pi) > 0, (1 - pi) * np.log((1 - pi) / (1 - pi_avg)), 0.0)
    corr = np.sum(t * (term1 + term2))
    return corr / (T * E)


def R_raw(pi: np.ndarray, t: np.ndarray, pi_avg: float) -> float:
    """Hutchens R index."""
    T = np.sum(t)
    denom = T * math.sqrt(max(pi_avg * (1 - pi_avg), 0.0))
    if denom == 0:
        return 0.0
    return 1.0 - (1.0 / denom) * np.sum(t * np.sqrt(pi * (1 - pi)))


def G_raw_fast(pi: np.ndarray, t: np.ndarray, pi_avg: float) -> float:
    """
    Computes:
      G = [sum_i sum_j t_i t_j |pi_i - pi_j|] / [2 T^2 pi_avg (1 - pi_avg)]

    This is algebraically equivalent to the double-sum but avoids forming n x n arrays.
    """
    T = np.sum(t)
    denom = 2 * (T ** 2) * pi_avg * (1 - pi_avg)
    if denom == 0:
        return 0.0

    # Sort by pi ascending
    idx = np.argsort(pi)
    pi_s = pi[idx]
    t_s = t[idx].astype(float)

    # cumulative sums for i<j contributions
    cum_t = np.cumsum(t_s)         # cum_t[j] = sum_{i<=j} t_s[i]
    cum_t_minus = np.concatenate(([0.0], cum_t[:-1]))  # sum_{i<j} t_i
    cum_tpi = np.cumsum(t_s * pi_s)
    cum_tpi_minus = np.concatenate(([0.0], cum_tpi[:-1]))

    # For each j (sorted), contribution from pairs i < j:
    # t_j * (pi_j * sum_{i<j} t_i - sum_{i<j} t_i*pi_i)
    terms = t_s * (pi_s * cum_t_minus - cum_tpi_minus)
    # Double-sum of absolute differences equals 2 * sum_{j} terms
    numer = 2.0 * np.sum(terms)
    return numer / denom


def Seg_raw(pi: np.ndarray, t: np.ndarray, pi_avg: float) -> float:
    """Gorard segregation (minority-perspective)."""
    T = np.sum(t)
    if T == 0:
        return 0.0
    return 0.5 / T * np.sum(t * np.abs((pi_avg - pi) / (1 - pi_avg)))


def Seg_Maj_raw(pi: np.ndarray, t: np.ndarray, pi_avg: float) -> float:
    """Gorard segregation (majority-perspective)."""
    T = np.sum(t)
    if T == 0:
        return 0.0
    return 0.5 / T * np.sum(t * np.abs(pi / pi_avg - 1))


def Avg_raw(pi: np.ndarray, t: np.ndarray, pi_avg: float) -> float:
    """Overall proportion of group1."""
    return float(pi_avg)


# Map of raw functions (use fast G implementation)
RAW_FUNCTIONS = {
    'D': D_raw,
    'S': S_raw,
    'Iso': Iso_raw,
    'Ent': Ent_raw,
    'R': R_raw,
    'G': G_raw_fast,
    'Avg': Avg_raw,
    'Seg': Seg_raw,
    'Seg_Maj': Seg_Maj_raw,
}


# -----------------------------
# Public wrappers for indices
# -----------------------------
def D(x):
    pi, t, w, pi_avg, T = _prepare_data(x)
    return D_raw(pi, t, pi_avg)


def S(x):
    pi, t, w, pi_avg, T = _prepare_data(x)
    return S_raw(pi, t, pi_avg)


def Iso(x):
    pi, t, w, pi_avg, T = _prepare_data(x)
    return Iso_raw(pi, t, pi_avg)


def Ent(x):
    pi, t, w, pi_avg, T = _prepare_data(x)
    return Ent_raw(pi, t, pi_avg)


def R(x):
    pi, t, w, pi_avg, T = _prepare_data(x)
    return R_raw(pi, t, pi_avg)


def G(x):
    """Gini using the exact fast algorithm."""
    pi, t, w, pi_avg, T = _prepare_data(x)
    return G_raw_fast(pi, t, pi_avg)


def Avg(x):
    pi, t, w, pi_avg, T = _prepare_data(x)
    return Avg_raw(pi, t, pi_avg)


def Seg(x):
    pi, t, w, pi_avg, T = _prepare_data(x)
    return Seg_raw(pi, t, pi_avg)


def Seg_Maj(x):
    pi, t, w, pi_avg, T = _prepare_data(x)
    return Seg_Maj_raw(pi, t, pi_avg)


# -----------------------------
# Correction methods
# -----------------------------
def no_correction(x):
    """Return prepared arrays unchanged."""
    return _prepare_data(x)


def info_correction(x, pi_tar: float = 0.5):
    """
    Information-theoretic correction (exponential tilting / I-projection).

    Returns: (pi, t_prime, w_prime, pi_avg_prime, T_prime)
    where t_prime and w_prime are reweighted totals (normalized).
    """
    x = _to_numpy(x)
    pi, t, w, pi_avg, T = _prepare_data(x)

    def tar_pi(v):
        logz = logsumexp(v * pi + np.log(t))
        return pi_tar - np.sum(w * np.exp(v * pi - logz))

    # initial guess
    var = np.sum(w * pi) / T - pi_avg ** 2
    std = np.sqrt(var) if var > 0 else 0.0
    A = (pi_tar - pi_avg) / std if std > 0 else (pi_tar - pi_avg)
    try:
        v = root(tar_pi, A, method='hybr').x[0]
    except Exception:
        # If root-finding fails, fall back to no correction shift
        v = 0.0

    logz = logsumexp(v * pi + np.log(t))
    t_prime = t * np.exp(v * pi - logz)
    w_prime = w * np.exp(v * pi - logz)
    pi_avg_prime = pi_tar
    T_prime = np.sum(t_prime)

    return pi, t_prime, w_prime, pi_avg_prime, T_prime


def n_terms_correction(x, pi_tar: float = 0.5, n_terms: int = 4):
    """
    Experimental cumulant/Taylor correction (approximation).
    This routine is kept for exploration; info_correction is the
    preferred, more principled method.
    """
    x = _to_numpy(x)
    pi, t, w, pi_avg, T = _prepare_data(x)

    pi_diff = pi - pi_avg
    var = np.sum(t * pi_diff ** 2) / T
    if var <= 0:
        return pi, t.copy(), w.copy(), pi_avg, T

    mu3 = np.sum(t * pi_diff ** 3) / T
    mu4 = np.sum(t * pi_diff ** 4) / T
    mu5 = np.sum(t * pi_diff ** 5) / T
    mu6 = np.sum(t * pi_diff ** 6) / T

    k3 = mu3
    k4 = mu4 - 3 * (var ** 2)
    k5 = mu5 - 10 * k3 * var
    k6 = mu6 - 15 * k4 * var - 10 * (k3 ** 2)

    c_diff = (pi_tar - pi_avg)
    cor0 = c_diff / var
    increments = [
        -(k3 / (2 * var ** 3)) * (c_diff ** 2),
        -((k4 / (6 * var ** 4)) - (k3 ** 2 / (2 * var ** 5))) * (c_diff ** 3),
        -((k5 / (var ** 5) - 10 * (k4 * k3) / (var ** 6) + 15 * (k3 ** 3) / (var ** 7)) / 24) * (c_diff ** 4),
        -((k6 / (var ** 6) - 15 * (k5 * k3) / (var ** 7) - 10 * (k4 ** 2) / (var ** 7) +
           105 * (k4 * (k3 ** 2)) / (var ** 8) - 105 * (k3 ** 4) / (var ** 9)) / 120) * (c_diff ** 5)
    ]

    n_terms = min(max(n_terms, 0), 4)
    if n_terms > 0:
        cor0 += sum(increments[:n_terms])

    cor = pi * cor0
    # Stable exponentiation: subtract max to avoid overflow
    cor_max = np.max(cor)
    weights = np.exp(cor - cor_max)
    t_exp = t * weights
    w_exp = w * weights
    pi_avg_prime = np.sum(w_exp) / np.sum(t_exp)

    return pi, t_exp, w_exp, pi_avg_prime, np.sum(t_exp)


def compute_index_corrected(pi, t_prime, w_prime, pi_avg_eff, index_name):
    """Compute index from corrected arrays using raw kernel map."""
    if index_name not in RAW_FUNCTIONS:
        raise ValueError(f"Unknown index: {index_name}.")
    fn = RAW_FUNCTIONS[index_name]
    return fn(pi, t_prime, pi_avg_eff)


def Index(x, index_name='D', pi_tar=0.5, method='info', n_terms=4, use_target_average=False):
    """
    Unified interface for computing segregation indices with optional corrections.

    method: 'none', 'info', 'n_terms'
    """
    if method == 'none':
        pi, t_prime, w_prime, pi_avg_prime, T_prime = no_correction(x)
        pi_avg_eff = pi_tar if use_target_average else pi_avg_prime
    elif method == 'info':
        pi, t_prime, w_prime, pi_avg_prime, T_prime = info_correction(x, pi_tar=pi_tar)
        pi_avg_eff = pi_tar
    elif method == 'n_terms':
        pi, t_prime, w_prime, pi_avg_prime, T_prime = n_terms_correction(x, pi_tar=pi_tar, n_terms=n_terms)
        pi_avg_eff = pi_tar if use_target_average else pi_avg_prime
    else:
        raise ValueError(f"Unknown method: {method}. Use 'none', 'info', or 'n_terms'.")

    return compute_index_corrected(pi, t_prime, w_prime, pi_avg_eff, index_name)


# -----------------------------
# Schelling model (left behaviorally unchanged)
# -----------------------------
class SchellingModel:
    """
    Stochastic Schelling segregation model.

    Grid cell values: 0 empty, 1 red, 2 blue.
    Movement probability uses a sigmoid of happiness improvement.
    """
    EMPTY = 0
    RED = 1
    BLUE = 2

    def __init__(
        self,
        grid_size: Tuple[int, int],
        num_red: int,
        num_blue: int,
        happiness_factor: float = 3.0,
        sample_rate: float = 0.01,
        initial_grid: Optional[np.ndarray] = None,
        happiness_function: Optional[callable] = None,
        random_seed: Optional[int] = None
    ):
        self.height, self.width = grid_size
        self.grid_size = self.height * self.width
        self.num_red = num_red
        self.num_blue = num_blue
        self.happiness_factor = happiness_factor
        self.sample_rate = sample_rate
        self.happiness_function = happiness_function
        self.random_seed = random_seed

        if random_seed is not None:
            self.rng = np.random.RandomState(random_seed)
        else:
            self.rng = np.random.RandomState()

        if num_red + num_blue > self.grid_size:
            raise ValueError(
                f"Too many agents ({num_red + num_blue}) for grid size ({self.grid_size})"
            )

        if initial_grid is not None:
            if initial_grid.shape != grid_size:
                raise ValueError(
                    f"Initial grid shape {initial_grid.shape} doesn't match grid_size {grid_size}"
                )
            self.grid = initial_grid.astype(np.int8)
        else:
            self.grid = self._create_random_grid()

        self.neighbor_offsets = self._compute_neighbor_offsets()
        self.step_count = 0
        self.total_moves = 0

        print(f"Schelling Model Initialized")
        print(f"  Grid: {self.height}×{self.width}")
        print(f"  Agents: {num_red:,} red, {num_blue:,} blue")
        print(f"  Occupancy: {(num_red+num_blue)/self.grid_size*100:.1f}%")
        print(f"  Happiness factor: {happiness_factor:.2f}")
        print(f"  Sample rate: {sample_rate:.3f}")
        if happiness_function is not None:
            print(f"  Using custom happiness function")
        if random_seed is not None:
            print(f"  Random seed: {random_seed}")

    def _create_random_grid(self) -> np.ndarray:
        flat = np.zeros(self.grid_size, dtype=np.int8)
        positions = self.rng.choice(
            self.grid_size,
            self.num_red + self.num_blue,
            replace=False
        )
        flat[positions[:self.num_red]] = self.RED
        flat[positions[self.num_red:]] = self.BLUE
        return flat.reshape((self.height, self.width))

    def _compute_neighbor_offsets(self) -> np.ndarray:
        offsets = [
            (dx, dy) for dx in range(-1, 2) for dy in range(-1, 2)
            if not (dx == 0 and dy == 0)
        ]
        return np.array(offsets)

    def _count_neighbors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Count red and blue neighbors using np.roll (toroidal wrap).
        """
        red_mask = (self.grid == self.RED).astype(np.int16)
        blue_mask = (self.grid == self.BLUE).astype(np.int16)
        red_neighbors = np.zeros_like(self.grid, dtype=np.int16)
        blue_neighbors = np.zeros_like(self.grid, dtype=np.int16)

        for dx, dy in self.neighbor_offsets:
            red_neighbors += np.roll(red_mask, shift=(dx, dy), axis=(0, 1))
            blue_neighbors += np.roll(blue_mask, shift=(dx, dy), axis=(0, 1))

        return red_neighbors, blue_neighbors

    def _calculate_happiness(
        self,
        agent_types: np.ndarray,
        red_neighbors: np.ndarray,
        blue_neighbors: np.ndarray
    ) -> np.ndarray:
        if self.happiness_function is None:
            total_neighbors = red_neighbors + blue_neighbors
            same_neighbors = np.where(agent_types == self.RED, red_neighbors, blue_neighbors)
            return np.divide(
                same_neighbors, total_neighbors,
                out=np.zeros_like(same_neighbors, dtype=float),
                where=total_neighbors > 0
            )
        else:
            happiness_values = np.zeros(len(agent_types), dtype=np.float64)
            for i, agent_type in enumerate(agent_types):
                happiness_values[i] = self.happiness_function(
                    agent_type, red_neighbors[i], blue_neighbors[i]
                )
            return happiness_values

    def step(self) -> int:
        red_neighbors, blue_neighbors = self._count_neighbors()

        red_positions = np.column_stack(np.where(self.grid == self.RED))
        blue_positions = np.column_stack(np.where(self.grid == self.BLUE))

        if len(red_positions) > 0 and len(blue_positions) > 0:
            all_positions = np.vstack([red_positions, blue_positions])
            all_types = np.concatenate([
                np.full(len(red_positions), self.RED, dtype=np.int8),
                np.full(len(blue_positions), self.BLUE, dtype=np.int8)
            ])
        elif len(red_positions) > 0:
            all_positions = red_positions
            all_types = np.full(len(red_positions), self.RED, dtype=np.int8)
        elif len(blue_positions) > 0:
            all_positions = blue_positions
            all_types = np.full(len(blue_positions), self.BLUE, dtype=np.int8)
        else:
            return 0  # No agents to move

        num_to_sample = max(1, int(len(all_positions) * self.sample_rate))
        sample_indices = self.rng.choice(len(all_positions), num_to_sample, replace=False)
        selected_positions = all_positions[sample_indices]
        agent_types = all_types[sample_indices]

        empty_positions = np.column_stack(np.where(self.grid == self.EMPTY))
        if len(empty_positions) == 0:
            return 0  # No empty cells to move to

        num_potential_moves = min(len(selected_positions), len(empty_positions))
        if num_potential_moves == 0:
            return 0

        target_indices = self.rng.choice(len(empty_positions), num_potential_moves, replace=False)
        target_positions = empty_positions[target_indices]

        selected_positions = selected_positions[:num_potential_moves]
        agent_types = agent_types[:num_potential_moves]

        current_rows, current_cols = selected_positions[:, 0], selected_positions[:, 1]
        current_happiness = self._calculate_happiness(
            agent_types,
            red_neighbors[current_rows, current_cols],
            blue_neighbors[current_rows, current_cols]
        )

        target_rows, target_cols = target_positions[:, 0], target_positions[:, 1]
        target_happiness = self._calculate_happiness(
            agent_types,
            red_neighbors[target_rows, target_cols],
            blue_neighbors[target_rows, target_cols]
        )

        happiness_improvement = target_happiness - current_happiness
        move_probabilities = 1.0 / (1.0 + np.exp(-self.happiness_factor * happiness_improvement))

        move_mask = self.rng.rand(num_potential_moves) < move_probabilities
        num_moves = np.sum(move_mask)

        if num_moves == 0:
            return 0

        moving_from = selected_positions[move_mask]
        moving_to = target_positions[move_mask]
        moving_types = agent_types[move_mask]

        self.grid[moving_from[:, 0], moving_from[:, 1]] = self.EMPTY
        self.grid[moving_to[:, 0], moving_to[:, 1]] = moving_types

        return num_moves

    def run(self, steps: int, save_progress_every: int = None) -> Optional[str]:
        """
        Run simulation for `steps`. If save_progress_every is provided,
        snapshots are saved and the path to the .npy file is returned.
        """
        start_time = time.time()
        saved_grids = []

        if save_progress_every is not None:
            saved_grids.append(self.grid.copy())

        progress_every = max(1, steps // 20)

        print(f"\nRunning simulation for {steps:,} steps...")
        if save_progress_every is not None:
            print(f"Saving grids every {save_progress_every:,} steps")

        for i in range(steps):
            moves = self.step()
            self.step_count += 1
            self.total_moves += moves

            if save_progress_every is not None and (i + 1) % save_progress_every == 0:
                saved_grids.append(self.grid.copy())

            if (i + 1) % progress_every == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (steps - (i + 1)) / rate if rate > 0 else 0

                print(f"Step {i+1:,}/{steps:,} ({(i+1)/steps*100:.1f}%) | "
                      f"Moves: {moves:,} | Total: {self.total_moves:,} | "
                      f"Rate: {rate:.1f} steps/s | ETA: {eta:.0f}s")

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Simulation Complete!")
        print(f"  Steps: {self.step_count:,}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Rate: {self.step_count/elapsed:.1f} steps/s")
        print(f"  Total moves: {self.total_moves:,}")
        print(f"  Moves per step: {self.total_moves/self.step_count:.2f}")
        print(f"{'='*60}")

        if save_progress_every is not None:
            if self.step_count % save_progress_every != 0:
                saved_grids.append(self.grid.copy())

            grids_array = np.array(saved_grids, dtype=np.int8)

            save_dir = "Simulation Results"
            os.makedirs(save_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"schelling_{timestamp}_"
                f"grid{self.height}x{self.width}_"
                f"r{self.num_red}_b{self.num_blue}_"
                f"hf{self.happiness_factor:.2f}_"
                f"sr{self.sample_rate:.3f}_"
                f"steps{steps}.npy"
            )

            filepath = os.path.join(save_dir, filename)
            np.save(filepath, grids_array)

            print(f"\nSaved {len(saved_grids)} grid states to:")
            print(f"  {filepath}")
            print(f"  Array shape: {grids_array.shape}")

            return filepath

        return None


# -----------------------------
# Batch simulation utilities (resume/detect logic preserved)
# -----------------------------
def _parse_schelling_filename(filename: str):
    """
    Parse filenames produced by SchellingModel.run saved in Simulation Results.
    """
    pattern = (
        r"^schelling_(\d{8}_\d{6})_"
        r"grid(\d+)x(\d+)_"
        r"r(\d+)_b(\d+)_"
        r"hf([0-9.]+)_"
        r"sr([0-9.]+)_"
        r"steps(\d+)\.npy$"
    )
    m = re.match(pattern, filename)
    if not m:
        return None

    (
        timestamp, h, w, num_red, num_blue,
        hf_str, sr_str, steps_str
    ) = m.groups()

    return {
        "timestamp": timestamp,
        "height": int(h),
        "width": int(w),
        "num_red": int(num_red),
        "num_blue": int(num_blue),
        "happiness_factor": float(hf_str),
        "sample_rate": float(sr_str),
        "steps": int(steps_str),
        "filename": filename,
    }


def run_simulation_batch(
    grid_size: Tuple[int, int],
    parameter_sets: List[Dict],
    steps: int,
    save_progress_every: int,
    batch_name: str = None,
    replicates: int = 1
) -> pd.DataFrame:
    """
    Run multiple Schelling simulations with resume-aware behavior.

    Behavior:
      - Creates "Simulation Results/<batch_name>" for outputs.
      - If a summary exists, attempts to reuse completed .npy runs and only runs missing jobs.
      - Saves a batch_summary.csv in the batch folder with metadata.
    """
    if batch_name is None:
        batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    root_dir = "Simulation Results"
    os.makedirs(root_dir, exist_ok=True)

    batch_dir = os.path.join(root_dir, batch_name)
    os.makedirs(batch_dir, exist_ok=True)

    H, W = grid_size
    total_grid_size = H * W

    def _param_key(num_red, num_blue, hf, sr):
        return (
            int(num_red),
            int(num_blue),
            round(float(hf), 2),
            round(float(sr), 3),
        )

    # ---- Prepare list of jobs ----
    jobs = []
    for params in parameter_sets:
        num_red = int(params['num_red'])
        num_blue = int(params['num_blue'])
        happiness_factor = float(params.get('happiness_factor', 3.0))
        sample_rate = float(params.get('sample_rate', 0.01))
        base_label = params['label']

        for rep in range(1, replicates + 1):
            if replicates > 1:
                label = f"{base_label}_rep{rep}"
            else:
                label = base_label

            jobs.append({
                "label": label,
                "base_label": base_label,
                "replicate": rep,
                "num_red": num_red,
                "num_blue": num_blue,
                "happiness_factor": happiness_factor,
                "sample_rate": sample_rate,
                "param_key": _param_key(num_red, num_blue, happiness_factor, sample_rate),
                "completed": False,
                "filepath": None,
                "total_moves": None,
            })

    jobs_by_param = defaultdict(list)
    for idx, job in enumerate(jobs):
        jobs_by_param[job["param_key"]].append(idx)

    # ---- Step 1: Load existing summary if present ----
    summary_path = os.path.join(batch_dir, "batch_summary.csv")
    if os.path.exists(summary_path):
        try:
            prev_df = pd.read_csv(summary_path)
        except Exception:
            prev_df = None
    else:
        prev_df = None

    if prev_df is not None and len(prev_df) > 0:
        for row in prev_df.itertuples():
            num_red = int(getattr(row, "num_red"))
            num_blue = int(getattr(row, "num_blue"))
            hf = float(getattr(row, "happiness_factor"))
            sr = float(getattr(row, "sample_rate"))
            base_label = getattr(row, "base_label")
            replicate = int(getattr(row, "replicate"))
            label = getattr(row, "label")
            filepath = getattr(row, "filepath")

            pkey = _param_key(num_red, num_blue, hf, sr)
            for idx in jobs_by_param.get(pkey, []):
                job = jobs[idx]
                if (job["base_label"] == base_label and
                        job["replicate"] == replicate and
                        job["label"] == label):
                    if filepath and os.path.exists(filepath):
                        try:
                            arr = np.load(filepath, mmap_mode='r')
                            if (arr.ndim == 3 and
                                arr.shape[1] == H and
                                arr.shape[2] == W):
                                n_snapshots = arr.shape[0]
                                q = steps // save_progress_every
                                r = steps % save_progress_every
                                expected_min = 1 + q
                                expected_max = expected_min + (1 if r > 0 else 0)
                                if expected_min <= n_snapshots <= expected_max:
                                    job["completed"] = True
                                    job["filepath"] = filepath
                                    if "total_moves" in prev_df.columns:
                                        job["total_moves"] = getattr(row, "total_moves")
                        except Exception:
                            pass
                    break

    # ---- Step 2: Look for .npy files in batch_dir not in summary ----
    existing_paths = {job["filepath"] for job in jobs if job["filepath"] is not None}
    for fname in os.listdir(batch_dir):
        if not fname.endswith(".npy"):
            continue

        full_path = os.path.join(batch_dir, fname)
        if full_path in existing_paths:
            continue

        meta = _parse_schelling_filename(fname)
        if meta is None:
            continue

        if (meta["height"], meta["width"]) != grid_size:
            continue
        if meta["steps"] != steps:
            continue

        pkey = _param_key(
            meta["num_red"],
            meta["num_blue"],
            meta["happiness_factor"],
            meta["sample_rate"],
        )

        candidate_indices = jobs_by_param.get(pkey, [])
        for idx in candidate_indices:
            job = jobs[idx]
            if job["completed"]:
                continue

            try:
                arr = np.load(full_path, mmap_mode='r')
                if arr.ndim != 3 or arr.shape[1] != H or arr.shape[2] != W:
                    continue
            except Exception:
                continue

            job["completed"] = True
            job["filepath"] = full_path
            job["total_moves"] = None
            existing_paths.add(full_path)
            break

    n_total = len(jobs)
    n_completed = sum(1 for j in jobs if j["completed"])
    n_to_run = n_total - n_completed

    print(f"{'='*70}")
    print(f"Batch Simulation: {batch_name}")
    print(f"  Total simulations (desired): {n_total}")
    print(f"  Already completed (reused):  {n_completed}")
    print(f"  Still to run:               {n_to_run}")
    print(f"  Parameter sets: {len(parameter_sets)}")
    print(f"  Replicates per set: {replicates}")
    print(f"  Steps per simulation: {steps:,}")
    print(f"{'='*70}")

    # ---- Step 3: Run missing simulations ----
    sim_counter = 0
    for job_idx, job in enumerate(jobs, start=1):
        if job["completed"]:
            continue

        sim_counter += 1
        print(f"\n{'='*70}")
        print(f"Running new simulation {sim_counter}/{n_to_run} (job {job_idx}/{n_total})")
        print(f"{'='*70}")
        print(f"Label: {job['label']}")
        print(f"Parameters: red={job['num_red']}, blue={job['num_blue']}, "
              f"hf={job['happiness_factor']}, sr={job['sample_rate']}")

        model = SchellingModel(
            grid_size=grid_size,
            num_red=job["num_red"],
            num_blue=job["num_blue"],
            happiness_factor=job["happiness_factor"],
            sample_rate=job["sample_rate"]
        )

        filepath = model.run(steps=steps, save_progress_every=save_progress_every)

        if filepath:
            new_filepath = os.path.join(batch_dir, os.path.basename(filepath))
            os.rename(filepath, new_filepath)
            job["filepath"] = new_filepath
            job["completed"] = True
            job["total_moves"] = model.total_moves

    # ---- Step 4: Build and save summary ----
    incomplete = [j for j in jobs if not j["completed"] or j["filepath"] is None]
    if incomplete:
        print("\nWARNING: Some jobs have no valid output file and will be omitted from summary:")
        for j in incomplete:
            print(f"  - {j['label']} (red={j['num_red']}, blue={j['num_blue']})")

    results = []
    for job in jobs:
        if job["filepath"] is None:
            continue

        total_agents = job["num_red"] + job["num_blue"]
        percent_blue = (job["num_blue"] / total_agents * 100) if total_agents > 0 else 0

        results.append({
            'label': job["label"],
            'base_label': job["base_label"],
            'replicate': job["replicate"],
            'filepath': job["filepath"],
            'num_red': job["num_red"],
            'num_blue': job["num_blue"],
            'percent_blue': percent_blue,
            'total_agents': total_agents,
            'total_grid_size': total_grid_size,
            'grid_height': H,
            'grid_width': W,
            'happiness_factor': job["happiness_factor"],
            'sample_rate': job["sample_rate"],
            'steps': steps,
            'save_progress_every': save_progress_every,
            'total_moves': job["total_moves"],
        })

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*70}")
    print(f"Batch Complete / Updated!")
    print(f"  Results directory: {batch_dir}")
    print(f"  Summary saved to:  {summary_path}")
    print(f"  Total simulations in summary: {len(summary_df)}")
    print(f"{'='*70}")

    return summary_df


def load_batch_results(batch_name: str) -> pd.DataFrame:
    """Load summary CSV for a previous batch."""
    summary_path = os.path.join("Simulation Results", batch_name, "batch_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Batch summary not found: {summary_path}")
    df = pd.read_csv(summary_path)
    print(f"Loaded batch: {batch_name}")
    print(f"  Simulations: {len(df)}")
    print(f"  Parameter sets: {df['base_label'].nunique()}")
    return df


def load_simulation_by_label(batch_name: str, label: str) -> np.ndarray:
    """Load a saved simulation (snapshots) by label from a batch."""
    summary = load_batch_results(batch_name)
    matching = summary[summary['label'] == label]
    if len(matching) == 0:
        available_labels = summary['label'].unique()
        raise ValueError(
            f"No simulation found with label: '{label}'\n"
            f"Available labels: {list(available_labels)}"
        )
    filepath = matching.iloc[0]['filepath']
    grids = np.load(filepath)
    print(f"Loaded simulation: {label}")
    print(f"  Shape: {grids.shape}")
    print(f"  Snapshots: {grids.shape[0]}")
    return grids


# -----------------------------
# Neighborhood partitioning and plotting helpers
# -----------------------------

def stripe_partition(
    shape: Tuple[int, int],
    mean_size: float,
    std_size: float,
    aspect_ratio_std: float = 0.3,
    force_conditions: bool = False,
    random_seed: Optional[int] = 42,
    std_target_error: float = 0.5
) -> Tuple[np.ndarray, List[Tuple]]:
    """
    Partition a rectangular grid into stripe-like neighborhoods.

    This function produces a grid of integer labels (1..K) and a list of
    rectangular block descriptors (label, row_slice, col_slice). It supports
    two modes:

      - sampling mode (force_conditions=False): builds stripes by sampling
        neighborhood areas from a normal(mean=mean_size, sd=std_size)
        then laying out stripe heights and widths. Use this for fast,
        "random" partitions.

      - forced mode (force_conditions=True): attempts to match the
        requested mean neighborhood size and std by (1) sampling multiple
        initial partitions, (2) splitting/merging blocks to reach the
        target number of blocks, and (3) iteratively shifting horizontal
        boundary segments to match the requested std_size. This is the
        mode used in the paper.

    Important usage notes (for reproducing paper results):
      * Use the same random_seed you used when producing the paper figures.
      * To reproduce the exact partition from the paper call with:
            force_conditions=True, random_seed=<same seed>,
            std_target_error=1e-3 (or the value you used).
      * Output: (labels_array, blocks_list). `blocks_list` contains tuples
        (label, row_slice, col_slice) describing rectangular bounding boxes
        for convenience — not all blocks are guaranteed perfectly rectangular.

    Parameters
    ----------
    shape : (M, N)
        Grid shape (rows, cols).
    mean_size : float
        Target average neighborhood area (in grid-cells).
    std_size : float
        Target standard deviation of neighborhood areas.
    aspect_ratio_std : float
        Controls variability of stripe aspect ratio in sampling mode (not used
        in forced mode).
    force_conditions : bool
        Whether to apply the 'forced' routine that more strictly matches
        mean/std of block sizes.
    random_seed : int | None
        RNG seed for reproducibility. If None, uses numpy global RNG.
    std_target_error : float
        Convergence tolerance used by the forced routine.

    Returns
    -------
    labels : np.ndarray (M x N)
        Integer labels (0 unused / background, 1..K blocks).
    blocks : list of (label, row_slice, col_slice)
        Bounding slices for each block label (useful for summarizing).
    """
    M, N = shape
    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random

    if not force_conditions:
        return _stripe_partition_sampling(M, N, mean_size, std_size, rng)
    else:
        return _stripe_partition_forced(M, N, mean_size, std_size, rng, std_target_error)


# ----- sampling-based constructor (keeps original behavior) -----
def _stripe_partition_sampling(M, N, mean_size, std_size, rng):
    """
    Build stripe partition by sequential sampling of stripe heights and
    widths. This reproduces the earlier 'sampling' routine.
    """
    stripes = []
    remaining_rows = M
    # Build horizontal stripe heights until full coverage
    while remaining_rows > 0:
        A = 0
        # ensure area >= 1
        while A < 1:
            A = int(round(rng.normal(mean_size, std_size)))
        # derive approximate height from area via square-root heuristic
        h = max(2, min(remaining_rows, int(round(math.sqrt(A)))))
        # ensure we don't leave tiny remainder
        if remaining_rows - h < 2:
            h = remaining_rows
        stripes.append(h)
        remaining_rows -= h

    labels = np.zeros((M, N), dtype=int)
    blocks = []
    label = 1
    row0 = 0

    # For each horizontal stripe, tile it left-to-right with sampled widths
    for h in stripes:
        col0 = 0
        while col0 < N:
            A = 0
            while A < 1:
                A = int(round(rng.normal(mean_size, std_size)))
            w = max(2, int(math.ceil(A / h)))
            if col0 + w > N:
                w = N - col0
            labels[row0:row0+h, col0:col0+w] = label
            blocks.append((label, slice(row0, row0+h), slice(col0, col0+w)))
            label += 1
            col0 += w
        row0 += h

    return labels, blocks


# ----- forced routine (keeps original behavior / logic) -----
def _stripe_partition_forced(M, N, mean_size, std_size, rng, std_target_error):
    """
    Produce a partition whose mean neighborhood size is close to mean_size
    and whose std is close to std_size. This mirrors the multi-step process
    used in the original code: sample several candidates, choose best,
    split/merge to reach target block count, then adjust horizontal
    boundaries to hit the requested std.
    """
    total_area = M * N
    target_n_blocks = int(round(total_area / mean_size))

    # Step 1: sample a few candidates and pick the one with mean closest to target
    n_trials = 20
    best_partition = None
    best_mean_error = float('inf')
    for _ in range(n_trials):
        trial_seed = rng.randint(0, 2**31)
        trial_rng = np.random.RandomState(trial_seed)
        labels_trial, blocks_trial = _stripe_partition_sampling(M, N, mean_size, std_size, trial_rng)
        sizes = np.bincount(labels_trial.ravel())[1:]
        actual_mean = sizes.mean()
        err = abs(actual_mean - mean_size)
        if err < best_mean_error:
            best_mean_error = err
            best_partition = (labels_trial.copy(), [b for b in blocks_trial])

    labels, blocks = best_partition

    # Build size cache: label -> size
    size_cache = {int(lbl): int(np.sum(labels == lbl)) for lbl in np.unique(labels) if lbl > 0}

    # Step 2: split or merge blocks until we reach target count
    current_n = len(blocks)
    while current_n < target_n_blocks:
        sizes_arr = np.array([size_cache[b[0]] for b in blocks])
        largest_idx = int(np.argmax(sizes_arr))
        labels, blocks, size_cache = _split_block_fast(labels, blocks, largest_idx, size_cache)
        current_n += 1

    while current_n > target_n_blocks:
        labels, blocks, size_cache = _merge_smallest_blocks_fast(labels, blocks, size_cache)
        current_n -= 1

    # Repair fragmentation (if any) after splits/merges
    labels, size_cache = _repair_fragmented_blocks(labels, size_cache)

    # Build horizontal boundary map (top_label, bottom_label) -> coords
    horizontal_boundaries = _build_horizontal_boundaries(labels)

    # Step 4: adjust horizontal boundaries to match std_size
    labels, size_cache = _adjust_boundaries_horizontal(
        labels, size_cache, horizontal_boundaries, mean_size, std_size, rng, std_target_error
    )

    # Final repair pass
    labels, size_cache = _repair_fragmented_blocks(labels, size_cache)

    # Rebuild rectangular block list from labels
    blocks = []
    for lbl in sorted(size_cache.keys()):
        coords = np.where(labels == lbl)
        if len(coords[0]) > 0:
            blocks.append((lbl, slice(int(coords[0].min()), int(coords[0].max()) + 1),
                              slice(int(coords[1].min()), int(coords[1].max()) + 1)))

    return labels, blocks


# ----- helpers (kept equivalent to original, but consolidated) -----
def _build_horizontal_boundaries(labels: np.ndarray) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """Return mapping of (top_label, bottom_label) -> list of (row, col) boundary coords."""
    top = labels[:-1, :]
    bottom = labels[1:, :]
    different = (top != bottom) & (top > 0) & (bottom > 0)
    rows, cols = np.where(different)
    bmap = defaultdict(list)
    for r, c in zip(rows, cols):
        bmap[(int(top[r, c]), int(bottom[r, c]) )].append((int(r), int(c)))
    return dict(bmap)


def _split_block_fast(labels, blocks, idx, size_cache):
    """Split a block roughly in half preferring vertical splits to keep stripes."""
    label = blocks[idx][0]
    rows, cols = np.where(labels == label)
    h = int(rows.max() - rows.min() + 1)
    w = int(cols.max() - cols.min() + 1)
    new_label = int(max(size_cache.keys()) + 1)

    if w >= h:
        mid_col = int(cols.min() + w // 2)
        mask = (labels == label) & (np.arange(labels.shape[1])[None, :] >= mid_col)
    else:
        mid_row = int(rows.min() + h // 2)
        mask = (labels == label) & (np.arange(labels.shape[0])[:, None] >= mid_row)

    labels[mask] = new_label
    size_cache[new_label] = int(np.sum(mask))
    size_cache[label] -= size_cache[new_label]

    # Update blocks list: update current idx and append new block
    rows1, cols1 = np.where(labels == label)
    rows2, cols2 = np.where(labels == new_label)
    if len(rows1) > 0:
        blocks[idx] = (label, slice(int(rows1.min()), int(rows1.max()) + 1),
                          slice(int(cols1.min()), int(cols1.max()) + 1))
    if len(rows2) > 0:
        blocks.append((new_label, slice(int(rows2.min()), int(rows2.max()) + 1),
                            slice(int(cols2.min()), int(cols2.max()) + 1)))
    return labels, blocks, size_cache


def _build_adjacency_fast(labels: np.ndarray) -> Dict[int, Set[int]]:
    """Fast adjacency graph between labeled blocks."""
    adjacency = defaultdict(set)
    left = labels[:, :-1]; right = labels[:, 1:]
    h_mask = (left != right) & (left > 0) & (right > 0)
    if np.any(h_mask):
        pairs = np.column_stack([left[h_mask], right[h_mask]])
        for a, b in pairs:
            adjacency[int(a)].add(int(b)); adjacency[int(b)].add(int(a))
    top = labels[:-1, :]; bottom = labels[1:, :]
    v_mask = (top != bottom) & (top > 0) & (bottom > 0)
    if np.any(v_mask):
        pairs = np.column_stack([top[v_mask], bottom[v_mask]])
        for a, b in pairs:
            adjacency[int(a)].add(int(b)); adjacency[int(b)].add(int(a))
    return dict(adjacency)


def _merge_smallest_blocks_fast(labels, blocks, size_cache):
    """Merge the smallest block into a neighbor (choose smallest adjacent neighbor)."""
    sizes = [size_cache[b[0]] for b in blocks]
    smallest_idx = int(np.argmin(sizes))
    smallest_label = blocks[smallest_idx][0]

    adjacency = _build_adjacency_fast(labels)
    adjacent = list(adjacency.get(smallest_label, []))
    if not adjacent:
        # fallback: merge into next smallest
        if len(sizes) > 1:
            merge_label = blocks[np.argpartition(sizes, 1)[1]][0]
        else:
            return labels, blocks, size_cache
    else:
        adj_sizes = [size_cache[int(l)] for l in adjacent]
        merge_label = adjacent[int(np.argmin(adj_sizes))]

    labels[labels == smallest_label] = merge_label
    size_cache[int(merge_label)] += size_cache[int(smallest_label)]
    del size_cache[int(smallest_label)]
    blocks.pop(smallest_idx)

    # update merged block bounding box
    merge_idx = next(i for i, b in enumerate(blocks) if b[0] == merge_label)
    rows, cols = np.where(labels == merge_label)
    if len(rows) > 0:
        blocks[merge_idx] = (merge_label, slice(int(rows.min()), int(rows.max()) + 1),
                                 slice(int(cols.min()), int(cols.max()) + 1))
    return labels, blocks, size_cache


def _repair_fragmented_blocks(labels: np.ndarray, size_cache: Dict[int, int]):
    """
    Detect and repair fragmented labeled blocks (connected components > 1).
    Small fragments are merged to adjacent blocks using binary dilation to
    find candidates (keeps main, merges small islands).
    """
    from scipy.ndimage import binary_dilation
    unique_labels = list(size_cache.keys())

    for block_label in list(unique_labels):
        if block_label not in size_cache:
            continue
        mask = (labels == block_label).astype(int)
        comps, ncomp = connected_components(mask)
        if ncomp > 1:
            # keep largest comp, merge the rest
            comp_sizes = [(comp_id, int(np.sum(comps == comp_id))) for comp_id in range(1, ncomp + 1)]
            comp_sizes.sort(key=lambda x: x[1], reverse=True)
            # merge all but the largest
            for comp_id, sz in comp_sizes[1:]:
                frag_mask = (comps == comp_id)
                dil = binary_dilation(frag_mask)
                neighbor_region = dil & (~frag_mask)
                adj_labels = list(np.unique(labels[neighbor_region]))
                adj_labels = [int(l) for l in adj_labels if l != block_label and l > 0]
                if not adj_labels:
                    continue
                # prefer labels still in cache
                valid = [l for l in adj_labels if l in size_cache]
                merge_target = valid[0] if valid else adj_labels[0]
                labels[frag_mask] = merge_target
                if merge_target in size_cache:
                    size_cache[merge_target] += sz
                if block_label in size_cache:
                    size_cache[block_label] -= sz

    # remove zero-sized entries
    to_del = [lbl for lbl, s in list(size_cache.items()) if s <= 0]
    for lbl in to_del:
        size_cache.pop(lbl, None)
    return labels, size_cache


def _adjust_boundaries_horizontal(
    labels: np.ndarray,
    size_cache: Dict[int, int],
    horizontal_boundaries: Dict[Tuple[int, int], List[Tuple[int, int]]],
    mean_size: float,
    std_size: float,
    rng: np.random.RandomState,
    std_target_error: float,
    max_iterations: int = 10000
):
    """
    Iteratively transfer contiguous horizontal segments across stripe
    boundaries until the observed std of block sizes approximates std_size.
    This preserves connectivity and avoids fragmenting blocks by testing
    connectivity after tentative transfers.
    """
    if not horizontal_boundaries:
        return labels, size_cache

    boundary_pairs = list(horizontal_boundaries.keys())
    M, N = labels.shape
    consecutive_failures = 0
    max_consecutive_failures = 50

    unique_labels = list(size_cache.keys())

    for iteration in range(max_iterations):
        sizes = np.array([size_cache[lbl] for lbl in unique_labels if lbl in size_cache])
        if sizes.size == 0:
            break
        current_std = float(sizes.std())
        if abs(current_std - std_size) < std_target_error:
            break
        if consecutive_failures > max_consecutive_failures:
            break
        if not boundary_pairs:
            break

        pair_idx = rng.randint(0, len(boundary_pairs))
        top_label, bottom_label = boundary_pairs[pair_idx]
        if top_label not in size_cache or bottom_label not in size_cache:
            if iteration % 100 == 0:
                horizontal_boundaries = _build_horizontal_boundaries(labels)
                boundary_pairs = list(horizontal_boundaries.keys())
            continue

        top_size = size_cache[top_label]; bottom_size = size_cache[bottom_label]

        # choose donor/receiver to move variance in desired direction
        if current_std > std_size:
            donor = top_label if top_size > bottom_size else bottom_label
            receiver = bottom_label if donor == top_label else top_label
        else:
            donor = top_label if top_size < bottom_size else bottom_label
            receiver = bottom_label if donor == top_label else top_label

        if size_cache[donor] <= 2:
            continue

        key = (top_label, bottom_label)
        if key not in horizontal_boundaries or len(horizontal_boundaries[key]) == 0:
            continue
        coords = horizontal_boundaries[key]

        # group by row for contiguous transfers
        rows_dict = defaultdict(list)
        for r, c in coords:
            rows_dict[r].append(c)
        if not rows_dict:
            continue

        # pick a row and contiguous segment from that row
        chosen_row = rng.choice(list(rows_dict.keys()))
        cols = sorted(rows_dict[chosen_row])
        # find contiguous segments
        segments = []
        if cols:
            seg_start = cols[0]; seg_end = cols[0]
            for c in cols[1:]:
                if c == seg_end + 1:
                    seg_end = c
                else:
                    segments.append((seg_start, seg_end + 1)); seg_start = c; seg_end = c
            segments.append((seg_start, seg_end + 1))

        if not segments:
            continue

        seg_idx = rng.randint(0, len(segments))
        s0, s1 = segments[seg_idx]
        seg_len = s1 - s0
        if seg_len > 5:
            s1 = s0 + rng.randint(1, min(6, seg_len))

        if donor == top_label:
            transfer_row = chosen_row
            cells = [(transfer_row, c) for c in range(s0, s1) if labels[transfer_row, c] == top_label]
        else:
            transfer_row = chosen_row + 1
            if transfer_row >= M:
                continue
            cells = [(transfer_row, c) for c in range(s0, s1) if labels[transfer_row, c] == bottom_label]

        if not cells:
            continue

        # tentative transfer: apply then test donor connectivity
        backup = {}
        for r, c in cells:
            backup[(r, c)] = labels[r, c]
            labels[r, c] = receiver

        donor_mask = (labels == donor).astype(int)
        if donor_mask.sum() > 0:
            _, num_comp = connected_components(donor_mask)
            connected_ok = (num_comp == 1)
        else:
            connected_ok = True

        if not connected_ok:
            # revert
            for (r, c), val in backup.items():
                labels[r, c] = val
            consecutive_failures += 1
            continue

        # accept transfer and update sizes
        transferred = len(cells)
        if donor == top_label:
            size_cache[top_label] -= transferred
            size_cache[bottom_label] += transferred
        else:
            size_cache[bottom_label] -= transferred
            size_cache[top_label] += transferred

        consecutive_failures = 0
        if iteration % 100 == 0:
            horizontal_boundaries = _build_horizontal_boundaries(labels)
            boundary_pairs = list(horizontal_boundaries.keys())
            unique_labels = list(size_cache.keys())

    return labels, size_cache


# ----- simple utility functions used elsewhere in the module -----
def analyze_partition(labels_or_tuple, blocks=None) -> dict:
    if blocks is None:
        labels, blocks = labels_or_tuple
    else:
        labels = labels_or_tuple
    sizes = np.bincount(labels.ravel())[1:]
    return {'mean': sizes.mean(), 'std': sizes.std(), 'min': sizes.min(), 'max': sizes.max(),
            'n_blocks': len(sizes), 'total_area': sizes.sum()}


def check_partition_integrity(labels: np.ndarray) -> dict:
    """Quick integrity and fragmentation check; prints a short summary."""
    M, N = labels.shape
    zero_count = int(np.sum(labels == 0))
    total_labeled = int(np.sum(labels > 0))
    unique = np.unique(labels[labels > 0])
    fragmented = []
    for lbl in unique:
        mask = (labels == lbl).astype(int)
        _, num_comp = connected_components(mask)
        if num_comp > 1:
            fragmented.append((int(lbl), int(num_comp)))
    print("=" * 60)
    print(f"Coverage: {total_labeled}/{M*N} cells labeled ({100*total_labeled/(M*N):.1f}%)")
    print(f"Unlabeled: {zero_count} cells")
    print(f"Blocks: {len(unique)} total")
    print(f"Fragmented: {len(fragmented)} blocks")
    print("=" * 60)
    return {'is_valid': zero_count == 0 and len(fragmented) == 0, 'zero_count': zero_count, 'fragmented_blocks': fragmented}


def verify_partition_connectivity(labels: np.ndarray, verbose: bool = True) -> dict:
    """Return connectivity diagnostics for each block and optionally print details."""
    unique = np.unique(labels[labels > 0])
    frag_blocks = []
    frag_details = {}
    for lbl in unique:
        comp_mask, ncomp = connected_components((labels == lbl).astype(int))
        if ncomp > 1:
            comp_sizes = [int(np.sum(comp_mask == i)) for i in range(1, ncomp + 1)]
            frag_blocks.append((int(lbl), int(ncomp)))
            frag_details[int(lbl)] = comp_sizes
            if verbose:
                print(f"Block {lbl} fragmented into {ncomp} pieces: {comp_sizes}")
    if verbose:
        if not frag_blocks:
            print(f"✓ All {len(unique)} blocks are connected")
        else:
            print(f"✗ Found {len(frag_blocks)} fragmented blocks")
    return {'is_valid': len(frag_blocks) == 0, 'total_blocks': len(unique), 'fragmented_blocks': frag_blocks, 'fragmentation_details': frag_details}


def summarize_neighborhoods(city: np.ndarray, blocks: List[Tuple]) -> np.ndarray:
    """Count red/blue agents inside each block; expects city grid with 1/2 labels."""
    mask_r = (city == 1); mask_b = (city == 2)
    counts = np.zeros((len(blocks), 2), dtype=int)
    for i, (_lbl, rs, cs) in enumerate(blocks):
        counts[i, 0] = int(mask_r[rs, cs].sum())
        counts[i, 1] = int(mask_b[rs, cs].sum())
    return counts

def _draw_label_boundaries(ax, labels, linewidth=1, color='black'):
    """
    Efficiently draw boundaries between regions with different labels.
    
    Uses vectorized numpy operations to find all boundary edges,
    then draws them using LineCollection for efficiency.
    """
    nrows, ncols = labels.shape
    segments = []

    # Horizontal edges: between row i and row i+1
    if nrows > 1:
        h_diff = labels[:-1, :] != labels[1:, :]
        h_rows, h_cols = np.where(h_diff)
        if len(h_rows) > 0:
            h_segments = np.stack([
                np.column_stack([h_cols - 0.5, h_rows + 0.5]),
                np.column_stack([h_cols + 0.5, h_rows + 0.5])
            ], axis=1)
            segments.append(h_segments)

    # Vertical edges: between col j and col j+1
    if ncols > 1:
        v_diff = labels[:, :-1] != labels[:, 1:]
        v_rows, v_cols = np.where(v_diff)
        if len(v_rows) > 0:
            v_segments = np.stack([
                np.column_stack([v_cols + 0.5, v_rows - 0.5]),
                np.column_stack([v_cols + 0.5, v_rows + 0.5])
            ], axis=1)
            segments.append(v_segments)

    if segments:
        all_segments = np.vstack(segments)
        lc = LineCollection(all_segments, colors=color, linewidths=linewidth)
        ax.add_collection(lc)

def plot_city_with_boundaries(
    city: np.ndarray,
    city_parse: Tuple[np.ndarray, List[Tuple]],
    colors_map: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 12),
    dpi: int = 100,
    linewidth: float = 2,
    zoom_fraction: float = 1.0
) -> Tuple:
    """
    Plot a city grid (values 0,1,2) with neighborhood boundaries overlaid.
    """
    full_labels, blocks = city_parse
    if zoom_fraction < 1.0:
        max_row = int(city.shape[0] * zoom_fraction)
        max_col = int(city.shape[1] * zoom_fraction)
        city = city[:max_row, :max_col]
        labels = full_labels[:max_row, :max_col]
    else:
        labels = full_labels

    if colors_map is None:
        colors_map = ['#ffffff', '#D0021B', '#4A90E2']
    cmap = ListedColormap(colors_map)
    norm = BoundaryNorm([0, 1, 2, 3], cmap.N)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(city, cmap=cmap, norm=norm, interpolation='nearest')
    _draw_label_boundaries(ax, labels, linewidth=linewidth, color='black')

    ax.set_xlim(-0.5, city.shape[1] - 0.5)
    ax.set_ylim(city.shape[0] - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    return fig, ax

def plot_metric_comparison(
    neighborhood_data: Dict[str, List[np.ndarray]],
    metric_type: str = "D",
    target_composition: float = 0.5,
    show_adjusted: bool = False,
    overlay_adjusted: bool = False,
    focal_group_label: str = "Red",
    save_fig = False,
    dpi: int = 300,
    figsize: Tuple[int, int] = (12, 7),
    verbose: bool = False,
    ax=None,
    show_annotations: bool = True,
    reduce_x_format: bool = False,
    alpha_unadjusted: float = 0.6,
    alpha_adjusted: float = 0.6,
    show_annotations_unadjusted: bool = None,
    show_annotations_adjusted: bool = None
) -> None:
    """
    Visualize and compare adjusted vs unadjusted segregation metrics across datasets.
    
    Parameters:
    -----------
    show_annotations : bool, default=True
        If True, display mean/std annotations and legend on the plot.
        If False, hide all annotations for cleaner visualization.
    
    reduce_x_format : bool, default=False
        If True, x-axis labels show only numbers (e.g., "5", "10", "15").
        If False, x-axis labels include percentage and group (e.g., "5% Red").
    
    alpha_unadjusted : float, default=0.6
        Transparency level for unadjusted boxes (0.0 = fully transparent, 1.0 = opaque).
        Whiskers, caps, and medians scale proportionally.
    
    alpha_adjusted : float, default=0.6
        Transparency level for adjusted boxes (0.0 = fully transparent, 1.0 = opaque).
        Whiskers, caps, and medians scale proportionally.
    
    show_annotations_unadjusted : bool, default=None
        If True, show annotations for unadjusted values.
        If False, hide annotations for unadjusted values.
        If None, uses the value of show_annotations.
    
    show_annotations_adjusted : bool, default=None
        If True, show annotations for adjusted values.
        If False, hide annotations for adjusted values.
        If None, uses the value of show_annotations.
    
    save_fig : bool or str, default=False
        If False, don't save the figure.
        If True, save with auto-generated filename.
        If string, save with that filename.
    """
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    if show_annotations_unadjusted is None:
        show_annotations_unadjusted = show_annotations
    if show_annotations_adjusted is None:
        show_annotations_adjusted = show_annotations

    if callable(metric_type):
        metric_type = metric_type.__name__

    available_datasets = sorted(
        neighborhood_data.keys(),
        key=lambda x: float(x) if x.replace('.', '').replace('-', '').isdigit() else x
    )

    if not available_datasets:
        if verbose:
            print("No datasets found in neighborhood_data")
        return

    metric_map = {
        "D": {"func": D, "name": "Dissimilarity"},
        "Ent": {"func": Ent, "name": "Entropy"},
        "S": {"func": S, "name": "Separation"},
        "G": {"func": G, "name": "Gini"},
        "R": {"func": R, "name": "Hutchens R"},
        "Iso": {"func": Iso, "name": "Isolation"},
        "Seg": {"func": Seg, "name": "Gorard Segregation (Blue)"},
        "Seg_Maj": {"func": Seg_Maj, "name": "Gorard Segregation (Red)"}
    }

    if metric_type not in metric_map:
        if verbose:
            print(f"Invalid metric type '{metric_type}'. Choose from: {list(metric_map.keys())}")
        return

    metric_info = metric_map[metric_type]
    metric_func = metric_info["func"]
    metric_name = metric_info["name"]

    if show_adjusted and 'Index' not in globals():
        if verbose:
            print("Warning: Index function not available. Disabling adjusted metrics.")
        show_adjusted = False

    global_colors = globals().get("colors", ["#0173b2", "#de8f05"])
    base_blue = mcolors.to_rgb(global_colors[0])
    base_gold = mcolors.to_rgb(global_colors[1])
    edge_color_unadjusted = mcolors.to_hex([c * 0.4 for c in base_blue])
    edge_color_adjusted = mcolors.to_hex([c * 0.5 for c in base_gold])

    line_alpha_unadjusted = min(1.0, alpha_unadjusted / 0.6)
    line_alpha_adjusted = min(1.0, alpha_adjusted / 0.6)

    regular_values, adjusted_values, statistics = {}, {}, []

    for dataset_key in available_datasets:
        neighborhoods = neighborhood_data[dataset_key]
        reg_vals = [metric_func(n) for n in neighborhoods]
        regular_values[dataset_key] = reg_vals
        statistics.append({
            "mean": np.mean(reg_vals),
            "std": np.std(reg_vals),
            "dataset": dataset_key,
            "is_adjusted": False
        })

        if show_adjusted:
            adj_vals = [Index(n, metric_type, target_composition) for n in neighborhoods]
            adjusted_values[dataset_key] = adj_vals
            statistics.append({
                "mean": np.mean(adj_vals),
                "std": np.std(adj_vals),
                "dataset": dataset_key,
                "is_adjusted": True
            })

    created_fig = ax is None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    n_datasets = len(available_datasets)
    spacing = 2.0

    box_positions, all_values, box_info = [], [], []

    for i, dataset_key in enumerate(available_datasets):
        base_pos = i * spacing + 1.0

        box_positions.append(base_pos - 0.4 if show_adjusted and not overlay_adjusted else base_pos)
        all_values.append(regular_values[dataset_key])
        box_info.append({"dataset": dataset_key, "type": "regular"})

        if show_adjusted and dataset_key in adjusted_values:
            pos = base_pos + (0.4 if not overlay_adjusted else 0)
            box_positions.append(pos)
            all_values.append(adjusted_values[dataset_key])
            box_info.append({"dataset": dataset_key, "type": "adjusted"})

    tick_positions = [i * spacing + 1.0 for i in range(n_datasets)]
    
    if reduce_x_format:
        tick_labels = [f"{key}" for key in available_datasets]
    else:
        tick_labels = [f"{key}% {focal_group_label}" for key in available_datasets]

    bp = ax.boxplot(
        all_values, positions=box_positions, patch_artist=True,
        showmeans=False, widths=0.6
    )

    for i, (box, info) in enumerate(zip(bp["boxes"], box_info)):
        is_regular = info["type"] == "regular"
        
        face_color = (
            mcolors.to_rgba(global_colors[0], alpha=alpha_unadjusted)
            if is_regular
            else mcolors.to_rgba(global_colors[1], alpha=alpha_adjusted)
        )
        
        base_edge_color = edge_color_unadjusted if is_regular else edge_color_adjusted
        line_alpha = line_alpha_unadjusted if is_regular else line_alpha_adjusted
        edge_color_with_alpha = mcolors.to_rgba(base_edge_color, alpha=line_alpha)
        
        box.set(facecolor=face_color, edgecolor=edge_color_with_alpha, linewidth=2.0)

    for i, info in enumerate(box_info):
        edge_color = (
            edge_color_unadjusted if info["type"] == "regular"
            else edge_color_adjusted
        )
        line_alpha = (
            line_alpha_unadjusted if info["type"] == "regular"
            else line_alpha_adjusted
        )

        for j in range(2):
            bp["whiskers"][i*2 + j].set(color=edge_color, linewidth=2.0, alpha=line_alpha)
            bp["caps"][i*2 + j].set(color=edge_color, linewidth=2.0, alpha=line_alpha)

        bp["medians"][i].set(color=edge_color, linewidth=2.5, alpha=line_alpha)
        bp["fliers"][i].set(markeredgecolor=edge_color, markersize=6, alpha=min(0.6, line_alpha * 0.6))

    if show_annotations_unadjusted or show_annotations_adjusted:
        annotation_positions = []
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        for stats in statistics:
            dataset_key = stats["dataset"]
            is_adjusted = stats["is_adjusted"]

            if is_adjusted and not show_annotations_adjusted:
                continue
            if not is_adjusted and not show_annotations_unadjusted:
                continue

            box_pos_idx = None
            dataset_idx = None
            for j, info in enumerate(box_info):
                if (info["dataset"] == dataset_key and
                    ((info["type"] == "adjusted") == is_adjusted)):
                    box_pos_idx = j
                    dataset_idx = available_datasets.index(dataset_key)
                    break

            if box_pos_idx is not None:
                x_pos = box_positions[box_pos_idx]
                y_pos = stats["mean"]

                if overlay_adjusted:
                    if dataset_idx == 0:
                        x_offset, ha_align = 0.5, "left"
                    elif dataset_idx == n_datasets - 1:
                        x_offset, ha_align = -0.5, "right"
                    else:
                        if is_adjusted:
                            x_offset, ha_align = 0.5, "left"
                        else:
                            x_offset, ha_align = -0.5, "right"
                else:
                    middle_left = n_datasets // 2 - 1
                    middle_right = n_datasets // 2

                    if dataset_idx == middle_left or dataset_idx == middle_right:
                        if is_adjusted:
                            x_offset, ha_align = -0.5, "right"
                        else:
                            x_offset, ha_align = 0.5, "left"
                    elif dataset_idx == 0:
                        x_offset, ha_align = 0.5, "left"
                    elif dataset_idx == n_datasets - 1:
                        x_offset, ha_align = -0.5, "right"
                    else:
                        if dataset_idx < middle_left:
                            x_offset, ha_align = -0.5, "right"
                        else:
                            x_offset, ha_align = 0.5, "left"

                annotation_x = x_pos + x_offset
                annotation_y = y_pos

                for prev_x, prev_y in annotation_positions:
                    if (abs(prev_x - annotation_x) < 1.2 and
                        abs(prev_y - annotation_y) < y_range * 0.08):
                        if annotation_y > prev_y:
                            annotation_y = prev_y + y_range * 0.1
                        else:
                            annotation_y = prev_y - y_range * 0.1

                annotation_y = np.clip(
                    annotation_y, y_min + y_range * 0.05, y_max - y_range * 0.05
                )

                annotation_positions.append((annotation_x, annotation_y))

                ax.annotate(
                    f"μ={stats['mean']:.4f}\nσ={stats['std']:.4f}",
                    xy=(x_pos, y_pos),
                    xytext=(annotation_x, annotation_y),
                    fontsize=12, ha=ha_align, va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.9, edgecolor="lightgray", linewidth=1.0),
                    arrowprops=dict(arrowstyle="-", color="gray", alpha=0.7,
                                    connectionstyle="arc3,rad=0.1", linewidth=1.2)
                )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#333333")
        ax.spines[spine].set_linewidth(1.2)

    ax.set_title(f"{metric_name} Index", fontsize=18, pad=28)

    if show_adjusted:
        target_pct = int(target_composition * 100)
        ax.text(
            0.5, 1.025, f"(Target = {target_pct}% {focal_group_label})",
            transform=ax.transAxes, fontsize=13, ha="center",
            style="italic", color="#666666"
        )

    ax.set_ylabel(f"{metric_name} Value", fontsize=14)

    if show_adjusted:
        legend = [
            Patch(facecolor=mcolors.to_rgba(global_colors[0], alpha=alpha_unadjusted),
                  edgecolor=edge_color_unadjusted, label="Unadjusted"),
            Patch(facecolor=mcolors.to_rgba(global_colors[1], alpha=alpha_adjusted),
                  edgecolor=edge_color_adjusted, label="Adjusted"),
        ]
        ax.legend(handles=legend, loc="upper right", frameon=True,
                  framealpha=0.9, edgecolor="gray", fontsize=12)

    if show_annotations_unadjusted or show_annotations_adjusted:
        ax.text(
            0.02, 0.02, "μ = mean, σ = standard deviation",
            transform=ax.transAxes, fontsize=11, alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.8, edgecolor="lightgray")
        )

    if created_fig:
        plt.tight_layout()

        if save_fig:
            if isinstance(save_fig, str):
                fname = save_fig
            else:
                mode = "overlay" if overlay_adjusted else "sidebyside"
                adj = "adjusted" if show_adjusted else "regular"
                fname = f"{metric_type}_{adj}_{mode}_target{int(target_composition*100)}.png"
            
            plt.savefig(fname, dpi=dpi, bbox_inches="tight", facecolor="white")
            if verbose:
                print(f"Saved figure as {fname}")

        plt.show()
    
    return ax



def plot_segregation_dynamics(
    datasets: np.ndarray,
    city_parse: Tuple[np.ndarray, List[Tuple]],
    metric: Union[str, callable] = 'D',
    labels: Optional[List[str]] = None,
    colors_list: Optional[List[str]] = None,
    use_adjusted: bool = False,
    target_composition: float = 0.5,
    correction_method: str = 'info',
    show_stats: bool = True,
    use_snapshots_after: int = 0,
    figsize: Tuple[int, int] = (14, 8),
    save_fig: bool = False,
    filename: Optional[str] = None,
    dpi: int = 300,
    alpha: float = 0.8,
    linewidth: float = 2.5,
    show_legend: bool = True,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    print_stats: bool = True
) -> Tuple:
    """
    Plot the evolution of a segregation index over time for one or more simulations.

    Parameters
    ----------
    datasets : np.ndarray
        Array of simulation snapshots with shape (n_runs, n_snapshots, H, W).
    city_parse : tuple
        Output of stripe_partition: (labels, blocks), used to aggregate cells into neighborhoods.
    metric : str or callable, default 'D'
        Segregation metric ('D', 'S', 'Iso', 'Ent', 'R', 'G', 'Seg', 'Seg_Maj') or a custom callable.
    use_adjusted : bool, default False
        If True, compute compositional-adjusted indices using Index().
    target_composition : float, default 0.5
        Target group composition for adjusted indices.
    correction_method : str, default 'info'
        Correction method passed to Index() ('info' or 'n_terms').
    use_snapshots_after : int, default 0
        Ignore early snapshots when computing summary statistics (burn-in).
    save_fig : bool, default False
        If True, save the figure to disk.
    filename : str, optional
        Output filename if save_fig is True.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """

    metric_map = {
        D: 'Dissimilarity',
        S: 'Separation',
        Iso: 'Isolation',
        Ent: 'Entropy',
        R: 'Hutchens R',
        G: 'Gini',
        Seg: 'Gorard Segregation (Red)',
        Seg_Maj: 'Gorard Segregation (Blue)'
    }

    string_to_func = {
        'D': D,
        'S': S,
        'Iso': Iso,
        'Ent': Ent,
        'R': R,
        'G': G,
        'Seg': Seg,
        'Seg_Maj': Seg_Maj
    }

    # Resolve metric
    if callable(metric):
        metric_func = metric
        metric_name = metric_map.get(metric, metric.__name__)
        metric_key = metric.__name__
    else:
        if metric not in string_to_func:
            raise ValueError(f"Unknown metric: '{metric}'. Choose from {list(string_to_func.keys())}")
        metric_func = string_to_func[metric]
        metric_name = metric_map[metric_func]
        metric_key = metric

    # Colors
    if colors_list is None:
        colors_list = globals().get('colors', plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # Labels
    n_datasets = len(datasets)
    if labels is None:
        labels = [f"Run {i+1}" for i in range(n_datasets)]
    elif len(labels) != n_datasets:
        raise ValueError(f"Number of labels ({len(labels)}) doesn't match datasets ({n_datasets})")

    fig, ax = plt.subplots(figsize=figsize)
    stats_list = []

    # ---- Main loop over simulations ----
    for i in range(n_datasets):
        color = colors_list[i % len(colors_list)]

        # Aggregate cell-level grids to neighborhood counts
        temp = np.array([summarize_neighborhoods(dat, city_parse[1]) for dat in datasets[i]])

        # Compute segregation index time series
        if use_adjusted:
            vals = np.array([
                Index(dat, metric_key, pi_tar=target_composition, method=correction_method)
                for dat in temp
            ])
        else:
            vals = np.array([metric_func(dat) for dat in temp])

        # Burn-in trimming for statistics
        if use_snapshots_after < len(vals):
            vals_for_stats = vals[use_snapshots_after:]
        else:
            vals_for_stats = vals
            if print_stats:
                print(f"Warning: use_snapshots_after ({use_snapshots_after}) >= data length ({len(vals)}); using all snapshots")

        mean_val = np.mean(vals_for_stats)
        std_val = np.std(vals_for_stats)
        stats_list.append((mean_val, std_val, color, labels[i], vals))

        if print_stats:
            suffix = f" (snapshots {use_snapshots_after}+)" if use_snapshots_after > 0 else ""
            print(f"{labels[i]}: μ={mean_val:.4f}, σ={std_val:.4f}{suffix}")

        steps_arr = np.arange(len(vals))
        ax.plot(steps_arr, vals, color=color, alpha=alpha, linewidth=linewidth, label=labels[i], zorder=3)

    # ---- Annotated summary stats ----
    if show_stats:
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        sorted_stats = sorted(stats_list, key=lambda x: x[0])
        y_positions = np.linspace(y_min + 0.1*y_range, y_max - 0.1*y_range, n_datasets)

        for idx, (mean_val, std_val, color, label, vals) in enumerate(sorted_stats):
            x_pos = len(vals) - 1
            stats_text = f"{label}\nμ={mean_val:.3f}\nσ={std_val:.3f}"

            ax.annotate(
                stats_text,
                xy=(x_pos, mean_val),
                xytext=(x_pos + len(vals)*0.08, y_positions[idx]),
                fontsize=9,
                ha='left',
                va='center',
                color=color,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          alpha=0.95, edgecolor=color, linewidth=1.5),
                arrowprops=dict(
                    arrowstyle='-',
                    color=color,
                    alpha=0.6,
                    linewidth=1.2,
                    shrinkA=0,
                    shrinkB=0
                )
            )

    # ---- Formatting ----
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8, zorder=0)

    if xlabel is None:
        xlabel = 'Snapshot'
    ax.set_xlabel(xlabel, fontsize=14, labelpad=10)

    if ylabel is None:
        ylabel = f'{metric_name} Index'
    ax.set_ylabel(ylabel, fontsize=14, labelpad=10)

    if title is None:
        if use_adjusted:
            title = f'{metric_name} Index Over Time (Adjusted, Target Composition = {target_composition})'
        else:
            title = f'{metric_name} Index Over Time'
    ax.set_title(title, fontsize=16, pad=15, fontweight='bold')

    if show_legend and not show_stats:
        ax.legend(loc='best', frameon=True, framealpha=0.9,
                  edgecolor='gray', fontsize=11, ncol=1 if n_datasets <= 6 else 2)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#333333')
        ax.spines[spine].set_linewidth(1.2)

    ax.tick_params(axis='both', which='major', labelsize=11,
                   colors='#333333', width=1.2)

    plt.tight_layout()

    # ---- Save figure ----
    if save_fig:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            adj_suffix = f"_adj{int(target_composition*100)}_{correction_method}" if use_adjusted else ""
            filename = f"segregation_dynamics_{metric_key}{adj_suffix}_{timestamp}.png"
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"\nSaved figure as: {filename}")

    plt.show()
    return fig, ax

# -----------------------------
# Module exports
# -----------------------------
__all__ = [
    'D', 'S', 'Iso', 'Ent', 'R', 'G', 'Seg', 'Seg_Maj', 'Avg',
    'Index', 'no_correction', 'info_correction', 'n_terms_correction',
    'SchellingModel',
    'run_simulation_batch', 'load_batch_results', 'load_simulation_by_label',
    'stripe_partition', 'summarize_neighborhoods',
    'plot_city_with_boundaries', 'plot_metric_comparison', 'plot_segregation_dynamics'
]
