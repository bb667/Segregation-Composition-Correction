"""
compositional_correction.py
===========================

Two-group segregation indices with the composition-invariance correction
from Barron et al. (2026), "When All Measures Fail the Same Way".

Quick start
-----------
    import compositional_correction as cc

    # Three accepted input forms — all equivalent:
    cc.dissimilarity(df, "white", "hispanic")    # DataFrame + column names
    cc.dissimilarity(matrix)                      # shape-(n, 2) array
    cc.dissimilarity(vec_a, vec_b)                # two 1D arrays

    # Pick an index by short code or by name (case-insensitive):
    cc.segregation_index(df, "white", "hispanic", index="D")
    cc.segregation_index(df, "white", "hispanic", index="dissimilarity")

    # Compositional correction:
    cc.dissimilarity(df, "white", "hispanic", target=0.50)   # one value
    cc.composition_curve(df, "white", "hispanic", index="D") # whole curve

    # Plot the curve. By default the curve runs from the 10th to the 90th
    # population-weighted percentile of composition — i.e., across the
    # central 80% of the population, where the data is dense enough to
    # support extrapolation. Override with `population_percentile=`, or
    # pass None to plot the full sweep:
    cc.plot_composition_curve(df, "white", "hispanic", index="D")

What the correction does
------------------------
Common evenness indices (D, S, H, R, G) depend systematically on a city's
overall composition mu, not only on how unevenly the two groups are
distributed across neighborhoods. The correction reweights neighborhoods
via exponential tilting (an I-projection) so the reweighted population
composition equals a chosen target mu*. Comparing indices computed at the
same mu* removes the compositional confound.

Two modes:
  - Pointwise. Pass `target=mu*`. A 1-D root-find selects the tilting
    parameter that hits mu*; one number out.
  - Curve. `composition_curve(...)` sweeps the tilting parameter and
    returns (mu, index) pairs along the whole admissible composition
    range. No root-finding.

Available indices
-----------------
    'D'   or 'dissimilarity'
    'S'   or 'separation'             (variance ratio)
    'H'   or 'entropy'                (Theil H)
    'R'   or 'hutchens'               (square-root index)
    'G'   or 'gini'
    'Iso' or 'isolation'              (xPx for the focal group)

Reference
---------
Barron, B., Hall, M., Rich, P., Cohen, I., Arias, T.A. (2026).
When All Measures Fail the Same Way: A Correction to Segregation Trends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq


__all__ = [
    "segregation_index",
    "dissimilarity", "separation", "entropy",
    "hutchens_r", "gini", "isolation",
    "composition_curve",
    "plot_composition_curve",
]


# ---------------------------------------------------------------------------
# Index name resolution: short codes and long names, case-insensitive.
# ---------------------------------------------------------------------------

_INDEX_ALIASES = {
    "d": "D",   "dissimilarity": "D",
    "s": "S",   "separation": "S",      "variance_ratio": "S",
    "h": "H",   "entropy": "H",         "theil": "H",
    "r": "R",   "hutchens": "R",        "hutchens_r": "R",   "square_root": "R",
    "g": "G",   "gini": "G",
    "iso": "Iso", "isolation": "Iso",   "xpx": "Iso",
}

_INDEX_LONG_NAMES = {
    "D": "Dissimilarity",
    "S": "Separation (variance ratio)",
    "H": "Entropy (Theil H)",
    "R": "Hutchens R (square root)",
    "G": "Gini",
    "Iso": "Isolation (xPx)",
}


def _canonical_index(name):
    """Map 'D', 'dissimilarity', 'Dissimilarity', etc. to a canonical code."""
    if not isinstance(name, str):
        raise TypeError(f"Index name must be a string, got {type(name).__name__}.")
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return _INDEX_ALIASES[key]
    except KeyError:
        codes = sorted(set(_INDEX_ALIASES.values()))
        raise ValueError(
            f"Unknown index {name!r}. Valid short codes: {codes}. "
            f"Long names also work (e.g. 'dissimilarity', 'gini', 'isolation')."
        )


# ---------------------------------------------------------------------------
# Input handling: accept any of three data forms.
# ---------------------------------------------------------------------------

def _to_pi_t(data, *args):
    """
    Convert any of the three accepted input forms into (pi, t):
        pi : focal-group share in each neighborhood, in [0, 1]
        t  : pairwise total (group A + group B) in each neighborhood

    Form 1 — DataFrame + two column names:
        _to_pi_t(df, "col_a", "col_b")

    Form 2 — 2D array of shape (n, 2), column 0 = group A, column 1 = group B:
        _to_pi_t(matrix)

    Form 3 — two 1D arrays of equal length, focal first:
        _to_pi_t(vec_a, vec_b)

    Empty neighborhoods (where group A + group B == 0) are dropped.
    """
    if isinstance(data, pd.DataFrame):
        if len(args) != 2:
            raise ValueError(
                "DataFrame input requires two column names: "
                "f(df, 'col_a', 'col_b')."
            )
        col_a, col_b = args
        a = np.asarray(data[col_a], dtype=float)
        b = np.asarray(data[col_b], dtype=float)
    else:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 2:
            if len(args) != 0:
                raise ValueError(
                    "Matrix input takes no extra arguments; column 0 is "
                    "group A and column 1 is group B."
                )
            if arr.shape[1] != 2:
                raise ValueError(
                    f"Matrix input must have shape (n, 2); got {arr.shape}."
                )
            a, b = arr[:, 0], arr[:, 1]
        elif arr.ndim == 1:
            if len(args) != 1:
                raise ValueError(
                    "Vector input requires a second 1D array: f(vec_a, vec_b)."
                )
            b = np.asarray(args[0], dtype=float)
            if b.ndim != 1 or b.shape != arr.shape:
                raise ValueError(
                    f"Group vectors must be 1D and the same length; "
                    f"got {arr.shape} and {b.shape}."
                )
            a = arr
        else:
            raise ValueError(
                f"Unsupported array shape {arr.shape}. Expected a 1D array "
                f"(with a second 1D array for group B) or a 2D array of "
                f"shape (n, 2)."
            )

    if (a < 0).any() or (b < 0).any():
        raise ValueError("Population counts must be non-negative.")

    t = a + b
    keep = t > 0
    if not keep.any():
        raise ValueError("No neighborhoods with non-zero total of the two groups.")

    return a[keep] / t[keep], t[keep]


# ---------------------------------------------------------------------------
# Core math: tilting, root finding, and the index formulas.
# ---------------------------------------------------------------------------

def _weights(pi, v):
    """exp(v * pi), shifted for numerical stability. Overall scale cancels."""
    log_w = v * pi
    log_w -= log_w.max()
    return np.exp(log_w)


def _tilted_mean(pi, t, v):
    """Population-weighted mean of pi after tilting weights by exp(v*pi)."""
    w = _weights(pi, v)
    return (t * w * pi).sum() / (t * w).sum()


def _solve_v(pi, t, target, v_bracket=(-50.0, 50.0)):
    """
    Find the tilting parameter v whose induced composition equals `target`.
    The map v -> tilted_mean(v) is strictly monotone, so Brent's method is
    well-posed.
    """
    lo, hi = float(pi.min()), float(pi.max())
    if not (lo < target < hi):
        raise ValueError(
            f"Target composition {target:.4f} is outside the achievable "
            f"range [{lo:.4f}, {hi:.4f}] of observed neighborhood shares. "
            f"The correction can only redistribute mass within the support "
            f"of pi."
        )
    return brentq(lambda v: _tilted_mean(pi, t, v) - target,
                  v_bracket[0], v_bracket[1])


def _compute(pi, t, mu, code):
    """
    Evaluate one segregation index from (pi, t) at composition mu.

    For the standard (unadjusted) value, pass raw t and mu = average(pi, t).
    For the corrected value, pass t * exp(v*pi) and mu = target. Everything
    else is the same textbook formula.
    """
    if not (0.0 < mu < 1.0):
        return 0.0
    T = t.sum()

    if code == "D":   # Dissimilarity
        return (t * np.abs(pi - mu)).sum() / (2.0 * T * mu * (1.0 - mu))

    if code == "S":   # Separation / variance ratio
        return (t * (pi - mu) ** 2).sum() / (T * mu * (1.0 - mu))

    if code == "H":   # Theil entropy
        E = -mu * np.log(mu) - (1.0 - mu) * np.log(1.0 - mu)
        if E <= 0.0:
            return 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            term_a = np.where(pi > 0, pi * np.log(pi / mu), 0.0)
            term_b = np.where(pi < 1,
                              (1.0 - pi) * np.log((1.0 - pi) / (1.0 - mu)),
                              0.0)
        return (t * (term_a + term_b)).sum() / (T * E)

    if code == "R":   # Hutchens R (square-root)
        return 1.0 - (t * np.sqrt(pi * (1.0 - pi))).sum() / (T * np.sqrt(mu * (1.0 - mu)))

    if code == "G":   # Gini — O(n log n) sorted-cumsum identity
        order = np.argsort(pi)
        pi_s = pi[order]
        t_s = t[order]
        cum_t = np.concatenate(([0.0], np.cumsum(t_s)[:-1]))
        cum_tpi = np.concatenate(([0.0], np.cumsum(t_s * pi_s)[:-1]))
        return (t_s * (pi_s * cum_t - cum_tpi)).sum() / (T ** 2 * mu * (1.0 - mu))

    if code == "Iso":  # Isolation xPx for the focal group
        return (t * pi ** 2).sum() / (T * mu)

    raise ValueError(f"Internal: unknown index code {code!r}.")


# ---------------------------------------------------------------------------
# Public API: one general entry point plus named convenience wrappers.
# ---------------------------------------------------------------------------

def segregation_index(data, *args, index="D", target=None):
    """
    Compute any two-group segregation index, optionally with the
    composition-invariance correction.

    Parameters
    ----------
    data, *args :
        Neighborhood data, in any of three forms (see module docstring):
          DataFrame:    f(df, "col_a", "col_b")
          Matrix:       f(M)             where M.shape == (n, 2)
          Two vectors:  f(vec_a, vec_b)  with focal group first
    index : str, default 'D'
        Which index. Short codes ('D', 'S', 'H', 'R', 'G', 'Iso') or
        long names ('dissimilarity', 'separation', 'entropy', 'hutchens',
        'gini', 'isolation'). Case-insensitive.
    target : float or None, default None
        If None, return the unadjusted index.
        If a float strictly inside (min(pi), max(pi)), apply the
        composition correction: reweight neighborhoods so the population
        composition equals `target`, then compute the index. A 1-D
        root-find solves for the tilting parameter.

    Returns
    -------
    float
    """
    code = _canonical_index(index)
    pi, t = _to_pi_t(data, *args)
    if target is None:
        mu = (t * pi).sum() / t.sum()
        return _compute(pi, t, mu, code)
    v = _solve_v(pi, t, target)
    return _compute(pi, t * _weights(pi, v), target, code)


# Named convenience wrappers — same signature, fixed index. ------------------

def dissimilarity(data, *args, target=None):
    """Dissimilarity index D. See `segregation_index` for input forms."""
    return segregation_index(data, *args, index="D", target=target)


def separation(data, *args, target=None):
    """Separation / variance-ratio index S."""
    return segregation_index(data, *args, index="S", target=target)


def entropy(data, *args, target=None):
    """Theil entropy index H."""
    return segregation_index(data, *args, index="H", target=target)


def hutchens_r(data, *args, target=None):
    """Hutchens R (square-root) index."""
    return segregation_index(data, *args, index="R", target=target)


def gini(data, *args, target=None):
    """Gini index (segregation form)."""
    return segregation_index(data, *args, index="G", target=target)


def isolation(data, *args, target=None):
    """Isolation index xPx for the focal group (passed first)."""
    return segregation_index(data, *args, index="Iso", target=target)


# ---------------------------------------------------------------------------
# Index-as-a-function-of-composition curve (no root finding).
# ---------------------------------------------------------------------------

def composition_curve(data, *args, index="D",
                      n_points=200, v_range=(-30.0, 30.0)):
    """
    Index value as a function of population composition, swept by varying
    the tilting parameter v over `v_range`. No root-finding is performed.

    Parameters
    ----------
    data, *args :
        Neighborhood data in any of the three accepted forms.
    index : str or sequence of str, default 'D'
        Which index to evaluate. Pass a sequence (e.g. ['D', 'H', 'G']) to
        get one column per index in a single sweep.
    n_points : int, default 200
        Number of v samples (and curve points).
    v_range : (float, float), default (-30, 30)
        Range of the tilting parameter. Widen if the returned composition
        range looks truncated relative to [min(pi), max(pi)].

    Returns
    -------
    pandas.DataFrame
        Columns: 'composition' plus one column per index (named by its
        canonical short code, e.g. 'D'). Sorted by composition.

    Note
    ----
    Because v is sampled uniformly, the resulting composition points
    cluster near the data's own mean and thin out toward the extremes.
    For uniform spacing in composition, call `segregation_index` in a
    loop with explicit `target` values.
    """
    codes = ([_canonical_index(index)] if isinstance(index, str)
             else [_canonical_index(s) for s in index])

    pi, t = _to_pi_t(data, *args)
    vs = np.linspace(v_range[0], v_range[1], n_points)

    mus = np.empty(n_points)
    vals = {code: np.empty(n_points) for code in codes}
    for k, v in enumerate(vs):
        tw = t * _weights(pi, v)
        mus[k] = (tw * pi).sum() / tw.sum()
        for code in codes:
            vals[code][k] = _compute(pi, tw, mus[k], code)

    order = np.argsort(mus)
    out = {"composition": mus[order]}
    for code in codes:
        out[code] = vals[code][order]
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Optional plotting helper.
# ---------------------------------------------------------------------------

def _population_bounds(pi, t, percentile):
    """
    Population-weighted percentile bounds on composition.

    Returns (lower, upper): the compositions below which / above which
    `percentile` percent of the total population lies.
    """
    if not (0 < percentile < 50):
        raise ValueError(
            f"population_percentile must be in (0, 50); got {percentile}."
        )
    order = np.argsort(pi)
    pi_s = pi[order]
    t_s = t[order]
    T = t_s.sum()
    cum = np.cumsum(t_s)
    frac = percentile / 100.0
    n = len(pi)
    idx_lo = min(int(np.searchsorted(cum, frac * T)), n - 1)
    idx_hi = min(int(np.searchsorted(cum, (1 - frac) * T)), n - 1)
    return float(pi_s[idx_lo]), float(pi_s[idx_hi])


def plot_composition_curve(data, *args, index="D",
                           ax=None, legend=False,
                           population_percentile=10,
                           n_points=200, v_range=(-30.0, 30.0)):
    """
    Plot the index's composition curve over the trustworthy range.

    By default the curve runs from the lower to the upper
    `population_percentile`-th population-weighted percentile of
    composition: with the default value of 10, the plot spans the
    central 80% of the population. Outside this range the data is
    sparse and the corrected index is an extrapolation.

    No root-finding is used. The tilting parameter v is swept uniformly
    over `v_range`, the resulting curve is sliced to the chosen
    percentile range, and the sweep is automatically widened if the
    initial range does not reach the bounds.

    Pass `population_percentile=None` to plot the full sweep with no
    slicing. The legend is off by default; pass `legend=True` to show it.

    Requires matplotlib.
    """
    import matplotlib.pyplot as plt

    code = _canonical_index(index)
    pi, t = _to_pi_t(data, *args)

    if population_percentile is None:
        curve = composition_curve(data, *args, index=code,
                                  n_points=n_points, v_range=v_range)
    else:
        lo, hi = _population_bounds(pi, t, population_percentile)
        lo_v, hi_v = v_range
        for _ in range(10):  # widen up to 10 times if the sweep falls short
            curve = composition_curve(data, *args, index=code,
                                      n_points=n_points,
                                      v_range=(lo_v, hi_v))
            if (curve["composition"].min() <= lo and
                    curve["composition"].max() >= hi):
                break
            lo_v *= 2
            hi_v *= 2
        curve = curve[(curve["composition"] >= lo) &
                      (curve["composition"] <= hi)]

    mu0 = (t * pi).sum() / t.sum()
    val0 = _compute(pi, t, mu0, code)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    name = _INDEX_LONG_NAMES[code]
    ax.plot(curve["composition"], curve[code], lw=2,
            label="Adjusted (I-projection)")
    ax.scatter([mu0], [val0], s=80, zorder=5, color="C3",
               label=f"Observed (composition = {mu0:.3f})")
    ax.set_xlabel("Composition")
    ax.set_ylabel(name)
    ax.set_title(f"{name} vs. composition")
    if legend:
        ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Eight synthetic neighborhoods, two groups.
    df = pd.DataFrame({
        "white":    [400, 600, 800, 200, 500, 350, 750, 100],
        "hispanic": [600, 400, 200, 800, 500, 650, 250, 900],
    })
    A = df["white"].to_numpy()
    B = df["hispanic"].to_numpy()
    M = np.column_stack([A, B])

    # All three input forms give the same answer.
    d_df = dissimilarity(df, "white", "hispanic")
    d_M  = dissimilarity(M)
    d_v  = dissimilarity(A, B)
    print("Three equivalent input forms:")
    print(f"  DataFrame:  D = {d_df:.4f}")
    print(f"  Matrix:     D = {d_M:.4f}")
    print(f"  Vectors:    D = {d_v:.4f}")

    # Every index, unadjusted.
    print("\nUnadjusted indices (white vs. hispanic):")
    for code in ("D", "S", "H", "R", "G", "Iso"):
        v = segregation_index(df, "white", "hispanic", index=code)
        print(f"  {code:>3s} = {v:.4f}")

    # Same indices, adjusted via the correction.
    print("\nAdjusted to mu = 0.50:")
    for code in ("D", "S", "H", "R", "G"):
        v = segregation_index(df, "white", "hispanic", index=code, target=0.50)
        print(f"  {code:>3s} = {v:.4f}")

    # Long names and short codes are interchangeable.
    print("\nName aliasing:")
    print(f"  index='D'             -> {segregation_index(df, 'white', 'hispanic', index='D'):.4f}")
    print(f"  index='dissimilarity' -> {segregation_index(df, 'white', 'hispanic', index='dissimilarity'):.4f}")
    print(f"  index='Gini'          -> {segregation_index(df, 'white', 'hispanic', index='Gini'):.4f}")

    # Full curve, multiple indices in one sweep.
    curve = composition_curve(df, "white", "hispanic", index=["D", "H", "G"])
    print(f"\nCurve: {len(curve)} points, columns = {list(curve.columns)}")
    print(curve.head(3).round(4).to_string(index=False))
