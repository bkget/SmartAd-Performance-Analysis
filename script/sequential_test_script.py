# -*- coding: utf-8 -*-
"""
sequential_test_script.py
=========================
Meeker's Conditional Sequential Probability Ratio Test (SPRT) for A/B testing.

SPRT Overview
-------------
Unlike classical tests that require a fixed sample size decided upfront,
Sequential tests allow continuous monitoring. At each new observation the
cumulative log-likelihood ratio S_n is compared against two boundaries:

  - Upper boundary U = -log(α / (1 - β))  → Reject H0 (evidence for lift)
  - Lower boundary L =  log(β / (1 - α))  → Accept H0 (no significant lift)

If S_n remains between L and U, we continue collecting data.

This implementation follows:
  Meeker (1981) – "A Comparison of Approximate Interval Estimators for the
  Bernoulli Parameter" (adapted conditional version for matched pairs).

Author: SmartAd Performance Analysis Team
"""

import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# Mathematical helper functions for the conditional SPRT
# ---------------------------------------------------------------------------

def _comb(n: int, k: int) -> int:
    """Integer binomial coefficient C(n, k)."""
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)


def _lchoose(b: int, j) -> np.ndarray:
    """Log of binomial coefficient; handles scalar or array j."""
    j_arr = np.atleast_1d(j)
    result = []
    for k in j_arr:
        result.append(math.log(_comb(b, int(k))) if 0 <= k <= b else 0.0)
    return np.array(result)


def _ftermlog(j, r: int, n: int, t: float, offset: float = 0) -> np.ndarray:
    """Log probability of each term in the non-central hypergeometric sum."""
    j = np.atleast_1d(j)
    xx = r - j
    lch = _lchoose(n, j)
    lchdiff = _lchoose(n, xx)
    lg = j * math.log(t)
    return lch + lchdiff + lg - offset


def _fterm(j, r: int, n: int, t: float, offset: float = 0) -> np.ndarray:
    """Exponentiated probability terms."""
    return np.exp(_ftermlog(j, r, n, t, offset))


def _f(r: int, n: int, t: float, offset: float = 0) -> float:
    """Non-central hypergeometric normalising constant."""
    upper = max(0, r - n)
    lower = min(n, r)
    rng = np.arange(upper, lower + 1)
    return float(np.sum(_fterm(rng, r, n, t, offset)))


def _logf(r: int, n: int, t: float, offset: float = 0) -> float:
    z = _f(r, n, t, offset)
    return math.log(z) if z > 0 else np.nan


def _h(x: int, r: int, n: int, t: float = 1) -> float:
    offset = _ftermlog(math.ceil(r / 2), r, n, t)[0]
    return _f(r, n, t, offset=_logf(r, n, 1, offset) + offset)


def _g(x: int, r: int, n: int, t1: float, t0: float = 1) -> float:
    """Log likelihood ratio contribution at a single observation."""
    return -math.log(_h(x, r, n, t1)) + math.log(_h(x, r, n, t0))


def _clower_upper(
    r: int, n: int, t1c: float, t0: float = 1,
    alpha: float = 0.05, beta: float = 0.10
) -> np.ndarray:
    """Compute conditional lower and upper decision boundaries at step n."""
    offset = _ftermlog(math.ceil(r / 2), r, n, t1c)[0]
    z = _logf(r, n, t1c, _logf(r, n, t0, offset) + offset)
    a = -math.log(alpha / (1 - beta))
    b = math.log(beta / (1 - alpha))
    lower = b
    upper = 1 + a
    return (np.array([lower, upper]) + z) / math.log(t1c / t0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def conditionalSPRT(
    x: np.ndarray,
    y: np.ndarray,
    t1: float,
    alpha: float = 0.05,
    beta: float = 0.10,
    stop: Optional[int] = None,
) -> dict:
    """
    Run Meeker's Conditional SPRT on two matched binary sequences.

    Parameters
    ----------
    x     : np.ndarray – Binary responses for the *exposed* (treatment) group.
    y     : np.ndarray – Binary responses for the *control* group.
    t1    : float      – Odds ratio under H1 (must exceed 1 to represent lift).
    alpha : float      – Type I error rate (default 0.05).
    beta  : float      – Type II error rate (default 0.10).
    stop  : int | None – Optional truncation point (max observations).

    Returns
    -------
    dict with keys:
        outcome          – Human-readable decision string.
        decision_index   – Observation index where decision was made (or NaN).
        n_observations   – Total observations evaluated.
        truncated        – Truncation point if applied (or NaN).
        truncate_decision– 'h0' | 'h1' | 'N/A' after truncation.
        x1_cumsum        – Cumulative successes in x.
        r_cumsum         – Cumulative x + y.
        stats            – Log-likelihood ratio series.
        limits           – Array of (lower, upper) boundary pairs.
        lower_bound      – Scalar l (global log boundary for H0 acceptance).
        upper_bound      – Scalar u (global log boundary for H1 rejection).
        n_range          – Observation indices array.
    """
    if t1 <= 1:
        raise ValueError("Odds ratio t1 must exceed 1 to represent a positive lift.")
    if alpha > 0.5 or beta > 0.5:
        import warnings
        warnings.warn("Unrealistic alpha or beta values supplied (> 0.5).")

    l = math.log(beta / (1 - alpha))
    u = -math.log(alpha / (1 - beta))

    sample_size = min(len(x), len(y))
    n = np.arange(1, sample_size + 1)

    if stop is not None:
        n = n[n <= int(stop)]

    x1 = np.cumsum(x[n - 1])
    r = x1 + np.cumsum(y[n - 1])
    stats = np.array([_g(int(x1[i]), int(r[i]), int(n[i]), t1) for i in range(len(n))])

    # Decision boundaries at each step
    limits = np.array([
        [math.floor(v) for v in _clower_upper(int(r[i]), int(n[i]), t1, alpha=alpha, beta=beta)]
        for i in range(len(n))
    ])

    # Find first crossing
    crossings = np.where((stats >= u) | (stats <= l))[0]
    if len(crossings) == 0:
        k = np.nan
        outcome = "Unable to conclude. More data needed."
        truncated = np.nan
        truncate_decision = "N/A"
    else:
        k = int(crossings[0])
        if stats[k] >= u:
            outcome = "[SIGNIFICANT] Exposed group produced a statistically significant increase in brand awareness."
        else:
            outcome = "[NO EFFECT] No statistically significant difference between the two groups."
        truncated = np.nan
        truncate_decision = "N/A"

    # Handle truncation
    if stop is not None and np.isnan(k):
        mid_r = r[-1]
        c1 = _clower_upper(int(mid_r), int(stop), t1, alpha=alpha, beta=beta)
        c1_val = math.floor(np.mean(c1) - 0.5)
        if x1[-1] <= c1_val:
            truncate_decision = "h0"
            outcome = "[TRUNCATED] No significant difference detected at maximum sample size."
        else:
            truncate_decision = "h1"
            outcome = "[TRUNCATED] Approximate evidence for significant increase at maximum limit."
        truncated = int(stop)

    return {
        "outcome": outcome,
        "decision_index": k,
        "n_observations": len(n),
        "truncated": truncated,
        "truncate_decision": truncate_decision,
        "x1_cumsum": x1,
        "r_cumsum": r,
        "stats": stats,
        "limits": limits,
        "lower_bound": l,
        "upper_bound": u,
        "n_range": n,
    }


def get_bernoulli_series(engagement_list: list, success_list: list) -> list:
    """
    Build a shuffled Bernoulli series from grouped engagement/success counts.

    For each (engagement, success) pair, creates a list of 1s and 0s
    where the number of 1s equals `success` and the rest are 0s, then
    randomly shuffles the order before appending to the series.

    Parameters
    ----------
    engagement_list : list[int] – Total engagements per time unit.
    success_list    : list[int] – Positive responses per time unit.

    Returns
    -------
    list[int] – Flat shuffled Bernoulli sequence.
    """
    series = []
    for eng, suc in zip(engagement_list, success_list):
        item = [1] * int(suc) + [0] * int(eng - suc)
        random.shuffle(item)
        series.extend(item)
    return series


def transform_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform raw A/B data into matched Bernoulli sequences for SPRT.

    Steps:
      1. Remove rows where both yes=0 and no=0 (non-respondents).
      2. Split into 'control' and 'exposed' groups.
      3. Create a date_hour composite timestamp column.
      4. Aggregate by date_hour: sum(yes) and count(no) as engagements.
      5. Build Bernoulli series from aggregated hourly data.

    Parameters
    ----------
    df : pd.DataFrame – Raw dataset with columns:
         [auction_id, experiment, date, hour, yes, no, ...].

    Returns
    -------
    (control_bernoulli, exposed_bernoulli) : tuple of np.ndarray
        Binary sequences suitable for conditionalSPRT.
    """
    clean_df = df.query("not (yes == 0 and no == 0)").copy()

    exposed = clean_df[clean_df["experiment"] == "exposed"].copy()
    control = clean_df[clean_df["experiment"] == "control"].copy()

    for grp in [exposed, control]:
        grp["hour"] = grp["hour"].astype(str).str.zfill(2)
        grp["date_hour"] = grp["date"] + ":" + grp["hour"]

    # Exposed
    exp_agg = exposed.groupby("date_hour").agg(success=("yes", "sum"), engagement=("no", "count")).reset_index()
    exposed_bernoulli = get_bernoulli_series(exp_agg["engagement"].tolist(), exp_agg["success"].tolist())

    # Control
    ctrl_agg = control.groupby("date_hour").agg(success=("yes", "sum"), engagement=("no", "count")).reset_index()
    control_bernoulli = get_bernoulli_series(ctrl_agg["engagement"].tolist(), ctrl_agg["success"].tolist())

    return np.array(control_bernoulli), np.array(exposed_bernoulli)


def plot_sprt(res: dict, return_fig: bool = False):
    """
    Plot the SPRT decision trajectory with upper and lower boundaries.

    Parameters
    ----------
    res        : dict – Output from conditionalSPRT().
    return_fig : bool – If True, return the figure; else plt.show().

    Returns
    -------
    matplotlib.figure.Figure if return_fig=True, else None.
    """
    n = res["n_range"]
    x1 = res["x1_cumsum"]
    limits = res["limits"]
    k = res["decision_index"]

    yl = limits[:, 0]
    yu = limits[:, 1]

    fig, ax = plt.subplots(figsize=(16, 8))

    # Trajectory
    ax.scatter(n, x1, color="#2b2d42", s=15, zorder=10, label="Observed cumulative successes")
    ax.plot(n, x1, color="#2b2d42", linewidth=1, alpha=0.5)

    # Boundaries
    lower_line, = ax.plot(n, yl, color="#4361ee", linewidth=2, label="Accept H₀ boundary")
    upper_line, = ax.plot(n, yu, color="#e63946", linewidth=2, label="Reject H₀ boundary")

    # Shaded regions
    y_min = min(yl.min(), x1.min()) - 2
    y_max = max(yu.max(), x1.max()) + 2
    ax.fill_between(n, yl, y_min, color="#4361ee", alpha=0.1)
    ax.fill_between(n, yu, y_max, color="#e63946", alpha=0.1)

    # Decision marker
    if not np.isnan(k):
        ax.axvline(x=n[k], color="#f4a261", linestyle="--", linewidth=2,
                   label=f"Decision at n={n[k]}")
        ax.scatter([n[k]], [x1[k]], color="#f4a261", s=150, zorder=20, marker="*")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Observations (n)", fontsize=13)
    ax.set_ylabel("Cumulative Successes (x₁)", fontsize=13)
    ax.set_title("Sequential Probability Ratio Test (SPRT) – Decision Trajectory", fontsize=15)
    ax.legend(fontsize=10, loc="upper left")
    plt.tight_layout()

    if return_fig:
        return fig
    plt.show()