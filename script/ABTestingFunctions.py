# -*- coding: utf-8 -*-
"""
ABTestingFunctions.py
=====================
Core statistical utilities for Classical A/B hypothesis testing.

This module provides all statistical building blocks for:
  - Pooled proportion and standard error estimation
  - Z-score and critical value calculation
  - Confidence interval construction
  - Minimum sample size computation
  - P-value computation for two-proportion Z-test
  - Normal distribution objects for visualization

Author: SmartAd Performance Analysis Team
"""

import numpy as np
import scipy.stats as scs


class ABTesting:
    """
    A collection of classical A/B testing statistical methods.

    These functions underpin the two-proportion Z-test used to
    determine whether the exposed group's brand-awareness conversion
    rate is significantly higher than the control group's rate.
    """

    def __init__(self):
        """Initialise the ABTesting helper class."""
        pass

    # ------------------------------------------------------------------
    # Core statistical building blocks
    # ------------------------------------------------------------------

    def pooled_prob(self, N_A: int, N_B: int, X_A: int, X_B: int) -> float:
        """
        Compute the pooled probability across two independent samples.

        Under the null hypothesis H0: p_A == p_B, we estimate a single
        shared probability from both samples.

        Parameters
        ----------
        N_A : int  – Total observations in group A (control).
        N_B : int  – Total observations in group B (exposed).
        X_A : int  – Successes in group A.
        X_B : int  – Successes in group B.

        Returns
        -------
        float  – Pooled proportion p_hat = (X_A + X_B) / (N_A + N_B).
        """
        return (X_A + X_B) / (N_A + N_B)

    def pooled_SE(self, N_A: int, N_B: int, X_A: int, X_B: int) -> float:
        """
        Compute the pooled standard error for two proportion difference.

        SE = sqrt( p_hat * (1 - p_hat) * (1/N_A + 1/N_B) )

        Parameters
        ----------
        N_A, N_B : int  – Group sample sizes.
        X_A, X_B : int  – Group success counts.

        Returns
        -------
        float  – Pooled standard error.
        """
        p_hat = self.pooled_prob(N_A, N_B, X_A, X_B)
        SE = np.sqrt(p_hat * (1 - p_hat) * (1 / N_A + 1 / N_B))
        return SE

    def z_val(self, sig_level: float = 0.05, two_tailed: bool = True) -> float:
        """
        Return the critical Z-value for a given significance level.

        Parameters
        ----------
        sig_level  : float – Significance level α (default 0.05).
        two_tailed : bool  – If True, split α equally between both tails.

        Returns
        -------
        float  – Critical Z-score.
        """
        z_dist = scs.norm()
        if two_tailed:
            area = 1 - sig_level / 2
        else:
            area = 1 - sig_level
        return z_dist.ppf(area)

    def confidence_interval(
        self,
        sample_mean: float = 0,
        sample_std: float = 1,
        sample_size: int = 1,
        sig_level: float = 0.05,
    ) -> tuple:
        """
        Compute a symmetric confidence interval around a sample mean.

        CI = mean ± z * (std / sqrt(n))

        Parameters
        ----------
        sample_mean : float  – Point estimate.
        sample_std  : float  – Standard deviation of the estimator.
        sample_size : int    – Number of observations.
        sig_level   : float  – Significance level for the interval.

        Returns
        -------
        (lower, upper) : tuple of floats.
        """
        z = self.z_val(sig_level)
        margin = z * sample_std / np.sqrt(sample_size)
        return (sample_mean - margin, sample_mean + margin)

    def ab_dist(self, stderr: float, mde: float = 0, group_type: str = "control"):
        """
        Build a SciPy normal distribution for visualising H0 or H1.

        Parameters
        ----------
        stderr     : float  – Pooled standard error.
        mde        : float  – Minimum detectable effect (mean shift for H1).
        group_type : str    – 'control' centres at 0; 'exposed' centres at mde.

        Returns
        -------
        scipy.stats.norm  – Frozen normal distribution object.
        """
        sample_mean = mde if group_type == "exposed" else 0
        return scs.norm(sample_mean, stderr)

    def p_val(self, N_A: int, N_B: int, p_A: float, p_B: float) -> float:
        """
        Compute the one-sided p-value for the two-proportion Z-test.

        Uses unpooled standard errors under H1 to compute the Z-score,
        then returns the survival-function (upper-tail) probability.

        Parameters
        ----------
        N_A : int   – Control group size.
        N_B : int   – Exposed group size.
        p_A : float – Control conversion rate (baseline).
        p_B : float – Exposed conversion rate (= p_A + d_hat).

        Returns
        -------
        float  – One-tailed p-value.
        """
        d_hat = p_B - p_A
        std_a = np.sqrt(p_A * (1 - p_A) / N_A)
        std_b = np.sqrt(p_B * (1 - p_B) / N_B)
        z_score = d_hat / np.sqrt(std_a**2 + std_b**2)
        return scs.norm().sf(z_score)

    # ------------------------------------------------------------------
    # Sample-size planning
    # ------------------------------------------------------------------

    def min_sample_size(
        self,
        bcr: float,
        mde: float,
        sig_level: float = 0.05,
        power: float = 0.80,
    ) -> int:
        """
        Estimate the minimum sample size per group for a desired power.

        Uses the standard formula for comparing two proportions:
            n = (z_α + z_β)² * [p1(1-p1) + p2(1-p2)] / mde²

        Parameters
        ----------
        bcr       : float  – Baseline conversion rate (control).
        mde       : float  – Minimum detectable effect (absolute lift).
        sig_level : float  – Significance level α.
        power     : float  – Desired statistical power (1 - β).

        Returns
        -------
        int  – Minimum sample size per group (rounded up).
        """
        p2 = bcr + mde
        z_alpha = self.z_val(sig_level, two_tailed=True)
        z_beta = self.z_val(1 - power, two_tailed=False)
        numerator = (z_alpha + z_beta) ** 2 * (bcr * (1 - bcr) + p2 * (1 - p2))
        denominator = mde ** 2
        return int(np.ceil(numerator / denominator))

    # ------------------------------------------------------------------
    # Observed lift & lift significance
    # ------------------------------------------------------------------

    def observed_lift(self, p_A: float, p_B: float) -> float:
        """Absolute difference in conversion rates (d_hat = p_B - p_A)."""
        return p_B - p_A

    def relative_lift(self, p_A: float, p_B: float) -> float:
        """Percentage lift relative to the control rate."""
        return (p_B - p_A) / p_A * 100