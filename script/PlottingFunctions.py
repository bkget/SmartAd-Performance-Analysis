# -*- coding: utf-8 -*-
"""
PlottingFunctions.py
====================
Low-level Matplotlib helper routines for A/B test distribution plots.

These helpers are called by HypothesisPlot to render:
  - Null distribution (H0 centred at 0)
  - Alternative distribution (H1 centred at MDE)
  - Shaded regions for α (Type I error), β (Type II error), and Power

Author: SmartAd Performance Analysis Team
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs

from ABTestingFunctions import ABTesting

ABT = ABTesting()


class PlottingFunctions:
    """Low-level helpers for annotating A/B test distribution figures."""

    def __init__(self):
        pass

    def plot_null(self, ax, stderr: float, label: str = "H₀ (Null)") -> None:
        """
        Draw the null-hypothesis normal distribution (centred at 0).

        Parameters
        ----------
        ax     : matplotlib Axes object.
        stderr : float – Pooled standard error (controls distribution width).
        label  : str   – Legend label for the curve.
        """
        null_dist = ABT.ab_dist(stderr, group_type="control")
        x = np.linspace(null_dist.ppf(0.0001), null_dist.ppf(0.9999), 500)
        ax.plot(x, null_dist.pdf(x), label=label, color="#4361ee", linewidth=2.5)

    def plot_alt(self, ax, stderr: float, mde: float, label: str = "H₁ (Alternative)") -> None:
        """
        Draw the alternative-hypothesis normal distribution (centred at MDE).

        Parameters
        ----------
        ax     : matplotlib Axes object.
        stderr : float – Pooled standard error.
        mde    : float – Minimum detectable effect (mean shift).
        label  : str   – Legend label for the curve.
        """
        alt_dist = ABT.ab_dist(stderr, mde=mde, group_type="exposed")
        x = np.linspace(alt_dist.ppf(0.0001), alt_dist.ppf(0.9999), 500)
        ax.plot(x, alt_dist.pdf(x), label=label, color="#f72585", linewidth=2.5)

    def show_area(
        self,
        ax,
        mde: float,
        stderr: float,
        sig_level: float = 0.05,
        area_type: str = "alpha",
    ) -> None:
        """
        Shade a region under one of the distributions.

        Parameters
        ----------
        ax         : matplotlib Axes.
        mde        : float – Effect size.
        stderr     : float – Pooled standard error.
        sig_level  : float – Significance level α.
        area_type  : str   – One of 'alpha', 'beta', or 'power'.
        """
        null = ABT.ab_dist(stderr, group_type="control")
        alt = ABT.ab_dist(stderr, mde=mde, group_type="exposed")

        # The critical threshold for rejecting H0 (right-tailed)
        z_crit = null.ppf(1 - sig_level)
        x = np.linspace(-8 * stderr, 8 * stderr + mde, 1000)

        if area_type == "alpha":
            # Region where H0 is incorrectly rejected (Type I error)
            ax.fill_between(
                x, null.pdf(x),
                where=(x >= z_crit),
                alpha=0.5, color="#e63946",
                label=f"α = {sig_level:.2f} (Type I error)",
            )
        elif area_type == "beta":
            # Region where H1 is incorrectly rejected (Type II error)
            ax.fill_between(
                x, alt.pdf(x),
                where=(x < z_crit),
                alpha=0.5, color="#f4a261",
                label=f"β (Type II error)",
            )
        elif area_type == "power":
            # Region where H1 is correctly detected
            ax.fill_between(
                x, alt.pdf(x),
                where=(x >= z_crit),
                alpha=0.5, color="#2a9d8f",
                label=f"Power = 1 - β",
            )