# -*- coding: utf-8 -*-
"""
HypothesisPlot.py
=================
High-level visualisation of A/B test hypothesis distributions.

Provides three plot types:
  1. hypo_plot  – General H0/H1 plot from group sizes and conversion rates.
  2. abplot     – Compact A/B test distribution with optional annotations.
  3. zplot      – Standard normal with confidence interval shading.

All methods optionally return the figure for embedding in Streamlit or
saving to disk instead of calling plt.show().

Author: SmartAd Performance Analysis Team
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs

from PlottingFunctions import PlottingFunctions
from ABTestingFunctions import ABTesting

PLTF = PlottingFunctions()
ABT = ABTesting()


class HypothesisPlot:
    """
    Render hypothesis-test visualisations for classical A/B experiments.

    Methods
    -------
    hypo_plot   – H0 vs H1 distributions from raw group parameters.
    abplot      – A/B test distribution from observed lift (d_hat).
    zplot       – Standard normal curve with confidence shading.
    """

    def __init__(self):
        """Initialise HypothesisPlot."""
        pass

    def hypo_plot(
        self,
        Control: int,
        Exposed: int,
        bcr: float,
        mde: float,
        sig_level: float = 0.05,
        show_power: bool = False,
        show_beta: bool = False,
        show_alpha: bool = False,
        show_p_value: bool = False,
        show_legend: bool = True,
        return_fig: bool = False,
    ):
        """
        Plot the null and alternative distributions for a planned A/B test.

        The null distribution is centred at 0 (no effect); the alternative
        is shifted by `mde` (minimum detectable effect).

        Parameters
        ----------
        Control    : int   – Number of control observations.
        Exposed    : int   – Number of exposed observations.
        bcr        : float – Baseline conversion rate (control group).
        mde        : float – Minimum detectable absolute lift.
        sig_level  : float – Significance level α (default 0.05).
        show_power : bool  – Shade the statistical power region.
        show_beta  : bool  – Shade the Type II error region.
        show_alpha : bool  – Shade the Type I error region.
        show_p_value: bool – Annotate the empirical p-value.
        show_legend: bool  – Display the legend.
        return_fig : bool  – Return figure instead of calling plt.show().

        Returns
        -------
        matplotlib.figure.Figure if return_fig=True, else None.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Compute pooled standard error
        X_C = bcr * Control
        X_E = (bcr + mde) * Exposed
        stderr = ABT.pooled_SE(Control, Exposed, X_C, X_E)

        # Draw both distributions
        PLTF.plot_null(ax, stderr)
        PLTF.plot_alt(ax, stderr, mde)
        ax.set_xlim(-8 * stderr, 8 * stderr + mde)

        # Shade requested regions
        if show_power:
            PLTF.show_area(ax, mde, stderr, sig_level, area_type="power")
        if show_alpha:
            PLTF.show_area(ax, mde, stderr, sig_level, area_type="alpha")
        if show_beta:
            PLTF.show_area(ax, mde, stderr, sig_level, area_type="beta")

        # Annotate p-value — computed dynamically
        if show_p_value:
            p_value = ABT.p_val(Control, Exposed, bcr, bcr + mde)
            null = ABT.ab_dist(stderr, group_type="control")
            ax.text(
                3 * stderr,
                null.pdf(0),
                f"P-value = {p_value:.4f}",
                fontsize=12,
                ha="left",
                color="#e63946",
                fontweight="bold",
            )

        if show_legend:
            plt.legend(fontsize=10)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Difference in conversion rate (d)", fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.set_title("A/B Test – Null vs Alternative Hypothesis Distributions", fontsize=14)
        plt.tight_layout()

        if return_fig:
            return fig
        plt.show()

    def abplot(
        self,
        N_A: int,
        N_B: int,
        bcr: float,
        d_hat: float,
        sig_level: float = 0.05,
        show_power: bool = False,
        show_alpha: bool = False,
        show_beta: bool = False,
        show_p_value: bool = False,
        show_legend: bool = True,
        return_fig: bool = False,
    ):
        """
        Plot A/B distributions from observed data (post-hoc).

        Parameters
        ----------
        N_A    : int   – Control group size.
        N_B    : int   – Exposed group size.
        bcr    : float – Baseline (control) conversion rate.
        d_hat  : float – Observed lift (p_B - p_A).
        Others : Same as hypo_plot.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        X_A = bcr * N_A
        X_B = (bcr + d_hat) * N_B
        stderr = ABT.pooled_SE(N_A, N_B, X_A, X_B)

        PLTF.plot_null(ax, stderr)
        PLTF.plot_alt(ax, stderr, d_hat)
        ax.set_xlim(-8 * stderr, 8 * stderr + d_hat)

        if show_power:
            PLTF.show_area(ax, d_hat, stderr, sig_level, area_type="power")
        if show_alpha:
            PLTF.show_area(ax, d_hat, stderr, sig_level, area_type="alpha")
        if show_beta:
            PLTF.show_area(ax, d_hat, stderr, sig_level, area_type="beta")

        if show_p_value:
            p_value = ABT.p_val(N_A, N_B, bcr, bcr + d_hat)
            null = ABT.ab_dist(stderr, group_type="control")
            ax.text(
                3 * stderr,
                null.pdf(0),
                f"P-value = {p_value:.4f}",
                fontsize=12,
                ha="left",
                color="#e63946",
                fontweight="bold",
            )

        if show_legend:
            plt.legend(fontsize=10)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Difference in conversion rate (d)", fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.set_title("Observed A/B Test Distribution", fontsize=14)
        plt.tight_layout()

        if return_fig:
            return fig
        plt.show()

    def zplot(
        self,
        area: float = 0.95,
        two_tailed: bool = True,
        align_right: bool = False,
        return_fig: bool = False,
    ):
        """
        Plot the standard normal curve with a confidence-interval shading.

        Parameters
        ----------
        area        : float – The confidence level (proportion of curve shaded).
        two_tailed  : bool  – If True, shade the central area symmetrically.
        align_right : bool  – For one-tailed tests, shade the right tail.
        return_fig  : bool  – Return figure instead of calling plt.show().
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        norm = scs.norm()
        x = np.linspace(-5, 5, 1000)
        y = norm.pdf(x)
        ax.plot(x, y, color="#4361ee", linewidth=2.5)

        if two_tailed:
            left = norm.ppf(0.5 - area / 2)
            right = norm.ppf(0.5 + area / 2)
            ax.vlines(right, 0, norm.pdf(right), color="grey", linestyle="--")
            ax.vlines(left, 0, norm.pdf(left), color="grey", linestyle="--")
            ax.fill_between(x, 0, y, color="#4361ee", alpha=0.25,
                            where=(x > left) & (x < right))
            ax.text(left, norm.pdf(left), f"z = {left:.3f}", fontsize=11,
                    rotation=90, va="bottom", ha="right")
            ax.text(right, norm.pdf(right), f"z = {right:.3f}", fontsize=11,
                    rotation=90, va="bottom", ha="left")
        else:
            if align_right:
                left = norm.ppf(1 - area)
                ax.vlines(left, 0, norm.pdf(left), color="grey", linestyle="--")
                ax.fill_between(x, 0, y, color="#4361ee", alpha=0.25, where=x > left)
                ax.text(left, norm.pdf(left), f"z = {left:.3f}", fontsize=11,
                        rotation=90, va="bottom", ha="right")
            else:
                right = norm.ppf(area)
                ax.vlines(right, 0, norm.pdf(right), color="grey", linestyle="--")
                ax.fill_between(x, 0, y, color="#4361ee", alpha=0.25, where=x < right)
                ax.text(right, norm.pdf(right), f"z = {right:.3f}", fontsize=11,
                        rotation=90, va="bottom", ha="left")

        ax.text(0, 0.08, f"Shaded area = {area:.3f}", fontsize=12, ha="center")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("z", fontsize=12)
        ax.set_ylabel("PDF", fontsize=12)
        ax.set_title("Standard Normal Distribution", fontsize=14)
        plt.tight_layout()

        if return_fig:
            return fig
        plt.show()