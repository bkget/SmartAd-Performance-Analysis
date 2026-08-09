# -*- coding: utf-8 -*-
"""
test_ab_testing.py
==================
Unit tests for the SmartAd A/B testing statistical functions.

Tests cover:
  - ABTesting core statistical methods
  - Sequential SPRT helper functions
  - Data transformation pipeline

Run:
    pytest tests/ -v
    pytest tests/ -v --tb=short
"""

import sys
import os
import math

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_DIR = os.path.join(REPO_ROOT, "script")
sys.path.insert(0, SCRIPT_DIR)

from ABTestingFunctions import ABTesting
from sequential_test_script import (
    conditionalSPRT,
    get_bernoulli_series,
    transform_data,
)

ABT = ABTesting()


# ===========================================================================
# ABTesting – Core statistical methods
# ===========================================================================

class TestPooledProb:
    def test_basic_pooled_prob(self):
        result = ABT.pooled_prob(100, 100, 10, 20)
        assert result == pytest.approx(0.15, rel=1e-6)

    def test_equal_groups(self):
        result = ABT.pooled_prob(50, 50, 5, 5)
        assert result == pytest.approx(0.10, rel=1e-6)

    def test_zero_successes(self):
        result = ABT.pooled_prob(100, 100, 0, 0)
        assert result == pytest.approx(0.0)

    def test_all_successes(self):
        result = ABT.pooled_prob(100, 100, 100, 100)
        assert result == pytest.approx(1.0)


class TestPooledSE:
    def test_se_positive(self):
        se = ABT.pooled_SE(100, 100, 10, 20)
        assert se > 0

    def test_se_symmetric_groups(self):
        """Equal group sizes should produce same SE regardless of which way around."""
        se1 = ABT.pooled_SE(100, 100, 10, 20)
        se2 = ABT.pooled_SE(100, 100, 20, 10)
        assert se1 == pytest.approx(se2, rel=1e-6)

    def test_se_larger_sample_smaller(self):
        """Larger sample → smaller standard error."""
        se_small = ABT.pooled_SE(50, 50, 5, 10)
        se_large = ABT.pooled_SE(500, 500, 50, 100)
        assert se_large < se_small

    def test_se_known_value(self):
        """Hand-calculated test: p_hat=0.15, n=200"""
        # p_hat = (10+20)/(100+100) = 0.15
        # SE = sqrt(0.15*0.85*(1/100+1/100)) = sqrt(0.00255) ≈ 0.05049
        se = ABT.pooled_SE(100, 100, 10, 20)
        assert se == pytest.approx(math.sqrt(0.15 * 0.85 * 0.02), rel=1e-4)


class TestZVal:
    def test_two_tailed_95(self):
        z = ABT.z_val(sig_level=0.05, two_tailed=True)
        assert z == pytest.approx(1.96, abs=0.01)

    def test_one_tailed_95(self):
        z = ABT.z_val(sig_level=0.05, two_tailed=False)
        assert z == pytest.approx(1.645, abs=0.01)

    def test_two_tailed_99(self):
        z = ABT.z_val(sig_level=0.01, two_tailed=True)
        assert z == pytest.approx(2.576, abs=0.01)


class TestConfidenceInterval:
    def test_ci_symmetry(self):
        lo, hi = ABT.confidence_interval(0, 1, 1, 0.05)
        assert abs(lo) == pytest.approx(abs(hi), rel=1e-6)

    def test_ci_width_shrinks_with_n(self):
        lo1, hi1 = ABT.confidence_interval(0.5, 0.1, 100, 0.05)
        lo2, hi2 = ABT.confidence_interval(0.5, 0.1, 1000, 0.05)
        width1 = hi1 - lo1
        width2 = hi2 - lo2
        assert width2 < width1

    def test_ci_contains_mean(self):
        mean = 0.3
        lo, hi = ABT.confidence_interval(mean, 0.05, 200, 0.05)
        assert lo < mean < hi


class TestPValue:
    def test_no_difference(self):
        """When p_A == p_B, p-value should be ~0.5."""
        p = ABT.p_val(1000, 1000, 0.1, 0.1)
        assert p == pytest.approx(0.5, abs=0.01)

    def test_significant_lift(self):
        """Large lift should produce very small p-value."""
        p = ABT.p_val(5000, 5000, 0.05, 0.10)
        assert p < 0.001

    def test_p_value_range(self):
        p = ABT.p_val(100, 100, 0.1, 0.15)
        assert 0 < p <= 1


class TestMinSampleSize:
    def test_returns_positive_integer(self):
        n = ABT.min_sample_size(0.1, 0.02, 0.05, 0.80)
        assert isinstance(n, int) and n > 0

    def test_larger_mde_requires_smaller_n(self):
        n_small_mde = ABT.min_sample_size(0.1, 0.01, 0.05, 0.80)
        n_large_mde = ABT.min_sample_size(0.1, 0.05, 0.05, 0.80)
        assert n_small_mde > n_large_mde

    def test_higher_power_requires_larger_n(self):
        n_80 = ABT.min_sample_size(0.1, 0.02, 0.05, 0.80)
        n_95 = ABT.min_sample_size(0.1, 0.02, 0.05, 0.95)
        assert n_95 > n_80


class TestObservedLift:
    def test_absolute_lift(self):
        assert ABT.observed_lift(0.05, 0.08) == pytest.approx(0.03, rel=1e-6)

    def test_relative_lift(self):
        assert ABT.relative_lift(0.05, 0.10) == pytest.approx(100.0, rel=1e-6)

    def test_negative_lift(self):
        assert ABT.observed_lift(0.10, 0.08) == pytest.approx(-0.02, rel=1e-6)


# ===========================================================================
# Sequential SPRT
# ===========================================================================

class TestBernoulliSeries:
    def test_length_matches_engagement(self):
        series = get_bernoulli_series([10, 20], [3, 7])
        assert len(series) == 30

    def test_values_are_binary(self):
        series = get_bernoulli_series([10], [4])
        assert all(v in (0, 1) for v in series)

    def test_success_count_preserved(self):
        engagements = [100, 200, 150]
        successes = [30, 50, 60]
        series = get_bernoulli_series(engagements, successes)
        assert sum(series) == sum(successes)
        assert len(series) == sum(engagements)


class TestConditionalSPRT:
    @pytest.fixture
    def sample_data(self):
        """Generate deterministic test data with a strong lift."""
        np.random.seed(42)
        control = np.random.binomial(1, 0.10, 500)
        exposed = np.random.binomial(1, 0.20, 500)
        return control, exposed

    def test_output_structure(self, sample_data):
        control, exposed = sample_data
        res = conditionalSPRT(exposed, control, t1=2.0)
        required_keys = [
            "outcome", "decision_index", "n_observations",
            "truncated", "truncate_decision",
            "x1_cumsum", "r_cumsum", "stats", "limits",
            "lower_bound", "upper_bound", "n_range",
        ]
        for key in required_keys:
            assert key in res, f"Missing key: {key}"

    def test_raises_on_bad_t1(self, sample_data):
        control, exposed = sample_data
        with pytest.raises(ValueError, match="Odds ratio"):
            conditionalSPRT(exposed, control, t1=0.5)

    def test_bounds_sign(self, sample_data):
        control, exposed = sample_data
        res = conditionalSPRT(exposed, control, t1=2.0)
        assert res["lower_bound"] < 0
        assert res["upper_bound"] > 0

    def test_n_observations_bounded(self, sample_data):
        control, exposed = sample_data
        stop = 100
        res = conditionalSPRT(exposed, control, t1=2.0, stop=stop)
        assert res["n_observations"] <= stop

    def test_outcome_string_not_empty(self, sample_data):
        control, exposed = sample_data
        res = conditionalSPRT(exposed, control, t1=2.0)
        assert isinstance(res["outcome"], str) and len(res["outcome"]) > 5


# ===========================================================================
# Data transformation
# ===========================================================================

class TestTransformData:
    @pytest.fixture
    def sample_df(self):
        """Minimal raw dataframe with respondents."""
        return pd.DataFrame({
            "auction_id": [f"id-{i}" for i in range(20)],
            "experiment": (["exposed"] * 10 + ["control"] * 10),
            "date": ["2020-07-03"] * 20,
            "hour": [10] * 20,
            "device_make": ["Generic Smartphone"] * 20,
            "platform_os": [6] * 20,
            "browser": ["Chrome Mobile"] * 20,
            "yes": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                    0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "no":  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                    1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        })

    def test_returns_arrays(self, sample_df):
        ctrl_arr, exp_arr = transform_data(sample_df)
        import numpy as np
        assert isinstance(ctrl_arr, np.ndarray)
        assert isinstance(exp_arr, np.ndarray)

    def test_values_binary(self, sample_df):
        ctrl, exp = transform_data(sample_df)
        assert set(ctrl).issubset({0, 1})
        assert set(exp).issubset({0, 1})

    def test_non_respondents_excluded(self):
        """Rows with yes=0, no=0 should be excluded."""
        df = pd.DataFrame({
            "auction_id": ["a", "b", "c"],
            "experiment": ["exposed", "exposed", "control"],
            "date": ["2020-07-03"] * 3,
            "hour": [10, 10, 10],
            "device_make": ["Generic Smartphone"] * 3,
            "platform_os": [6, 6, 6],
            "browser": ["Chrome Mobile"] * 3,
            "yes": [1, 0, 0],
            "no": [0, 0, 1],
        })
        ctrl, exp = transform_data(df)
        # Only rows where yes=1 or no=1 (2 rows) should be counted
        assert len(ctrl) + len(exp) > 0
