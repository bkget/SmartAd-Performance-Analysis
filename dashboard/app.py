# -*- coding: utf-8 -*-
"""
SmartAd Performance Analysis – Interactive Dashboard
=====================================================
Multi-tab Streamlit dashboard powered by Plotly for interactive
exploration of A/B test results across three methodologies:
Classical, Sequential (SPRT), and Machine Learning.

Fully adaptable to Streamlit's native Light & Dark mode.

Run:
    python -m streamlit run dashboard/app.py
"""

import os
import sys
import warnings
import math
import random

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as scs
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_DIR = os.path.join(REPO_ROOT, "script")
DATA_DIR = os.path.join(REPO_ROOT, "data")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
sys.path.insert(0, SCRIPT_DIR)

from ABTestingFunctions import ABTesting
from sequential_test_script import (
    conditionalSPRT,
    transform_data,
    get_bernoulli_series,
)

ABT = ABTesting()

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartAd Performance Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dynamic CSS using Streamlit's native CSS theme variables
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

      html, body, [class*="css"] { 
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; 
      }

      /* Hero Header */
      .hero-title {
          font-size: 2.6rem;
          font-weight: 800;
          background: linear-gradient(135deg, #2563eb 0%, #4f46e5 50%, #ec4899 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          line-height: 1.25;
          margin-bottom: 0.2rem;
          letter-spacing: -0.02em;
      }
      .hero-sub {
          font-size: 1.05rem;
          color: var(--text-color);
          opacity: 0.75;
          font-weight: 500;
          margin-bottom: 1.5rem;
      }

      /* Native Theme Adaptable Metric Cards */
      .metric-card {
          background-color: var(--secondary-background-color);
          border: 1px solid rgba(128, 128, 128, 0.2);
          border-radius: 14px;
          padding: 1.2rem 1.4rem;
          box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
          transition: transform 0.2s ease, box-shadow 0.2s ease;
      }
      .metric-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 22px rgba(0, 0, 0, 0.08);
      }
      .metric-label {
          font-size: 0.78rem;
          font-weight: 700;
          color: var(--text-color);
          opacity: 0.65;
          text-transform: uppercase;
          letter-spacing: 0.06em;
      }
      .metric-value {
          font-size: 2.1rem;
          font-weight: 800;
          color: var(--text-color);
          line-height: 1.2;
          margin-top: 0.2rem;
      }
      .metric-delta {
          font-size: 0.85rem;
          font-weight: 600;
          color: #2563eb;
          margin-top: 0.25rem;
      }

      /* Section headers */
      .section-header {
          font-size: 1.35rem;
          font-weight: 700;
          color: var(--text-color);
          padding-bottom: 0.4rem;
          border-bottom: 2px solid rgba(128, 128, 128, 0.2);
          margin-bottom: 1.2rem;
          margin-top: 1rem;
      }

      /* Native Theme Adaptable Callout Boxes */
      .callout-success {
          background-color: rgba(16, 185, 129, 0.12);
          border-left: 5px solid #10b981;
          border-radius: 10px;
          padding: 1.1rem 1.3rem;
          color: var(--text-color);
          font-size: 0.98rem;
          margin: 1rem 0;
      }
      .callout-danger {
          background-color: rgba(239, 68, 68, 0.12);
          border-left: 5px solid #ef4444;
          border-radius: 10px;
          padding: 1.1rem 1.3rem;
          color: var(--text-color);
          font-size: 0.98rem;
          margin: 1rem 0;
      }
      .callout-info {
          background-color: rgba(59, 130, 246, 0.12);
          border-left: 5px solid #3b82f6;
          border-radius: 10px;
          padding: 1.1rem 1.3rem;
          color: var(--text-color);
          font-size: 0.98rem;
          margin: 1rem 0;
      }

      /* --- OVERHAULED SIDEBAR NAVIGATION PILLS --- */
      div[data-testid="stSidebar"] div[role="radiogroup"] {
          gap: 6px !important;
          display: flex !important;
          flex-direction: column !important;
      }
      
      div[data-testid="stSidebar"] div[role="radiogroup"] label {
          background: rgba(128, 128, 128, 0.08) !important;
          border: 1px solid rgba(128, 128, 128, 0.15) !important;
          border-radius: 12px !important;
          padding: 10px 14px !important;
          width: 100% !important;
          margin: 0 !important;
          transition: all 0.2s ease-in-out !important;
          cursor: pointer !important;
          display: flex !important;
          align-items: center !important;
      }
      
      div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
          background: rgba(59, 130, 246, 0.15) !important;
          border-color: #3b82f6 !important;
          transform: translateX(4px) !important;
      }
      
      /* Hide standard circular radio dot */
      div[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
          display: none !important;
      }
      
      div[data-testid="stSidebar"] div[role="radiogroup"] label[aria-checked="true"] {
          background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
          border-color: #6366f1 !important;
          color: #ffffff !important;
          box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
          font-weight: 700 !important;
      }
      div[data-testid="stSidebar"] div[role="radiogroup"] label[aria-checked="true"] * {
          color: #ffffff !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data loading & caching
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data():
    raw_path = os.path.join(DATA_DIR, "AdSmart_AB_testing_data.csv")
    df_raw = pd.read_csv(raw_path)

    clean_path = os.path.join(DATA_DIR, "clean_data.csv")
    if os.path.exists(clean_path):
        df_clean = pd.read_csv(clean_path)
    else:
        df_clean = _clean_data(df_raw)

    return df_raw, df_clean


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Inline cleaning when clean_data.csv is not present."""
    clean = df.query("not (yes == 0 and no == 0)").copy()
    clean["aware"] = clean["yes"].astype(int)
    clean["date"] = pd.to_datetime(clean["date"])
    clean["day_of_week"] = clean["date"].dt.day_name()
    clean["is_exposed"] = (clean["experiment"] == "exposed").astype(int)

    def brand(d):
        d = str(d).lower()
        for k, v in [("samsung","Samsung"),("iphone","Apple"),("pixel","Google"),
                     ("huawei","Huawei"),("oneplus","OnePlus"),("xiaomi","Xiaomi"),
                     ("htc","HTC"),("nokia","Nokia"),("lg","LG"),
                     ("motorola","Motorola"),("moto","Motorola"),("generic","Generic")]:
            if k in d:
                return v
        return "Other"

    clean["device_brand"] = clean["device_make"].apply(brand)
    clean["os_name"] = clean["platform_os"].map({5: "iOS", 6: "Android"}).fillna("Other")

    def browser_grp(b):
        b = str(b).lower()
        if "facebook" in b: return "Facebook"
        if "samsung" in b: return "Samsung Internet"
        if "webview" in b: return "Chrome WebView"
        if "chrome mobile ios" in b: return "Chrome iOS"
        if "chrome mobile" in b: return "Chrome Mobile"
        if "mobile safari" in b: return "Mobile Safari"
        return "Other"

    clean["browser_group"] = clean["browser"].apply(browser_grp)
    return clean


# ---------------------------------------------------------------------------
# Transparent Plotly Theme (Adapts automatically to Streamlit Native Theme)
# ---------------------------------------------------------------------------

PLOTLY_BG = "rgba(0,0,0,0)"
PLOTLY_PAPER_BG = "rgba(0,0,0,0)"

PALETTE = {
    "control": "#2563eb",   # Royal Blue
    "exposed": "#ec4899",   # Vibrant Pink
    "success": "#10b981",   # Emerald Green
    "warn": "#f59e0b",      # Amber
    "danger": "#ef4444",    # Red
    "neutral": "#64748b",   # Slate Grey
    "accent": "#8b5cf6",    # Purple
}


def render_sidebar():
    with st.sidebar:
        # Sidebar Banner
        st.markdown(
            """
            <div style='text-align:center; padding:0.8rem 0 1.2rem; background: rgba(128, 128, 128, 0.08); border-radius: 16px; border: 1px solid rgba(128, 128, 128, 0.15); margin-bottom: 1.2rem;'>
              <div style='font-size:2.4rem;'>📊</div>
              <div style='font-size:1.2rem; font-weight:800; letter-spacing:-0.01em;'>SmartAd Analytics</div>
              <div style='font-size:0.78rem; opacity:0.75; margin-top:0.15rem;'>
                A/B Testing Intelligence Platform
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Navigation Pill Radio
        st.markdown("<div style='font-size:0.75rem; font-weight:800; opacity:0.7; letter-spacing:0.08em; margin-bottom:0.4rem;'>NAVIGATION</div>", unsafe_allow_html=True)
        tab_choice = st.radio(
            "Select Analysis",
            [
                "🏠 Executive Summary",
                "🔍 Data Explorer",
                "📐 Classical A/B Test",
                "📈 Sequential SPRT",
                "🤖 Machine Learning",
            ],
            label_visibility="collapsed",
        )

        st.markdown("<div style='margin-bottom: 1.2rem;'></div>", unsafe_allow_html=True)

        # Global Settings
        st.markdown("<div style='font-size:0.75rem; font-weight:800; opacity:0.7; letter-spacing:0.08em; margin-bottom:0.4rem;'>GLOBAL SETTINGS</div>", unsafe_allow_html=True)
        sig_level = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01,
                              help="Probability of Type I error (false positive).")
        power = st.slider("Statistical Power (1−β)", 0.70, 0.99, 0.80, 0.01,
                          help="Probability of detecting a true effect.")

        st.markdown(
            "<div style='font-size:0.75rem; opacity:0.6; text-align:center; margin-top:2rem;'>"
            "SmartAd Performance Analysis<br>Campaign: Jul 3–10, 2020"
            "</div>",
            unsafe_allow_html=True,
        )

    return tab_choice, sig_level, power


# ---------------------------------------------------------------------------
# Metric card helper
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, delta: str = "", col=None):
    html = f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {'<div class="metric-delta">'+delta+'</div>' if delta else ''}
    </div>
    """
    target = col if col else st
    target.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 1: Executive Summary
# ---------------------------------------------------------------------------

def tab_executive_summary(df_raw, df_clean, sig_level, power):
    st.markdown('<div class="hero-title">SmartAd Performance Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">A/B testing intelligence for brand-awareness lift measurement '
        '| Campaign: July 3–10, 2020</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    total = len(df_raw)
    respondents = len(df_clean)

    ctrl = df_clean[df_clean["experiment"] == "control"]
    exp_ = df_clean[df_clean["experiment"] == "exposed"]

    p_ctrl = ctrl["aware"].mean()
    p_exp = exp_["aware"].mean()
    lift = p_exp - p_ctrl
    rel_lift = lift / p_ctrl * 100 if p_ctrl > 0 else 0

    N_C, N_E = len(ctrl), len(exp_)
    X_C, X_E = ctrl["aware"].sum(), exp_["aware"].sum()
    p_hat = (X_C + X_E) / (N_C + N_E)
    se = math.sqrt(p_hat * (1 - p_hat) * (1 / N_C + 1 / N_E))
    z = lift / se if se > 0 else 0
    pval = scs.norm().sf(abs(z)) * 2

    metric_card("Total Impressions", f"{total:,}", f"{respondents:,} respondents", c1)
    metric_card("Control Conv. Rate", f"{p_ctrl:.2%}", f"{N_C:,} users", c2)
    metric_card("Exposed Conv. Rate", f"{p_exp:.2%}", f"{N_E:,} users", c3)
    metric_card("Absolute Lift", f"{lift:+.3%}", f"{rel_lift:+.1f}% relative lift", c4)
    sig_label = "✅ Significant" if pval < sig_level else "❌ Not Significant"
    metric_card("P-value", f"{pval:.4f}", sig_label, c5)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Conversion Rate Comparison</div>', unsafe_allow_html=True)
        fig = go.Figure()
        groups = ["Control", "Exposed"]
        rates = [p_ctrl, p_exp]
        colors = [PALETTE["control"], PALETTE["exposed"]]
        fig.add_trace(go.Bar(
            x=groups, y=rates,
            marker_color=colors,
            text=[f"{r:.2%}" for r in rates],
            textposition="outside",
            width=0.45,
        ))
        fig.add_hline(y=p_ctrl, line_dash="dot", line_color=PALETTE["neutral"],
                      annotation_text="Baseline", annotation_position="right")
        fig.update_layout(
            paper_bgcolor=PLOTLY_PAPER_BG,
            plot_bgcolor=PLOTLY_BG,
            yaxis_title="Conversion Rate",
            yaxis_tickformat=".1%",
            showlegend=False,
            height=380,
            margin=dict(t=30, b=20, l=40, r=20),
        )
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.markdown('<div class="section-header">Daily Awareness Trend</div>', unsafe_allow_html=True)
        df_clean["date_dt"] = pd.to_datetime(df_clean["date"])
        daily = df_clean.groupby(["date_dt", "experiment"])["aware"].mean().reset_index()
        daily.columns = ["date", "experiment", "aware_rate"]

        fig2 = px.line(
            daily, x="date", y="aware_rate", color="experiment",
            color_discrete_map={"control": PALETTE["control"], "exposed": PALETTE["exposed"]},
            markers=True, labels={"aware_rate": "Awareness Rate", "experiment": "Group"},
        )
        fig2.update_layout(
            paper_bgcolor=PLOTLY_PAPER_BG,
            plot_bgcolor=PLOTLY_BG,
            yaxis_tickformat=".1%",
            height=380,
            margin=dict(t=30, b=20, l=40, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig2, width='stretch')

    st.markdown('<div class="section-header">Awareness Rate by Hour of Day</div>', unsafe_allow_html=True)

    hour_df = df_clean.groupby(["hour", "experiment"])["aware"].agg(["mean", "count"]).reset_index()
    hour_df.columns = ["hour", "experiment", "aware_rate", "count"]

    fig3 = px.bar(
        hour_df, x="hour", y="aware_rate", color="experiment",
        barmode="group",
        color_discrete_map={"control": PALETTE["control"], "exposed": PALETTE["exposed"]},
        labels={"aware_rate": "Awareness Rate", "hour": "Hour of Day"},
    )
    fig3.update_layout(
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_BG,
        yaxis_tickformat=".1%",
        height=340,
        margin=dict(t=10, b=20, l=40, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig3, width='stretch')

    callout_cls = "callout-success" if pval < sig_level else "callout-danger"
    conclusion = (
        f"The exposed group shows a statistically significant increase in brand awareness "
        f"(p = {pval:.4f} < α = {sig_level}). The ad campaign generated a **{lift:+.3%} absolute lift** "
        f"({rel_lift:+.1f}% relative improvement) over the control."
        if pval < sig_level else
        f"At α = {sig_level}, no statistically significant difference was found "
        f"(p = {pval:.4f}). The observed lift of {lift:+.3%} is not statistically significant."
    )
    st.markdown(f'<div class="{callout_cls}">{conclusion}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 2: Data Explorer
# ---------------------------------------------------------------------------

def tab_data_explorer(df_raw, df_clean):
    st.markdown('<div class="section-header">🔍 Data Explorer</div>', unsafe_allow_html=True)

    with st.expander("📌 Dataset Information", expanded=True):
        st.markdown("""
        **SmartAd A/B Testing Dataset** — collected July 3–10, 2020.

        | Column | Type | Description |
        |---|---|---|
        | `auction_id` | UUID | Unique ad impression identifier |
        | `experiment` | categorical | `exposed` (creative ad) or `control` (dummy ad) |
        | `date` | date | Day of impression |
        | `hour` | int 0–23 | Hour of day |
        | `device_make` | string | Device model name |
        | `platform_os` | int | Operating system: `5`=iOS, `6`=Android |
        | `browser` | string | Browser used |
        | `yes` | binary | `1` = brand aware after viewing ad |
        | `no` | binary | `1` = not brand aware |
        """)

    col1, col2, col3 = st.columns(3)
    with col1:
        exp_filter = st.multiselect("Experiment Group", ["control", "exposed"],
                                    default=["control", "exposed"])
    with col2:
        browser_filter = st.multiselect(
            "Browser Group",
            sorted(df_clean["browser_group"].unique()) if "browser_group" in df_clean.columns
            else sorted(df_raw["browser"].unique()),
            default=None,
        )
    with col3:
        if "device_brand" in df_clean.columns:
            brand_filter = st.multiselect(
                "Device Brand", sorted(df_clean["device_brand"].unique()), default=None
            )
        else:
            brand_filter = []

    filtered = df_clean.copy() if "experiment" in df_clean.columns else df_raw.copy()
    if exp_filter:
        filtered = filtered[filtered["experiment"].isin(exp_filter)]
    if browser_filter and "browser_group" in filtered.columns:
        filtered = filtered[filtered["browser_group"].isin(browser_filter)]
    if brand_filter and "device_brand" in filtered.columns:
        filtered = filtered[filtered["device_brand"].isin(brand_filter)]

    st.markdown(f"**Showing {len(filtered):,} records** after applying filters:")
    st.dataframe(
        filtered.head(500),
        width='stretch',
        height=320,
    )

    st.markdown('<div class="section-header">Distribution Analysis</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        if "browser_group" in filtered.columns:
            br_counts = filtered["browser_group"].value_counts().reset_index()
            br_counts.columns = ["browser", "count"]
        else:
            br_counts = filtered["browser"].value_counts().head(8).reset_index()
            br_counts.columns = ["browser", "count"]

        fig = px.pie(br_counts, values="count", names="browser",
                     hole=0.45, title="Browser Share",
                     color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_layout(paper_bgcolor=PLOTLY_PAPER_BG,
                          plot_bgcolor=PLOTLY_BG, height=360, margin=dict(t=40, b=10))
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, width='stretch')

    with c2:
        if "device_brand" in filtered.columns:
            dev_counts = filtered["device_brand"].value_counts().reset_index()
            dev_counts.columns = ["brand", "count"]
            fig2 = px.bar(dev_counts, x="count", y="brand", orientation="h",
                          title="Device Brand Share",
                          color="count", color_continuous_scale="Blues")
            fig2.update_layout(paper_bgcolor=PLOTLY_PAPER_BG,
                                plot_bgcolor=PLOTLY_BG, height=360,
                                margin=dict(t=40, b=10), showlegend=False,
                                coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig2, width='stretch')


# ---------------------------------------------------------------------------
# Tab 3: Classical A/B Test
# ---------------------------------------------------------------------------

def tab_classical(df_clean, sig_level, power):
    st.markdown('<div class="section-header">📐 Classical A/B Hypothesis Testing</div>', unsafe_allow_html=True)

    ctrl = df_clean[df_clean["experiment"] == "control"]
    exp_ = df_clean[df_clean["experiment"] == "exposed"]

    N_C = len(ctrl)
    N_E = len(exp_)
    X_C = int(ctrl["aware"].sum())
    X_E = int(exp_["aware"].sum())
    p_ctrl = X_C / N_C
    p_exp = X_E / N_E
    d_hat = p_exp - p_ctrl

    st.markdown("### 🎛 Experiment Parameters")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        bcr_override = st.number_input(
            "Baseline Conversion Rate (BCR)",
            min_value=0.001, max_value=0.999,
            value=round(p_ctrl, 4), step=0.001, format="%.4f",
        )
        mde = st.slider("Minimum Detectable Effect (MDE)",
                        0.001, 0.10, 0.01, 0.001)
    with cc2:
        use_observed = st.toggle("Use observed lift as MDE", value=True)
        mde_value = round(abs(d_hat), 4) if use_observed else mde
        st.info(f"MDE in use: **{mde_value:.4f}** ({mde_value/bcr_override*100:.1f}% relative)")
    with cc3:
        min_n = ABT.min_sample_size(bcr_override, mde_value, sig_level, power)
        metric_card("Required Sample Size / Group", f"{min_n:,}",
                    "✅ Sufficient Data" if min(N_C, N_E) >= min_n else "⚠️ Underpowered", cc3)

    st.divider()

    X_A = bcr_override * N_C
    X_B = (bcr_override + d_hat) * N_E
    stderr = ABT.pooled_SE(N_C, N_E, X_A, X_B)
    pval = ABT.p_val(N_C, N_E, bcr_override, p_exp)
    ci_low, ci_high = ABT.confidence_interval(d_hat, stderr, N_C + N_E, sig_level)

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    metric_card("Control Rate", f"{p_ctrl:.3%}", f"n = {N_C:,}", col_r1)
    metric_card("Exposed Rate", f"{p_exp:.3%}", f"n = {N_E:,}", col_r2)
    metric_card("Observed Lift (d̂)", f"{d_hat:+.3%}", f"Relative: {d_hat/p_ctrl*100:+.1f}%", col_r3)
    metric_card("P-value", f"{pval:.4f}",
                "✅ Significant" if pval < sig_level else "❌ Not Significant", col_r4)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="callout-info">95% Confidence Interval for lift: '
        f'<strong>[{ci_low:+.4f}, {ci_high:+.4f}]</strong></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 📊 Hypothesis Sampling Distribution")

    stderr_plot = ABT.pooled_SE(N_C, N_E, X_A, X_B)
    x_range = np.linspace(-6 * stderr_plot, d_hat + 6 * stderr_plot, 1000)

    null_dist = scs.norm(0, stderr_plot)
    alt_dist = scs.norm(d_hat, stderr_plot)
    z_crit = null_dist.ppf(1 - sig_level)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_range, y=null_dist.pdf(x_range),
        name="H₀ — Null (No Effect)", mode="lines",
        line=dict(color=PALETTE["control"], width=3),
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=alt_dist.pdf(x_range),
        name="H₁ — Alternative (Lift Exists)", mode="lines",
        line=dict(color=PALETTE["exposed"], width=3),
    ))

    fig.add_vline(x=z_crit, line_dash="dash", line_color=PALETTE["warn"],
                  annotation_text=f"Critical threshold: {z_crit:.4f}",
                  annotation_position="top right")
    fig.add_vline(x=d_hat, line_dash="dot", line_color=PALETTE["success"],
                  annotation_text=f"Observed lift: {d_hat:.4f}",
                  annotation_position="top left")

    fig.update_layout(
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_BG,
        height=420,
        xaxis_title="Difference in Conversion Rate (d)",
        yaxis_title="Probability Density",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=40, l=60, r=20),
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Tab 4: Sequential SPRT
# ---------------------------------------------------------------------------

def tab_sequential(df_raw, sig_level, power):
    st.markdown('<div class="section-header">📈 Sequential Probability Ratio Test (SPRT)</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        t1 = st.slider("Odds Ratio under H₁ (t₁)", 1.1, 5.0, 2.0, 0.1)
    with col2:
        beta_val = st.slider("Type II Error Rate (β)", 0.05, 0.30, 0.10, 0.01)
    with col3:
        max_obs = st.slider("Max Observations (truncation)", 100, 2000, 800, 50)

    with st.spinner("Running SPRT..."):
        control_arr, exposed_arr = transform_data(df_raw)

    res = conditionalSPRT(
        x=exposed_arr,
        y=control_arr,
        t1=t1,
        alpha=sig_level,
        beta=beta_val,
        stop=max_obs,
    )

    is_positive = "significant increase" in res["outcome"].lower() or "h1" in str(res["truncate_decision"]).lower()
    callout_cls = "callout-success" if is_positive else (
        "callout-danger" if "no statistically" in res["outcome"].lower() else "callout-info"
    )
    st.markdown(f'<div class="{callout_cls}"><strong>Outcome:</strong> {res["outcome"]}</div>',
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    k_disp = str(res["decision_index"]) if not (isinstance(res["decision_index"], float) and math.isnan(res["decision_index"])) else "—"
    metric_card("Decision at Observation", k_disp, "", m1)
    metric_card("Total Observations", str(res["n_observations"]), "", m2)
    metric_card("Lower Boundary (L)", f"{res['lower_bound']:.4f}", "Accept H₀ below", m3)
    metric_card("Upper Boundary (U)", f"{res['upper_bound']:.4f}", "Reject H₀ above", m4)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔬 SPRT Decision Trajectory")

    n = res["n_range"]
    x1 = res["x1_cumsum"]
    limits = res["limits"]
    yl = limits[:, 0]
    yu = limits[:, 1]
    k = res["decision_index"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=n, y=yu, mode="lines",
        line=dict(color=PALETTE["danger"], width=2.5, dash="dash"),
        name="Upper boundary (reject H₀)",
    ))
    fig.add_trace(go.Scatter(
        x=n, y=yl, mode="lines",
        line=dict(color=PALETTE["control"], width=2.5, dash="dash"),
        name="Lower boundary (accept H₀)",
    ))
    fig.add_trace(go.Scatter(
        x=n, y=x1, mode="lines+markers",
        line=dict(color=PALETTE["accent"], width=2.5),
        marker=dict(size=4, color=PALETTE["accent"]),
        name="Cumulative successes (exposed)",
    ))

    if not (isinstance(k, float) and math.isnan(k)):
        fig.add_vline(x=int(n[k]), line_dash="dot", line_color=PALETTE["warn"],
                      annotation_text=f"Decision at n={n[k]}")
        fig.add_trace(go.Scatter(
            x=[n[k]], y=[x1[k]], mode="markers",
            marker=dict(symbol="star", size=18, color=PALETTE["warn"]),
            name=f"Decision point (n={n[k]})",
        ))

    fig.update_layout(
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_BG,
        height=450,
        xaxis_title="Observation Number (n)",
        yaxis_title="Cumulative Successes (x₁)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=40, l=60, r=20),
    )
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Tab 5: Machine Learning
# ---------------------------------------------------------------------------

def tab_machine_learning(df_clean, sig_level):
    st.markdown('<div class="section-header">🤖 Machine Learning Analysis</div>', unsafe_allow_html=True)

    try:
        from sklearn.model_selection import train_test_split, StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score, f1_score, roc_auc_score,
            confusion_matrix, classification_report, roc_curve
        )
        import scipy.stats as stat
        try:
            from xgboost import XGBClassifier
            has_xgboost = True
        except ImportError:
            has_xgboost = False
    except ImportError:
        st.error("scikit-learn is required.")
        return

    cat_cols = []
    feature_cols = ["is_exposed", "hour"]
    if "day_of_week" in df_clean.columns:
        feature_cols.append("day_of_week")
    if "device_brand" in df_clean.columns:
        feature_cols.append("device_brand"); cat_cols.append("device_brand")
    if "os_name" in df_clean.columns:
        feature_cols.append("os_name"); cat_cols.append("os_name")
    if "browser_group" in df_clean.columns:
        feature_cols.append("browser_group"); cat_cols.append("browser_group")

    df_ml = df_clean[feature_cols + ["aware"]].dropna()
    num_cols = [c for c in feature_cols if c not in cat_cols]
    X = pd.concat([
        df_ml[num_cols].reset_index(drop=True),
        pd.get_dummies(df_ml[cat_cols], drop_first=True).reset_index(drop=True)
    ], axis=1)
    y = df_ml["aware"].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    clf_options = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "Random Forest": "random_forest",
    }
    if has_xgboost:
        clf_options["XGBoost"] = "xgboost"

    col_sel, col_folds = st.columns([2, 1])
    with col_sel:
        clf_label = st.selectbox("Select Classifier", list(clf_options.keys()))
    with col_folds:
        n_folds = st.slider("CV Folds", 3, 10, 5)

    clf_key = clf_options[clf_label]

    @st.cache_data(show_spinner="Training model...")
    def train_model(clf_key, n_folds, X_train_vals, X_test_vals, y_train_vals, y_test_vals, feature_names):
        X_tr = pd.DataFrame(X_train_vals, columns=feature_names)
        X_te = pd.DataFrame(X_test_vals, columns=feature_names)
        y_tr = pd.Series(y_train_vals)
        y_te = pd.Series(y_test_vals)

        clf_map = {
            "logistic_regression": LogisticRegression(max_iter=1000, solver="lbfgs"),
            "decision_tree": DecisionTreeClassifier(max_depth=4, random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        }
        if has_xgboost:
            clf_map["xgboost"] = XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                use_label_encoder=False, eval_metric="logloss", random_state=42, verbosity=0,
            )

        clf = clf_map[clf_key]
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_accs, fold_f1s, fold_aucs = [], [], []
        for tr_idx, val_idx in skf.split(X_tr, y_tr):
            Xf, yf = X_tr.iloc[tr_idx], y_tr.iloc[tr_idx]
            Xv, yv = X_tr.iloc[val_idx], y_tr.iloc[val_idx]
            clf.fit(Xf, yf)
            preds = clf.predict(Xv)
            proba = clf.predict_proba(Xv)[:, 1] if hasattr(clf, "predict_proba") else preds
            fold_accs.append(accuracy_score(yv, preds))
            fold_f1s.append(f1_score(yv, preds, zero_division=0))
            fold_aucs.append(roc_auc_score(yv, proba) if len(np.unique(yv)) > 1 else np.nan)

        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else y_pred

        return {
            "clf": clf,
            "fold_accs": fold_accs, "fold_f1s": fold_f1s, "fold_aucs": fold_aucs,
            "y_pred": y_pred, "y_proba": y_proba,
            "acc": accuracy_score(y_te, y_pred),
            "f1": f1_score(y_te, y_pred, zero_division=0),
            "auc": roc_auc_score(y_te, y_proba) if len(np.unique(y_te)) > 1 else np.nan,
            "cm": confusion_matrix(y_te, y_pred),
            "report": classification_report(y_te, y_pred, zero_division=0),
        }

    result = train_model(
        clf_key, n_folds,
        X_train.values, X_test.values,
        y_train.values, y_test.values,
        list(X.columns),
    )

    clf = result["clf"]
    X_tr_df = pd.DataFrame(X_train.values, columns=X.columns)

    st.markdown("### 📊 Performance Summary")
    m1, m2, m3 = st.columns(3)
    metric_card("Test Accuracy", f"{result['acc']:.3%}", f"CV: {np.mean(result['fold_accs']):.3%}", m1)
    metric_card("Test F1 Score", f"{result['f1']:.4f}", f"CV: {np.mean(result['fold_f1s']):.4f}", m2)
    metric_card("Test ROC-AUC", f"{result['auc']:.4f}", f"CV: {np.nanmean(result['fold_aucs']):.4f}", m3)

    st.divider()

    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.markdown("#### Confusion Matrix")
        cm = result["cm"]
        cm_labels = ["Not Aware", "Aware"]
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            x=cm_labels, y=cm_labels,
            labels={"x": "Predicted", "y": "Actual"},
        )
        fig_cm.update_layout(
            paper_bgcolor=PLOTLY_PAPER_BG, plot_bgcolor=PLOTLY_BG,
            height=340, margin=dict(t=10, b=30, l=60, r=20),
        )
        st.plotly_chart(fig_cm, width='stretch')

    with col_roc:
        st.markdown("#### ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, result["y_proba"])
        auc_val = result["auc"]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            line=dict(color=PALETTE["exposed"], width=2.5),
            name=f"ROC (AUC = {auc_val:.4f})",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color=PALETTE["neutral"], dash="dash", width=1.5),
            name="Random Classifier",
        ))
        fig_roc.update_layout(
            paper_bgcolor=PLOTLY_PAPER_BG, plot_bgcolor=PLOTLY_BG,
            height=340, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            margin=dict(t=10, b=30, l=60, r=20),
        )
        st.plotly_chart(fig_roc, width='stretch')

    st.divider()

    if clf_key == "logistic_regression":
        try:
            decision = clf.decision_function(X_tr_df)
            denom = 2.0 * (1.0 + np.cosh(decision))
            denom_tile = np.tile(denom, (X_tr_df.shape[1], 1)).T
            X_mat = X_tr_df.values.astype(np.float64)
            F_ij = np.dot((X_mat / denom_tile).T, X_mat)
            Cramer_Rao = np.linalg.pinv(F_ij)
            sigma = np.sqrt(np.maximum(0, np.diagonal(Cramer_Rao)))
            z_scores = np.where(sigma > 0, clf.coef_[0] / sigma, 0.0)
            pvals = [float(2 * stat.norm.sf(abs(z))) for z in z_scores]
        except Exception:
            pvals = [np.nan] * len(X.columns)

        p_df = pd.DataFrame({
            "Feature": X.columns.tolist(),
            "Coefficient": clf.coef_[0],
            "P-value": pvals,
            "Significant": [p < sig_level for p in pvals],
        }).sort_values("P-value")

        colors = [PALETTE["success"] if s else PALETTE["danger"] for s in p_df["Significant"]]
        fig_pv = go.Figure(go.Bar(
            x=p_df["P-value"], y=p_df["Feature"],
            orientation="h", marker_color=colors, opacity=0.85,
            text=[f"{p:.4f}" for p in p_df["P-value"]], textposition="outside",
        ))
        fig_pv.add_vline(x=sig_level, line_dash="dash", line_color=PALETTE["warn"],
                         annotation_text=f"α = {sig_level}", annotation_position="top right")
        fig_pv.update_layout(
            paper_bgcolor=PLOTLY_PAPER_BG, plot_bgcolor=PLOTLY_BG,
            height=max(350, len(p_df) * 28),
            xaxis_title="P-value", title="Feature Significance P-values",
            margin=dict(t=40, b=20, l=180, r=80),
        )
        st.plotly_chart(fig_pv, width='stretch')

    elif hasattr(clf, "feature_importances_"):
        fi_df = pd.DataFrame({
            "Feature": X.columns.tolist(),
            "Importance": clf.feature_importances_,
        }).sort_values("Importance", ascending=False).head(15)

        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h", marker_color=PALETTE["control"],
            opacity=0.85,
        ))
        fig_fi.update_layout(
            paper_bgcolor=PLOTLY_PAPER_BG, plot_bgcolor=PLOTLY_BG,
            height=max(350, len(fi_df) * 28),
            xaxis_title="Importance Score", title=f"Top Feature Importances ({clf_label})",
            margin=dict(t=40, b=20, l=180, r=80),
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig_fi, width='stretch')


# ---------------------------------------------------------------------------
# Main app entry point
# ---------------------------------------------------------------------------

def main():
    df_raw, df_clean = load_data()
    tab_choice, sig_level, power = render_sidebar()

    if "Executive Summary" in tab_choice:
        tab_executive_summary(df_raw, df_clean, sig_level, power)
    elif "Data Explorer" in tab_choice:
        tab_data_explorer(df_raw, df_clean)
    elif "Classical" in tab_choice:
        tab_classical(df_clean, sig_level, power)
    elif "Sequential" in tab_choice:
        tab_sequential(df_raw, sig_level, power)
    elif "Machine Learning" in tab_choice:
        tab_machine_learning(df_clean, sig_level)


if __name__ == "__main__":
    main()
