## SmartAd Performance Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-FF4B4B?style=flat-square&logo=streamlit)
![MLflow](https://img.shields.io/badge/MLflow-1.26%2B-0194E2?style=flat-square&logo=mlflow)
![DVC](https://img.shields.io/badge/DVC-2.10%2B-945DD6?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**A production-grade A/B testing intelligence platform for measuring brand-awareness lift from online advertising campaigns.**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Methodologies](#methodologies)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Interactive Dashboard](#interactive-dashboard)
- [MLOps Pipeline](#mlops-pipeline)
- [Testing](#testing)
- [Collaborators](#collaborators)

---

## Overview

SmartAd is an advertising company that runs online ad campaigns and wants to quantify the **increase in brand awareness** caused by its creative ads. This repository provides a full analytical and ML pipeline to rigorously answer the question:

> *Did displaying the SmartAd creative ad result in a statistically significant increase in user brand awareness compared to the control (dummy) ad?*

Three complementary approaches are implemented:
1. **Classical A/B Testing** — Two-proportion Z-test with hypothesis distribution visualisation
2. **Sequential SPRT Testing** — Meeker's Conditional SPRT for continuous monitoring and early stopping
3. **Machine Learning** — Predictive modelling to identify which user attributes drive brand awareness

---

## Business Problem

Traditional advertising effectiveness metrics (CTR, impressions) don't directly measure **brand recall or awareness**. SmartAd collects post-ad survey responses from users (`yes` = brand aware, `no` = not aware) to quantify this.

**Key hypotheses:**
- **H₀ (Null):** The exposed group's awareness rate equals the control group's rate (p_exposed = p_control)
- **H₁ (Alternative):** The exposed group has a higher awareness rate (p_exposed > p_control)

**Why three approaches?**

| Approach | Strength | Weakness |
|---|---|---|
| Classical Z-test | Interpretable, widely understood | Requires fixed sample size in advance |
| Sequential SPRT | Can stop early, controls error rates continuously | More complex to implement |
| Machine Learning | Reveals *which features* drive awareness | Less interpretable for causal inference |

---

## Dataset

The dataset contains **8,079 ad impressions** collected from July 3–10, 2020.

### Schema

| Column | Type | Description |
|---|---|---|
| `auction_id` | UUID | Unique ad impression identifier |
| `experiment` | categorical | `exposed` (creative ad) or `control` (dummy ad) |
| `date` | date | Day of the impression |
| `hour` | int (0–23) | Hour of day |
| `device_make` | string | Device model name (e.g., Samsung SM-G960F) |
| `platform_os` | int | Operating system: `5`=iOS, `6`=Android |
| `browser` | string | Browser used (Chrome Mobile, Facebook, etc.) |
| `yes` | binary | `1` if user reported brand awareness |
| `no` | binary | `1` if user reported no brand awareness |

### Key Observations

- **Response rate:** ~15% of users responded to the brand awareness survey
- **Exposed vs Control:** Roughly equal split between groups
- **Device landscape:** Dominated by Android Chrome Mobile and Samsung devices
- **High cardinality:** 250+ unique device models (grouped into brands for ML)

---

## Methodologies

### 1. Classical A/B Testing (`notebooks/classical_AB_testing.ipynb`)

Uses a two-proportion Z-test to compare conversion rates:

- **Pooled standard error:** SE = √[ p̂(1-p̂)(1/N_A + 1/N_B) ]
- **Z-statistic:** z = (p_B - p_A) / SE
- **Minimum sample size** estimation for planning future campaigns
- **Confidence intervals** for the observed lift
- **Visualisation** of H₀ and H₁ sampling distributions

### 2. Sequential SPRT (`notebooks/sequential_AB_testing.ipynb`)

Implements Meeker's Conditional SPRT for matched binary sequences:

- At each observation n, compute the log-likelihood ratio Sₙ
- Compare against decision boundaries: L = log(β/(1-α)) and U = -log(α/(1-β))
- **Stop early** if Sₙ crosses either boundary
- Converts hourly aggregated data into Bernoulli time series

### 3. Machine Learning (`notebooks/ml_AB_testing.ipynb`)

Frames brand awareness as a binary classification problem:

- **Features:** experiment group, hour, day, device brand, OS, browser
- **Models:** Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Evaluation:** Stratified 5-fold CV, accuracy, F1, ROC-AUC
- **Interpretability:** Feature p-values (Logistic Regression), feature importances (tree models)

---

## Project Structure

```
SmartAd-Performance-Analysis/
├── data/
│   ├── AdSmart_AB_testing_data.csv    # Raw dataset (8,079 impressions)
│   └── clean_data.csv                 # Processed, feature-engineered data
│
├── script/
│   ├── ABTestingFunctions.py          # Classical A/B statistical methods
│   ├── PlottingFunctions.py           # Matplotlib visualisation helpers
│   ├── HypothesisPlot.py              # H₀/H₁ distribution visualisations
│   ├── sequential_test_script.py      # Meeker's Conditional SPRT
│   ├── train_classifiers.py           # ML classifier training class
│   ├── data_pipeline.py               # Data loading & feature engineering
│   ├── train_pipeline.py              # MLOps training entry point
│   └── evaluate.py                    # MLOps evaluation entry point
│
├── notebooks/
│   ├── EDA_pipeline.ipynb             # Exploratory data analysis
│   ├── classical_AB_testing.ipynb     # Classical A/B test walkthrough
│   ├── sequential_AB_testing.ipynb    # SPRT walkthrough
│   └── ml_AB_testing.ipynb            # ML analysis walkthrough
│
├── dashboard/
│   └── app.py                         # Interactive Streamlit dashboard
│
├── models/
│   └── model.pkl                      # Saved trained model
│
├── tests/
│   └── test_ab_testing.py             # Pytest unit tests
│
├── dvc.yaml                           # DVC pipeline definition
├── params.yaml                        # Pipeline parameters
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- Git

### Installation

```bash
git clone https://github.com/bkget/SmartAd-Performance-Analysis
cd SmartAd-Performance-Analysis
pip install -r requirements.txt
```

### Run the data pipeline

```bash
python script/data_pipeline.py
```

### Run unit tests

```bash
pytest tests/ -v
```

### Train classifiers

```bash
python script/train_pipeline.py
# Or with custom settings:
python script/train_pipeline.py --classifier xgboost --folds 5
```

---

## Interactive Dashboard

Launch the multi-tab Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

### Dashboard Features

| Tab | Description |
|---|---|
| Executive Summary | Campaign KPIs, conversion rates, daily trend, hourly heatmap |
| Data Explorer | Interactive filters by group/browser/device, distribution charts |
| Classical A/B Test | Hypothesis distributions, confidence intervals, power curves |
| Sequential SPRT | Live SPRT trajectory, log-likelihood ratio, early stopping analysis |
| Machine Learning | Model comparison, ROC curves, confusion matrices, feature importance |

---

## MLOps Pipeline

The project uses **DVC** for data versioning and pipeline reproducibility, and **MLflow** for experiment tracking.

### Run the full pipeline

```bash
dvc repro
```

### View MLflow experiments

```bash
mlflow ui
# Open http://localhost:5000
```

### Pipeline stages

```
data/AdSmart_AB_testing_data.csv
        │
        ▼
   [prepare]  ─── data_pipeline.py
        │
        ▼
  data/clean_data.csv
        │
        ▼
    [train]   ─── train_pipeline.py
        │
        ▼
  models/model.pkl
        │
        ▼
  [evaluate]  ─── evaluate.py
        │
        ▼
  metrics.json
```

---

## Testing

Run the full test suite:

```bash
# All tests with verbose output
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=script --cov-report=html
```

Test coverage includes:
- Pooled probability and standard error calculations
- Z-score and confidence interval accuracy
- P-value computation and edge cases
- Minimum sample size estimation
- SPRT boundary conditions and output structure
- Bernoulli series generation
- Data transformation pipeline

---

## Collaborators

- [Endework Abera](https://github.com/Endework)
- [Didier Iradukunda](https://github.com/didier-i)
- [Biruk Getaneh](https://github.com/bkget)