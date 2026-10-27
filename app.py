"""LoanGuard AI - Explainable Loan Default Prediction Using TabNet.

Main Entry Point and Home Dashboard.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    BASELINE_LR_FILE,
    DATASET_PATH,
    METRICS_REPORT_FILE,
    PREPROCESSOR_FILE,
    TABNET_MODEL_FILE,
)
from src.ui_components import (
    FINTECH_PALETTE,
    apply_fintech_theme,
    render_metric_card,
)

# Page configuration
st.set_page_config(
    page_title="LoanGuard AI | TabNet Credit Scoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom Fintech styles
apply_fintech_theme()

# Sidebar branding & quick overview
with st.sidebar:
    st.markdown("### 🛡️ LoanGuard AI")
    st.caption("Explainable Credit Default Prediction Engine")
    st.markdown("---")
    st.markdown("**Architecture:** PyTorch TabNet")
    st.markdown("**Dataset:** UCI Credit Card Clients (30,000 records)")
    st.markdown("**Attention Mechanism:** Entmax Sparse Selection")
    st.markdown("---")
    st.info(
        "💡 **Prototype Notice:** LoanGuard AI is an educational demonstration prototype. "
        "It is not intended for real-world automated lending or credit underwriting."
    )

# Hero Header
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🛡️ LoanGuard AI</div>
        <div class="hero-subtitle">
            Next-Generation Explainable Loan Default Prediction powered by Google Research's 
            <strong>TabNet</strong> deep learning architecture. Combining tabular neural representations 
            with step-by-step sparse attention masks.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Educational Disclaimer Banner
st.markdown(
    """
    <div class="disclaimer-box">
        ⚠️ <strong>Educational Prototype:</strong> This application is built for research and educational purposes. 
        In actual consumer lending, regulatory frameworks (e.g., Fair Lending Acts, GDPR) require strict auditing 
        of adverse action notices and prohibit discriminatory demographic features.
    </div>
    """,
    unsafe_allow_html=True,
)

# System Artifact Status Check
col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    dataset_ready = DATASET_PATH.exists()
    status_text = "Ready (Cached)" if dataset_ready else "Not Found"
    render_metric_card("Dataset Status", status_text, "30,000 Client Records" if dataset_ready else "Run data loader")

with col_s2:
    preprocessor_ready = PREPROCESSOR_FILE.exists()
    status_text = "Fitted" if preprocessor_ready else "Not Fitted"
    render_metric_card("Preprocessor", status_text, "Leakage-Safe Scaler & Encoders")

with col_s3:
    model_ready = TABNET_MODEL_FILE.exists()
    status_text = "Trained" if model_ready else "Untrained"
    render_metric_card("TabNet Model", status_text, "PyTorch Deep Tabular Net")

with col_s4:
    metrics_ready = METRICS_REPORT_FILE.exists()
    status_text = "Available" if metrics_ready else "Pending"
    render_metric_card("Evaluation Report", status_text, "Held-Out Test Set Metrics")

st.markdown("---")

# Load and display actual evaluation metrics if available
if metrics_ready:
    with open(METRICS_REPORT_FILE, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    tabnet_m = metrics_data.get("models", {}).get("TabNet", {})
    lr_m = metrics_data.get("models", {}).get("Logistic Regression", {})

    st.subheader("📊 Key Benchmark Performance (Held-Out Test Set)")
    st.caption("Actual empirical performance measured on 4,500 unobserved test applicants.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        roc_val = f"{tabnet_m.get('roc_auc', 0.0):.3f}"
        diff_roc = tabnet_m.get("roc_auc", 0) - lr_m.get("roc_auc", 0)
        sub = f"{'+' if diff_roc >= 0 else ''}{diff_roc:.3f} vs Logistic Reg" if lr_m else "Primary Metric"
        render_metric_card("TabNet ROC-AUC", roc_val, sub)

    with c2:
        pr_val = f"{tabnet_m.get('pr_auc', 0.0):.3f}"
        render_metric_card("Precision-Recall AUC", pr_val, "Precision over default minority")

    with c3:
        rec_val = f"{tabnet_m.get('recall', 0.0):.1%}"
        render_metric_card("Default Recall", rec_val, "Share of actual defaults caught")

    with c4:
        acc_val = f"{tabnet_m.get('accuracy', 0.0):.1%}"
        render_metric_card("Overall Accuracy", acc_val, "Total correct classifications")
else:
    st.warning("⚡ Model has not been trained yet. Run `python src/train.py` or use the training panel to generate artifacts.")

st.markdown("---")

# Technical Overview: How TabNet Works
st.subheader("🧠 Why TabNet for Credit Scoring?")

tab_col1, tab_col2 = st.columns([3, 2])

with tab_col1:
    st.markdown(
        r"""
        Conventional machine learning for tabular datasets is dominated by decision tree ensembles (such as XGBoost 
        or LightGBM), while standard Multi-Layer Perceptrons (MLPs) struggle with tabular inductive bias.

        **TabNet (Arik & Pfister, Google Research)** bridges this gap by bringing tree-like sequential feature selection 
        into a differentiable neural architecture:
        
        1. **Sequential Multi-Step Decision Steps:** 
           Information flows through consecutive decision steps ($N_{steps}$). At each step, a dedicated attentive transformer 
           selects a sparse subset of features to inspect.
        2. **Sparsemax / Entmax Masking:** 
           Instead of standard dense softmax attention, TabNet employs sparse attention ($\mathrm{entmax}_{1.5}$), 
           driving non-relevant feature weights strictly to zero.
        3. **Feature Reuse & Prior Scales:** 
           A memory mechanism tracks how frequently each feature has been queried in prior decision steps, penalizing redundant 
           feature usage and encouraging diverse attention across complementary indicators.
        4. **Native Instance-Level Explainability:** 
           Because feature selection masks are explicit tensors computed during inference, TabNet natively provides exact 
           per-applicant attention masks without relying on costly external approximations.
        """
    )

with tab_col2:
    st.markdown(
        """
        <div class="explain-box">
            <h4>🎯 Key Architectural Hyperparameters</h4>
            <ul>
                <li><strong>n_d, n_a:</strong> 16 (Dimension of decision and attention embeddings)</li>
                <li><strong>n_steps:</strong> 4 (Sequential reasoning steps)</li>
                <li><strong>gamma:</strong> 1.3 (Relaxation parameter enforcing feature diversity)</li>
                <li><strong>lambda_sparse:</strong> 1e-3 (Sparsity regularization loss)</li>
                <li><strong>cat_emb_dim:</strong> Learnable categorical entity embeddings for Sex, Education, and Marital Status</li>
                <li><strong>Virtual Batch Size:</strong> 128 (Ghost Batch Normalization)</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Quick Navigation Section
st.subheader("🧭 Application Navigation")
nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    st.markdown("#### 1. Applicant Prediction")
    st.write("Input applicant financial details to compute default probability, risk tier, and individual feature attention.")

with nav2:
    st.markdown("#### 2. Model Performance")
    st.write("Examine confusion matrices, ROC curves, PR curves, and benchmark comparisons against Logistic Regression.")

with nav3:
    st.markdown("#### 3. Explainability")
    st.write("Visualize global feature importance and inspect step-by-step attention masks across TabNet reasoning steps.")

with nav4:
    st.markdown("#### 4. Dataset Explorer")
    st.write("Explore raw client distributions, correlation matrices, demographic splits, and target class balance.")
