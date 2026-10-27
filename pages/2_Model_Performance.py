"""Page 2: Model Performance & Comparative Evaluation Benchmark."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import METRICS_REPORT_FILE, REPORTS_DIR
from src.ui_components import (
    FINTECH_PALETTE,
    apply_fintech_theme,
    render_metric_card,
    style_plotly_chart,
)

st.set_page_config(
    page_title="Model Performance | LoanGuard AI",
    page_icon="📈",
    layout="wide",
)
apply_fintech_theme()

st.title("📈 Model Evaluation & Comparative Benchmark")
st.caption("Empirical performance evaluation across TabNet, Logistic Regression, and Random Forest on held-out test data.")

if not METRICS_REPORT_FILE.exists():
    st.warning("⚡ Evaluation metrics have not been generated yet. Please run `python src/evaluate.py` to evaluate models.")
    st.stop()

with open(METRICS_REPORT_FILE, "r", encoding="utf-8") as f:
    metrics_data = json.load(f)

models_data = metrics_data.get("models", {})
test_n = metrics_data.get("test_sample_count", 4500)
default_rate = metrics_data.get("test_default_rate", 0.221)

# Imbalance Educational Notice
st.markdown(
    f"""
    <div class="disclaimer-box">
        ⚖️ <strong>Class Imbalance & The Fallacy of Accuracy:</strong><br>
        {metrics_data.get('imbalance_explanation', '')}
    </div>
    """,
    unsafe_allow_html=True,
)

# Test Population Overview
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("Test Population", f"{test_n:,}", "Held-out unobserved records")
with c2:
    render_metric_card("Test Default Rate", f"{default_rate:.1%}", f"{metrics_data.get('test_default_count', 0)} defaults")
with c3:
    tabnet_roc = models_data.get("TabNet", {}).get("roc_auc", 0.0)
    render_metric_card("TabNet ROC-AUC", f"{tabnet_roc:.4f}", "Rank-order discrimination")
with c4:
    tabnet_pr = models_data.get("TabNet", {}).get("pr_auc", 0.0)
    render_metric_card("TabNet PR-AUC", f"{tabnet_pr:.4f}", "Precision over minority class")

st.markdown("---")

# 1. Comparative Benchmark Summary Table
st.subheader("📊 Comparative Model Performance Table")

table_rows = []
for model_name, m in models_data.items():
    table_rows.append({
        "Model Architecture": model_name,
        "ROC-AUC": f"{m['roc_auc']:.4f}",
        "PR-AUC": f"{m['pr_auc']:.4f}",
        "Recall": f"{m['recall']:.4f}",
        "Precision": f"{m['precision']:.4f}",
        "F1-Score": f"{m['f1']:.4f}",
        "Accuracy": f"{m['accuracy']:.4f}",
    })

df_metrics = pd.DataFrame(table_rows)
st.dataframe(df_metrics, hide_index=True, use_container_width=True)

st.markdown("---")

# 2. ROC and PR Curve Visualizations
col_roc, col_pr = st.columns(2)

with col_roc:
    st.subheader("📉 Receiver Operating Characteristic (ROC) Curves")
    st.caption("Plots True Positive Rate (Sensitivity) vs False Positive Rate (1 - Specificity).")

    fig_roc = go.Figure()
    # Diagonal random guess line
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#64748b", dash="dash"),
            name="Random Guess (AUC = 0.50)",
        )
    )

    colors = {
        "TabNet": FINTECH_PALETTE["tabnet_color"],
        "Logistic Regression": FINTECH_PALETTE["baseline_color"],
        "Random Forest": FINTECH_PALETTE["rf_color"],
    }

    for model_name, m in models_data.items():
        roc_coords = m.get("roc_curve", {})
        fig_roc.add_trace(
            go.Scatter(
                x=roc_coords.get("fpr", []),
                y=roc_coords.get("tpr", []),
                mode="lines",
                name=f"{model_name} (AUC = {m['roc_auc']:.3f})",
                line=dict(color=colors.get(model_name, "#ffffff"), width=2.5),
            )
        )

    fig_roc.update_layout(
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
    )
    st.plotly_chart(style_plotly_chart(fig_roc), use_container_width=True)

with col_pr:
    st.subheader("🎯 Precision-Recall (PR) Curves")
    st.caption("Crucial for imbalanced default detection; assesses precision as recall increases.")

    fig_pr = go.Figure()
    # Baseline line at default rate
    fig_pr.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[default_rate, default_rate],
            mode="lines",
            line=dict(color="#64748b", dash="dash"),
            name=f"No-Skill Baseline ({default_rate:.1%})",
        )
    )

    for model_name, m in models_data.items():
        pr_coords = m.get("pr_curve", {})
        fig_pr.add_trace(
            go.Scatter(
                x=pr_coords.get("recall", []),
                y=pr_coords.get("precision", []),
                mode="lines",
                name=f"{model_name} (PR-AUC = {m['pr_auc']:.3f})",
                line=dict(color=colors.get(model_name, "#ffffff"), width=2.5),
            )
        )

    fig_pr.update_layout(
        xaxis_title="Recall (Default Catch Rate)",
        yaxis_title="Precision (Accuracy of Default Flag)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
    )
    st.plotly_chart(style_plotly_chart(fig_pr), use_container_width=True)

st.markdown("---")

# 3. Confusion Matrix Breakdown
st.subheader("🧮 Confusion Matrix Comparison")
st.caption("Detailed distribution of True Positives (defaults identified), False Positives (good borrowers misflagged), False Negatives (missed defaults), and True Negatives.")

cm_cols = st.columns(len(models_data))
for idx, (model_name, m) in enumerate(models_data.items()):
    with cm_cols[idx]:
        cm_info = m.get("confusion_matrix", {})
        matrix_data = [
            [cm_info.get("tn", 0), cm_info.get("fp", 0)],
            [cm_info.get("fn", 0), cm_info.get("tp", 0)],
        ]
        fig_cm = px.imshow(
            matrix_data,
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            x=["Non-Default (0)", "Default (1)"],
            y=["Non-Default (0)", "Default (1)"],
            text_auto=True,
            color_continuous_scale="Tealgrn" if model_name == "TabNet" else "Purp",
        )
        fig_cm.update_layout(
            title=f"{model_name} Confusion Matrix",
            height=320,
        )
        st.plotly_chart(style_plotly_chart(fig_cm, height=320), use_container_width=True)

        st.caption(
            f"**{model_name}:** Caught **{cm_info.get('tp', 0)}** defaults out of "
            f"{cm_info.get('tp', 0) + cm_info.get('fn', 0)} actual defaults "
            f"({m['recall']:.1%} Recall)."
        )
