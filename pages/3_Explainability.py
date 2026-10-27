"""Page 3: TabNet Native Explainability & Sequential Attention Masks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    FEATURE_DESCRIPTIONS,
    FEATURE_IMPORTANCE_FILE,
    PREPROCESSOR_FILE,
    TABNET_MODEL_FILE,
    TEST_DATA_CACHE,
)
from src.predict import load_artifacts
from src.ui_components import (
    FINTECH_PALETTE,
    apply_fintech_theme,
    render_metric_card,
    style_plotly_chart,
)

st.set_page_config(
    page_title="Explainability | LoanGuard AI",
    page_icon="🔬",
    layout="wide",
)
apply_fintech_theme()

st.title("🔬 TabNet Native Explainability & Attention Dynamics")
st.caption("Deep inspection of Sparsemax/Entmax feature routing across sequential decision steps.")

# Scientific & Regulatory Caveat Box
st.markdown(
    """
    <div class="explain-box">
        <h4>🔬 What TabNet Attention Masks Can and Cannot Establish</h4>
        <ul>
            <li><strong>What It CAN Establish:</strong> TabNet's masks mathematically quantify which subset of features the neural network 
            actively focused on to produce the prediction. Features with near-zero attention were masked out during that step.</li>
            <li><strong>What It CANNOT Establish:</strong> Attention does <em>not</em> imply real-world causality. An attention weight of 40% on 
            <code>PAY_0</code> means the model heavily relied on September repayment delay; it does not scientifically prove that 
            the delay was the root cause of insolvency.</li>
            <li><strong>Directionality:</strong> TabNet attention masks are non-negative magnitude scores. They indicate feature salience, not 
            whether a feature drove default risk upward or downward.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

if not FEATURE_IMPORTANCE_FILE.exists() or not TABNET_MODEL_FILE.exists():
    st.warning("⚡ Model artifacts not found. Run `python src/train.py` to extract global feature importances.")
    st.stop()

# 1. Global Feature Importance
st.subheader("🌐 Global Feature Importance")
st.caption("Aggregated across all training and test observations, showing which financial indicators TabNet relies upon most frequently.")

with open(FEATURE_IMPORTANCE_FILE, "r", encoding="utf-8") as f:
    imp_records = json.load(f)

df_imp = pd.DataFrame(imp_records).sort_values(by="importance", ascending=True)

fig_global = go.Figure(
    go.Bar(
        x=df_imp["importance_pct"],
        y=df_imp["feature"],
        orientation="h",
        marker=dict(
            color=df_imp["importance_pct"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="% Importance"),
        ),
        text=[f"{v:.1f}%" for v in df_imp["importance_pct"]],
        textposition="auto",
        hovertext=df_imp["description"],
    )
)
fig_global.update_layout(
    title="Global Feature Importance Ranking (Total = 100%)",
    xaxis_title="Relative Importance (%)",
    yaxis_title="Feature Identifier",
    height=550,
)
st.plotly_chart(style_plotly_chart(fig_global, height=550), use_container_width=True)

st.markdown("---")

# 2. Interactive Test-Sample Instance Inspector
st.subheader("🔎 Individual Applicant Attention Mask Explorer")
st.caption("Pick an applicant from the held-out test split to inspect their multi-step attention path.")

if TEST_DATA_CACHE.exists():
    df_test = pd.read_parquet(TEST_DATA_CACHE)
    model, preprocessor = load_artifacts()

    sample_idx = st.slider("Select Test Applicant Index:", min_value=0, max_value=min(len(df_test) - 1, 500), value=12, step=1)
    sample_row = df_test.iloc[[sample_idx]]

    from src.explain import explain_single_applicant
    exp = explain_single_applicant(model, preprocessor, sample_row)

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        render_metric_card("Applicant Index", f"#{sample_idx}", f"Ground Truth: {'Default' if sample_row['default_payment_next_month'].iloc[0] == 1 else 'Non-Default'}")
    with col_info2:
        render_metric_card("Predicted Default Prob", f"{exp['predicted_probability']:.1%}", exp["risk_category"])
    with col_info3:
        top_f = exp["top_features"][0]
        render_metric_card("Primary Salient Feature", top_f["feature"], f"{top_f['attention_pct']:.1f}% of total attention")

    # Step-wise Heatmap
    st.markdown("#### Sequential Decision Step Routing")
    step_df = exp["step_attention_df"]
    # Only keep features that had noticeable attention
    active_mask = step_df[step_df.sum(axis=1) > 0.02].sort_values(by="Step 1", ascending=False)

    fig_steps = px.imshow(
        active_mask,
        labels=dict(x="TabNet Decision Step", y="Feature", color="Attention"),
        color_continuous_scale="Viridis",
        aspect="auto",
    )
    fig_steps.update_layout(
        title=f"Applicant #{sample_idx}: Sparsity Mask Across Decision Steps (N_steps = 4)",
        height=400,
    )
    st.plotly_chart(style_plotly_chart(fig_steps, height=400), use_container_width=True)

    with st.expander("📄 View Raw Applicant Indicators"):
        st.dataframe(sample_row, use_container_width=True)
else:
    st.info("Test dataset cache not found. Run `python src/train.py` to cache test samples.")
