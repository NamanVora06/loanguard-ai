"""Page 1: Applicant Default Prediction & Instant Explanation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    CATEGORICAL_MAPPINGS,
    DEFAULT_RISK_THRESHOLDS,
    FEATURE_DESCRIPTIONS,
    PREPROCESSOR_FILE,
    TABNET_MODEL_FILE,
)
from src.predict import load_artifacts, predict_applicant
from src.ui_components import (
    FINTECH_PALETTE,
    apply_fintech_theme,
    render_metric_card,
    style_plotly_chart,
)

# Page settings
st.set_page_config(
    page_title="Applicant Prediction | LoanGuard AI",
    page_icon="👤",
    layout="wide",
)
apply_fintech_theme()

st.title("👤 Applicant Loan Default Risk Predictor")
st.caption("Enter applicant financial indicators or select a benchmark persona to generate an explainable default assessment.")

# Model availability check
if not TABNET_MODEL_FILE.exists() or not PREPROCESSOR_FILE.exists():
    st.error(
        "⚠️ Trained TabNet model or preprocessor artifacts are not found. "
        "Please train the model first by executing `python src/train.py` from your terminal."
    )
    st.stop()

# Sidebar: Risk Threshold Configuration
with st.sidebar:
    st.subheader("⚙️ Risk Threshold Policy")
    st.caption("Configure credit policy cutoff thresholds for default probability.")
    low_cutoff = st.slider("Low-to-Medium Risk Cutoff", min_value=0.05, max_value=0.40, value=0.20, step=0.01)
    high_cutoff = st.slider("Medium-to-High Risk Cutoff", min_value=0.30, max_value=0.80, value=0.50, step=0.01)
    thresholds = {"low": low_cutoff, "medium": high_cutoff}

    st.markdown("---")
    st.subheader("📋 Benchmark Persona Presets")
    preset_choice = st.selectbox(
        "Load Sample Applicant Profile:",
        [
            "Custom Manual Input",
            "Persona A: Prime Low-Risk Borrower",
            "Persona B: Moderate Risk Revolver",
            "Persona C: High-Risk Delinquent Applicant",
        ],
    )

# Predefined Preset Values
presets = {
    "Persona A: Prime Low-Risk Borrower": {
        "LIMIT_BAL": 350000.0,
        "SEX": 2,
        "EDUCATION": 1,
        "MARRIAGE": 2,
        "AGE": 38,
        "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
        "BILL_AMT1": 22000.0, "BILL_AMT2": 19500.0, "BILL_AMT3": 17800.0,
        "BILL_AMT4": 25000.0, "BILL_AMT5": 21000.0, "BILL_AMT6": 18000.0,
        "PAY_AMT1": 22000.0, "PAY_AMT2": 19500.0, "PAY_AMT3": 17800.0,
        "PAY_AMT4": 25000.0, "PAY_AMT5": 21000.0, "PAY_AMT6": 18000.0,
    },
    "Persona B: Moderate Risk Revolver": {
        "LIMIT_BAL": 120000.0,
        "SEX": 1,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 29,
        "PAY_0": 1, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 85000.0, "BILL_AMT2": 82000.0, "BILL_AMT3": 79000.0,
        "BILL_AMT4": 74000.0, "BILL_AMT5": 69000.0, "BILL_AMT6": 65000.0,
        "PAY_AMT1": 3500.0, "PAY_AMT2": 4000.0, "PAY_AMT3": 3500.0,
        "PAY_AMT4": 3000.0, "PAY_AMT5": 3000.0, "PAY_AMT6": 3000.0,
    },
    "Persona C: High-Risk Delinquent Applicant": {
        "LIMIT_BAL": 50000.0,
        "SEX": 1,
        "EDUCATION": 3,
        "MARRIAGE": 2,
        "AGE": 24,
        "PAY_0": 2, "PAY_2": 2, "PAY_3": 2, "PAY_4": 2, "PAY_5": 2, "PAY_6": 2,
        "BILL_AMT1": 49000.0, "BILL_AMT2": 48500.0, "BILL_AMT3": 47200.0,
        "BILL_AMT4": 46000.0, "BILL_AMT5": 45000.0, "BILL_AMT6": 44000.0,
        "PAY_AMT1": 1000.0, "PAY_AMT2": 1000.0, "PAY_AMT3": 1000.0,
        "PAY_AMT4": 0.0, "PAY_AMT5": 1000.0, "PAY_AMT6": 0.0,
    },
}

active_vals = presets.get(preset_choice, {})

# Input Form
with st.form("applicant_input_form"):
    st.subheader("1. Applicant Demographics & Credit Line")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        limit_bal = st.number_input(
            "Credit Limit (NT$)",
            min_value=10000.0,
            max_value=2000000.0,
            value=float(active_vals.get("LIMIT_BAL", 150000.0)),
            step=10000.0,
            help="Total amount of credit extended to applicant.",
        )
    with col2:
        age = st.number_input(
            "Age (Years)",
            min_value=18,
            max_value=90,
            value=int(active_vals.get("AGE", 32)),
            step=1,
        )
    with col3:
        sex_options = {1: "Male", 2: "Female"}
        default_sex = int(active_vals.get("SEX", 1))
        sex = st.selectbox(
            "Gender",
            options=list(sex_options.keys()),
            format_func=lambda x: sex_options[x],
            index=0 if default_sex == 1 else 1,
        )
    with col4:
        edu_options = {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}
        default_edu = int(active_vals.get("EDUCATION", 2))
        edu_idx = list(edu_options.keys()).index(default_edu) if default_edu in edu_options else 1
        education = st.selectbox(
            "Education Level",
            options=list(edu_options.keys()),
            format_func=lambda x: edu_options[x],
            index=edu_idx,
        )
    with col5:
        marr_options = {1: "Married", 2: "Single", 3: "Others / Divorced"}
        default_marr = int(active_vals.get("MARRIAGE", 2))
        marr_idx = list(marr_options.keys()).index(default_marr) if default_marr in marr_options else 1
        marriage = st.selectbox(
            "Marital Status",
            options=list(marr_options.keys()),
            format_func=lambda x: marr_options[x],
            index=marr_idx,
        )

    st.markdown("---")
    st.subheader("2. Repayment Status History (Past 6 Months)")
    st.caption("Status codes: -2=Inactive, -1=Paid in full, 0=Revolving credit, 1=Delay 1 month, 2=Delay 2 months, 3+=Delay 3+ months")

    p_cols = st.columns(6)
    pay_status_keys = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    pay_inputs = {}

    months = [
        ("PAY_0", "Month 1 (Sept)"),
        ("PAY_2", "Month 2 (Aug)"),
        ("PAY_3", "Month 3 (July)"),
        ("PAY_4", "Month 4 (June)"),
        ("PAY_5", "Month 5 (May)"),
        ("PAY_6", "Month 6 (April)"),
    ]

    for idx, (col_name, label) in enumerate(months):
        with p_cols[idx]:
            def_val = int(active_vals.get(col_name, 0))
            if def_val not in pay_status_keys:
                def_val = 0
            val_idx = pay_status_keys.index(def_val)
            pay_inputs[col_name] = st.selectbox(
                label,
                options=pay_status_keys,
                index=val_idx,
                key=f"input_{col_name}",
            )

    st.markdown("---")
    st.subheader("3. Bill Statement Balances & Repayments (NT$)")

    b_cols = st.columns(6)
    bill_inputs = {}
    for i in range(1, 7):
        with b_cols[i - 1]:
            b_key = f"BILL_AMT{i}"
            bill_inputs[b_key] = st.number_input(
                f"Bill M-{i} (NT$)",
                min_value=-50000.0,
                max_value=1500000.0,
                value=float(active_vals.get(b_key, 25000.0)),
                step=5000.0,
            )

    pay_cols = st.columns(6)
    pay_amt_inputs = {}
    for i in range(1, 7):
        with pay_cols[i - 1]:
            p_key = f"PAY_AMT{i}"
            pay_amt_inputs[p_key] = st.number_input(
                f"Paid M-{i} (NT$)",
                min_value=0.0,
                max_value=1000000.0,
                value=float(active_vals.get(p_key, 5000.0)),
                step=1000.0,
            )

    submitted = st.form_submit_button("🔍 Run TabNet Prediction & Explanation", use_container_width=True)

# Process Inference upon form submission
if submitted or preset_choice != "Custom Manual Input":
    applicant_data = {
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        **pay_inputs,
        **bill_inputs,
        **pay_amt_inputs,
    }

    with st.spinner("Executing TabNet neural forward pass and extracting sequential attention masks..."):
        model, preprocessor = load_artifacts()
        result = predict_applicant(
            applicant_dict=applicant_data,
            thresholds=thresholds,
            model=model,
            preprocessor=preprocessor,
        )

    prob = result["predicted_probability"]
    risk_cat = result["risk_category"]
    risk_col = result["risk_color"]
    pred_cls = result["predicted_class"]

    st.markdown("---")
    st.subheader("🎯 Assessment Results")

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Default Risk Probability</div>
                <div class="metric-card-value" style="color: {risk_col};">{prob:.1%}</div>
                <div class="metric-card-subtitle">Likelihood of next-month default</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_col2:
        badge_class = "risk-low" if "Low" in risk_cat else ("risk-medium" if "Medium" in risk_cat else "risk-high")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Assigned Risk Tier</div>
                <div style="margin-top: 10px;">
                    <span class="risk-badge {badge_class}">{risk_cat}</span>
                </div>
                <div class="metric-card-subtitle" style="margin-top: 14px;">Policy cutoffs: [{low_cutoff:.0%}, {high_cutoff:.0%}]</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_col3:
        cls_text = "Default Anticipated (1)" if pred_cls == 1 else "Non-Default (0)"
        render_metric_card("Standard Binary Output", cls_text, "Evaluated at standard 50% cutoff")

    # Instance-Level Explainability
    st.markdown("---")
    st.subheader("🔍 Model Explainability: Feature Attention Allocation")

    st.markdown(
        f"""
        <div class="explain-box">
            <strong>Scientific Explainability Note:</strong> {result['scientific_disclaimer']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    exp_col1, exp_col2 = st.columns([3, 2])

    with exp_col1:
        st.markdown("#### Top Features Receiving Model Attention")
        breakdown = result["features_breakdown"]
        top_10 = breakdown[:10]

        df_bar = pd.DataFrame(top_10).sort_values(by="attention_weight", ascending=True)

        fig_att = go.Figure(
            go.Bar(
                x=df_bar["attention_pct"],
                y=df_bar["feature"],
                orientation="h",
                marker=dict(
                    color=df_bar["attention_pct"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="% Attention"),
                ),
                text=[f"{v:.1f}%" for v in df_bar["attention_pct"]],
                textposition="auto",
                hovertext=df_bar["feature_label"],
            )
        )
        fig_att.update_layout(
            title="Individual Feature Attention Weights (Normalized %)",
            xaxis_title="Attention Share (%)",
            yaxis_title="Feature",
        )
        st.plotly_chart(style_plotly_chart(fig_att, height=380), use_container_width=True)

    with exp_col2:
        st.markdown("#### Detailed Attention & Raw Values")
        table_rows = []
        for item in breakdown[:7]:
            table_rows.append({
                "Feature": item["feature"],
                "Input Value": item["raw_value"],
                "Attention %": f"{item['attention_pct']:.1f}%",
                "Description": item["feature_label"],
            })
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

    # Decision Steps Heatmap
    with st.expander("🔬 View Attention Masks Across TabNet Decision Steps"):
        st.caption("TabNet's attentive transformers route information over sequential steps. See which features were prioritized at each reasoning phase.")
        step_df = result["step_attention_df"]
        # Filter to features that received non-zero attention in at least one step
        active_steps = step_df[step_df.sum(axis=1) > 0.01].sort_values(by="Step 1", ascending=False)

        fig_hm = px.imshow(
            active_steps,
            labels=dict(x="Decision Step", y="Feature", color="Attention"),
            color_continuous_scale="Teal",
            aspect="auto",
        )
        st.plotly_chart(style_plotly_chart(fig_hm, height=350), use_container_width=True)
