"""Page 4: Dataset Explorer & Demographic Distribution Analysis."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    CATEGORICAL_MAPPINGS,
    DATASET_PATH,
    FEATURE_DESCRIPTIONS,
    TARGET_COL,
)
from src.data_loader import load_or_fetch_dataset
from src.ui_components import (
    FINTECH_PALETTE,
    apply_fintech_theme,
    render_metric_card,
    style_plotly_chart,
)

st.set_page_config(
    page_title="Dataset Explorer | LoanGuard AI",
    page_icon="📁",
    layout="wide",
)
apply_fintech_theme()

st.title("📁 Credit Portfolio Dataset Explorer")
st.caption("Inspect population demographics, repayment behavior, statement distributions, and data integrity.")

# Privacy notice
st.markdown(
    """
    <div class="explain-box">
        🔒 <strong>Data Privacy & Compliance:</strong> This dataset contains no Personally Identifiable Information (PII) 
        such as names, addresses, social security numbers, or account IDs. It adheres to open research standards from the 
        UCI Machine Learning Repository.
    </div>
    """,
    unsafe_allow_html=True,
)

# Load data with caching
@st.cache_data
def get_cached_data() -> pd.DataFrame:
    return load_or_fetch_dataset()

df = get_cached_data()

# Summary Metrics
tot_records = len(df)
default_count = int(df[TARGET_COL].sum())
non_default_count = tot_records - default_count
default_rate = default_count / tot_records

col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric_card("Total Observations", f"{tot_records:,}", "23 predictive indicators")
with col2:
    render_metric_card("Default Rate", f"{default_rate:.1%}", f"{default_count:,} defaulting accounts")
with col3:
    render_metric_card("Non-Default Accounts", f"{non_default_count:,}", f"{(1 - default_rate):.1%} of portfolio")
with col4:
    missing_count = int(df.isnull().sum().sum())
    render_metric_card("Missing Values", f"{missing_count:,}", "Clean dataset / pre-imputed")

st.markdown("---")

# 1. Target Class Balance & Demographics
col_pie, col_demo = st.columns([1, 1])

with col_pie:
    st.subheader("⚖️ Target Class Balance")
    fig_donut = go.Figure(
        data=[
            go.Pie(
                labels=["Non-Default (0)", "Default (1)"],
                values=[non_default_count, default_count],
                hole=0.55,
                marker=dict(colors=["#10b981", "#ef4444"]),
                textinfo="label+percent",
            )
        ]
    )
    fig_donut.update_layout(title="Portfolio Default vs Non-Default Split", height=320)
    st.plotly_chart(style_plotly_chart(fig_donut, height=320), use_container_width=True)

with col_demo:
    st.subheader("🎓 Education Level Breakdown")
    edu_map = CATEGORICAL_MAPPINGS["EDUCATION"]
    edu_counts = df["EDUCATION"].map(lambda x: edu_map.get(x, "Others")).value_counts().reset_index()
    edu_counts.columns = ["Education", "Count"]

    fig_edu = px.bar(
        edu_counts,
        x="Education",
        y="Count",
        color="Education",
        color_discrete_sequence=px.colors.qualitative.Prism,
    )
    fig_edu.update_layout(title="Distribution by Education Level", height=320, showlegend=False)
    st.plotly_chart(style_plotly_chart(fig_edu, height=320), use_container_width=True)

st.markdown("---")

# 2. Numerical Feature Distributions
st.subheader("📊 Financial Feature Distributions")
num_choice = st.selectbox(
    "Select Feature to Inspect:",
    [
        "LIMIT_BAL",
        "AGE",
        "PAY_0",
        "BILL_AMT1",
        "PAY_AMT1",
    ],
    format_func=lambda x: f"{x} - {FEATURE_DESCRIPTIONS.get(x, x)}",
)

col_hist, col_box = st.columns(2)

with col_hist:
    fig_hist = px.histogram(
        df,
        x=num_choice,
        color=TARGET_COL,
        barmode="overlay",
        color_discrete_map={0: "#10b981", 1: "#ef4444"},
        nbins=40,
        title=f"Histogram of {num_choice} by Default Status",
    )
    st.plotly_chart(style_plotly_chart(fig_hist), use_container_width=True)

with col_box:
    fig_box = px.box(
        df,
        x=TARGET_COL,
        y=num_choice,
        color=TARGET_COL,
        color_discrete_map={0: "#10b981", 1: "#ef4444"},
        title=f"Boxplot of {num_choice} vs Default Status",
    )
    st.plotly_chart(style_plotly_chart(fig_box), use_container_width=True)

st.markdown("---")

# 3. Interactive Data Table Sample
st.subheader("📋 Dataset Preview (First 50 Rows)")
st.dataframe(df.head(50), use_container_width=True)

# Data Schema Reference
with st.expander("📖 Feature Dictionary & Schema Definitions"):
    schema_rows = []
    for col in df.columns:
        schema_rows.append({
            "Column Name": col,
            "Data Type": str(df[col].dtype),
            "Description": FEATURE_DESCRIPTIONS.get(col, "Target indicator (0=Non-Default, 1=Default)" if col == TARGET_COL else col),
            "Non-Null Count": int(df[col].notnull().sum()),
            "Sample Value": str(df[col].iloc[0]),
        })
    st.dataframe(pd.DataFrame(schema_rows), hide_index=True, use_container_width=True)
