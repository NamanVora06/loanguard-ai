"""UI styling, theme configurations, and custom layout components for LoanGuard AI."""

import plotly.graph_objects as go
import streamlit as st

# Custom Fintech Color Palette
FINTECH_PALETTE = {
    "primary": "#0f766e",      # Deep Teal
    "primary_light": "#14b8a6",# Bright Teal
    "navy_dark": "#0a192f",    # Dark Navy
    "navy_card": "#1e293b",    # Slate Blue / Card BG
    "navy_border": "#334155",  # Border
    "text_main": "#f8fafc",    # Light slate
    "text_muted": "#94a3b8",   # Muted gray
    "accent_green": "#10b981", # Emerald
    "accent_amber": "#f59e0b", # Amber
    "accent_red": "#ef4444",   # Crimson
    "tabnet_color": "#06b6d4", # Cyan
    "baseline_color": "#8b5cf6",# Violet
    "rf_color": "#ec4899",     # Pink
}


def apply_fintech_theme() -> None:
    """Applies modern custom CSS to ensure a professional, polished fintech look."""
    st.markdown(
        """
        <style>
        /* Main background and font styling */
        .stApp {
            background-color: #0b1120;
            color: #f1f5f9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Metric card styling */
        .metric-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            margin-bottom: 16px;
        }
        .metric-card-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            margin-bottom: 6px;
            font-weight: 600;
        }
        .metric-card-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .metric-card-subtitle {
            font-size: 0.80rem;
            color: #14b8a6;
            margin-top: 4px;
        }

        /* Hero Banner */
        .hero-banner {
            background: linear-gradient(135deg, #042f2e 0%, #0f172a 100%);
            border: 1px solid #14b8a6;
            border-radius: 14px;
            padding: 28px;
            margin-bottom: 24px;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 8px;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: #99f6e4;
            line-height: 1.5;
        }

        /* Risk Badges */
        .risk-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .risk-low {
            background-color: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid #10b981;
        }
        .risk-medium {
            background-color: rgba(245, 158, 11, 0.15);
            color: #f59e0b;
            border: 1px solid #f59e0b;
        }
        .risk-high {
            background-color: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid #ef4444;
        }

        /* Disclaimer box */
        .disclaimer-box {
            background-color: #0f172a;
            border-left: 4px solid #f59e0b;
            padding: 14px 18px;
            border-radius: 6px;
            margin: 16px 0;
            font-size: 0.88rem;
            color: #cbd5e1;
        }

        /* Scientific explainability notice */
        .explain-box {
            background-color: #042f2e;
            border-left: 4px solid #14b8a6;
            padding: 14px 18px;
            border-radius: 6px;
            margin: 16px 0;
            font-size: 0.88rem;
            color: #ccfbf1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, subtitle: str = "") -> None:
    """Renders a custom HTML metric card with fintech styling."""
    sub_html = f'<div class="metric-card-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-title">{title}</div>
            <div class="metric-card-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plotly_chart(fig: go.Figure, height: int = 400) -> go.Figure:
    """Styles a Plotly figure to harmonize seamlessly with the dark navy fintech UI."""
    fig.update_layout(
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        font=dict(family="Segoe UI, sans-serif", color="#cbd5e1", size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        height=height,
        xaxis=dict(
            gridcolor="#1e293b",
            zerolinecolor="#334155",
            showline=True,
            linecolor="#334155",
        ),
        yaxis=dict(
            gridcolor="#1e293b",
            zerolinecolor="#334155",
            showline=True,
            linecolor="#334155",
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor="#334155",
            borderwidth=1,
        ),
    )
    return fig
