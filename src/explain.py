"""Explainability module for LoanGuard AI.

Leverages PyTorch-TabNet's native sequential attention masks (Entmax/Sparsemax)
to extract both global and instance-level feature attention across decision steps.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import (
    ALL_FEATURE_COLS,
    CATEGORICAL_MAPPINGS,
    DEFAULT_RISK_THRESHOLDS,
    FEATURE_DESCRIPTIONS,
)


def get_risk_category(
    probability: float,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[str, str]:
    """Maps a default probability to a descriptive risk tier and color code."""
    thresh = thresholds or DEFAULT_RISK_THRESHOLDS
    low_cut = thresh.get("low", 0.20)
    med_cut = thresh.get("medium", 0.50)

    if probability < low_cut:
        return "Low Risk", "#10b981"  # Emerald green
    elif probability < med_cut:
        return "Medium Risk", "#f59e0b"  # Amber orange
    else:
        return "High Risk", "#ef4444"  # Crimson red


def explain_single_applicant(
    model: Any,
    preprocessor: Any,
    applicant_data: Union[pd.DataFrame, Dict[str, Any]],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Generates a complete explainable prediction for an individual applicant.

    Extracts:
    1. Predicted default probability and binary decision.
    2. Configurable risk category (Low, Medium, High).
    3. Aggregated instance-level feature attention weights from TabNet explain().
    4. Step-wise attention masks across each TabNet decision step.
    5. Scientifically rigorous explainability notes (Attention vs Causality).
    """
    if isinstance(applicant_data, dict):
        df = pd.DataFrame([applicant_data])
    else:
        df = applicant_data.copy()

    # Preprocess applicant data using the exact fitted pipeline
    X = preprocessor.transform(df)
    feature_names = preprocessor.get_feature_names()

    # Model inference
    probs = model.predict_proba(X)[0]
    default_prob = float(probs[1])
    predicted_class = int(default_prob >= 0.50)
    risk_category, risk_color = get_risk_category(default_prob, thresholds)

    # Native TabNet Explainability: explain_matrix and step masks
    explain_matrix, masks = model.explain(X)
    instance_attention = explain_matrix[0]  # Shape: (num_features,)

    # Normalize attention so it sums to 1.0 (or 100%) for clean interpretation
    att_sum = instance_attention.sum()
    norm_attention = (instance_attention / att_sum) if att_sum > 0 else instance_attention

    # Build per-feature attention breakdown
    features_breakdown = []
    for idx, col in enumerate(feature_names):
        val = df[col].iloc[0] if col in df.columns else None
        att_score = float(norm_attention[idx])
        features_breakdown.append({
            "feature": col,
            "feature_label": FEATURE_DESCRIPTIONS.get(col, col),
            "raw_value": val,
            "attention_weight": att_score,
            "attention_pct": round(att_score * 100.0, 2),
        })

    # Sort features by highest attention weight
    features_breakdown = sorted(features_breakdown, key=lambda x: x["attention_weight"], reverse=True)

    # Step-wise attention breakdown
    step_masks = {}
    for step_idx, step_mask in masks.items():
        step_att = step_mask[0]
        step_sum = step_att.sum()
        norm_step = (step_att / step_sum) if step_sum > 0 else step_att
        step_masks[f"Step {step_idx + 1}"] = [float(s) for s in norm_step]

    step_df = pd.DataFrame(step_masks, index=feature_names)

    return {
        "predicted_probability": default_prob,
        "predicted_class": predicted_class,
        "risk_category": risk_category,
        "risk_color": risk_color,
        "features_breakdown": features_breakdown,
        "top_features": features_breakdown[:5],
        "step_attention_df": step_df,
        "feature_names": feature_names,
        "raw_applicant_values": df.iloc[0].to_dict(),
        "scientific_disclaimer": (
            "TabNet attention masks quantify the sparsity and relative focus the neural "
            "network placed on each feature during its decision steps. Attention weights "
            "indicate feature relevance in model routing; they do not establish direct "
            "causality, nor do they imply that a feature increased or decreased default odds."
        ),
    }


def extract_global_feature_importance(
    model: Any,
    feature_names: List[str],
) -> pd.DataFrame:
    """Extracts and formats global feature importances from a trained TabNet model."""
    importances = model.feature_importances_
    total = importances.sum()
    normalized = (importances / total) * 100.0 if total > 0 else importances

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
        "importance_pct": np.round(normalized, 2),
        "description": [FEATURE_DESCRIPTIONS.get(f, f) for f in feature_names],
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return df_imp
