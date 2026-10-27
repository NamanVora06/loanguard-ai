"""Unit tests for LoanGuard AI TabNet inference, predictions, and explainability."""

import numpy as np
import pandas as pd
import pytest
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

from config import ALL_FEATURE_COLS, TARGET_COL
from src.data_loader import generate_realistic_synthetic_data
from src.explain import explain_single_applicant, get_risk_category
from src.predict import predict_applicant
from src.preprocessing import TabNetDataPreprocessor


@pytest.fixture(scope="module")
def mini_trained_model():
    """Trains a lightweight TabNet model on a small dataset for fast testing."""
    df = generate_realistic_synthetic_data(n_samples=300, random_state=42)
    preprocessor = TabNetDataPreprocessor()
    df_train, df_val, _ = preprocessor.split_data(df, train_size=0.70, val_size=0.15, test_size=0.15)

    X_train = preprocessor.fit_transform(df_train)
    y_train = df_train[TARGET_COL].values.astype(np.int64)
    X_val = preprocessor.transform(df_val)
    y_val = df_val[TARGET_COL].values.astype(np.int64)

    tabnet = TabNetClassifier(
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.2,
        cat_idxs=preprocessor.cat_idxs,
        cat_dims=preprocessor.cat_dims,
        cat_emb_dim=[1, 1, 1],
        mask_type="entmax",
        seed=42,
        verbose=0,
    )
    tabnet.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_val, y_val)],
        max_epochs=2,
        batch_size=64,
        virtual_batch_size=32,
    )
    return tabnet, preprocessor


def test_risk_category_mapping():
    """Verifies probability threshold mapping to descriptive risk categories."""
    assert get_risk_category(0.05)[0] == "Low Risk"
    assert get_risk_category(0.19)[0] == "Low Risk"
    assert get_risk_category(0.20)[0] == "Medium Risk"
    assert get_risk_category(0.49)[0] == "Medium Risk"
    assert get_risk_category(0.50)[0] == "High Risk"
    assert get_risk_category(0.95)[0] == "High Risk"


def test_single_applicant_prediction(mini_trained_model):
    """Verifies that individual prediction returns valid probabilities and schema."""
    model, preprocessor = mini_trained_model

    applicant_data = {
        "LIMIT_BAL": 100000.0,
        "SEX": 1,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 30,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 20000.0, "BILL_AMT2": 19000.0, "BILL_AMT3": 18000.0,
        "BILL_AMT4": 17000.0, "BILL_AMT5": 16000.0, "BILL_AMT6": 15000.0,
        "PAY_AMT1": 2000.0, "PAY_AMT2": 2000.0, "PAY_AMT3": 2000.0,
        "PAY_AMT4": 2000.0, "PAY_AMT5": 2000.0, "PAY_AMT6": 2000.0,
    }

    result = predict_applicant(
        applicant_dict=applicant_data,
        model=model,
        preprocessor=preprocessor,
    )

    # 1. Output probability bounds
    prob = result["predicted_probability"]
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0

    # 2. Predicted binary class
    pred_cls = result["predicted_class"]
    assert pred_cls in [0, 1]

    # 3. Risk category
    assert result["risk_category"] in ["Low Risk", "Medium Risk", "High Risk"]

    # 4. Feature attention breakdown
    breakdown = result["features_breakdown"]
    assert len(breakdown) == len(ALL_FEATURE_COLS)

    total_att_pct = sum(item["attention_pct"] for item in breakdown)
    # Sum of normalized percentages should be ~100%
    assert abs(total_att_pct - 100.0) < 1.0

    # 5. Step-wise attention matrix
    step_df = result["step_attention_df"]
    assert step_df.shape[0] == len(ALL_FEATURE_COLS)
    assert step_df.shape[1] == 3  # n_steps = 3 in fixture
