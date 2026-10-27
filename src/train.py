"""Model training script for LoanGuard AI.

Trains the PyTorch TabNetClassifier with Entmax attention and early stopping,
alongside a Scikit-Learn Logistic Regression baseline on identical splits.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.linear_model import LogisticRegression

from config import (
    BASELINE_LR_FILE,
    DATASET_PATH,
    FEATURE_IMPORTANCE_FILE,
    MODELS_DIR,
    PREPROCESSOR_FILE,
    REPORTS_DIR,
    TABNET_MODEL_FILE,
    TABNET_PARAMS,
    TARGET_COL,
    TEST_DATA_CACHE,
    TRAINING_PARAMS,
)
from src.data_loader import load_or_fetch_dataset
from src.explain import extract_global_feature_importance
from src.preprocessing import TabNetDataPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across numpy and PyTorch for deterministic training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_models() -> Dict[str, any]:
    """Executes the complete training workflow for TabNet and the baseline model."""
    set_seed(TABNET_PARAMS["seed"])

    # 1. Load data
    logger.info("Loading dataset...")
    df = load_or_fetch_dataset()

    # 2. Preprocess & Stratified Split
    preprocessor = TabNetDataPreprocessor()
    df_train, df_val, df_test = preprocessor.split_data(df, train_size=0.70, val_size=0.15, test_size=0.15)

    # Cache held-out test set for evaluate.py and interactive dashboard testing
    logger.info(f"Caching held-out test set ({len(df_test)} samples) to {TEST_DATA_CACHE}...")
    df_test.to_parquet(TEST_DATA_CACHE, index=False)

    # Fit preprocessor strictly on training data
    X_train = preprocessor.fit_transform(df_train)
    y_train = df_train[TARGET_COL].values.astype(np.int64)

    X_val = preprocessor.transform(df_val)
    y_val = df_val[TARGET_COL].values.astype(np.int64)

    X_test = preprocessor.transform(df_test)
    y_test = df_test[TARGET_COL].values.astype(np.int64)

    # Save fitted preprocessor
    preprocessor.save(PREPROCESSOR_FILE)

    # 3. Configure and Train TabNetClassifier
    logger.info("Configuring TabNetClassifier...")
    tabnet = TabNetClassifier(
        n_d=TABNET_PARAMS["n_d"],
        n_a=TABNET_PARAMS["n_a"],
        n_steps=TABNET_PARAMS["n_steps"],
        gamma=TABNET_PARAMS["gamma"],
        lambda_sparse=TABNET_PARAMS["lambda_sparse"],
        cat_idxs=preprocessor.cat_idxs,
        cat_dims=preprocessor.cat_dims,
        cat_emb_dim=preprocessor.cat_emb_dims,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(
            lr=TRAINING_PARAMS["learning_rate"],
            weight_decay=TRAINING_PARAMS["weight_decay"],
        ),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 10, "gamma": 0.90},
        mask_type=TABNET_PARAMS["mask_type"],
        seed=TABNET_PARAMS["seed"],
        verbose=1,
    )

    logger.info("Starting TabNet training with early stopping on validation AUC...")
    # TabNet 'weights=1' balances class weights automatically based on training class frequencies
    tabnet.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=TRAINING_PARAMS["max_epochs"],
        patience=TRAINING_PARAMS["patience"],
        batch_size=TRAINING_PARAMS["batch_size"],
        virtual_batch_size=TRAINING_PARAMS["virtual_batch_size"],
        num_workers=TRAINING_PARAMS["num_workers"],
        drop_last=TRAINING_PARAMS["drop_last"],
        weights=1,
    )

    # TabNet save_model appends .zip automatically
    save_prefix = str(TABNET_MODEL_FILE).replace(".zip", "")
    tabnet.save_model(save_prefix)
    logger.info(f"TabNet model saved to {TABNET_MODEL_FILE}")

    # Extract & Save Global Feature Importance
    feature_names = preprocessor.get_feature_names()
    global_imp_df = extract_global_feature_importance(tabnet, feature_names)
    global_imp_df.to_json(FEATURE_IMPORTANCE_FILE, orient="records", indent=2)
    logger.info(f"Global feature importance saved to {FEATURE_IMPORTANCE_FILE}")

    # 4. Train Scikit-Learn Logistic Regression & Random Forest Baselines
    logger.info("Training Scikit-Learn Logistic Regression baseline on identical split...")
    lr_baseline = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=TABNET_PARAMS["seed"],
    )
    lr_baseline.fit(X_train, y_train)
    joblib.dump(lr_baseline, BASELINE_LR_FILE)
    logger.info(f"Baseline Logistic Regression saved to {BASELINE_LR_FILE}")

    logger.info("Training Scikit-Learn Random Forest baseline on identical split...")
    from sklearn.ensemble import RandomForestClassifier
    rf_baseline = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight="balanced",
        random_state=TABNET_PARAMS["seed"],
        n_jobs=-1,
    )
    rf_baseline.fit(X_train, y_train)
    rf_file = MODELS_DIR / "baseline_rf.joblib"
    joblib.dump(rf_baseline, rf_file)
    logger.info(f"Baseline Random Forest saved to {rf_file}")

    return {
        "status": "success",
        "tabnet_model_path": str(TABNET_MODEL_FILE),
        "preprocessor_path": str(PREPROCESSOR_FILE),
        "baseline_lr_path": str(BASELINE_LR_FILE),
        "baseline_rf_path": str(rf_file),
        "test_cache_path": str(TEST_DATA_CACHE),
    }


if __name__ == "__main__":
    train_models()
