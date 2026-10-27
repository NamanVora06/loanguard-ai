"""Inference and prediction pipeline for LoanGuard AI.

Provides single-applicant and batch loan default prediction with
runtime validation, probability calibration, and explainability integration.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier

from config import (
    ALL_FEATURE_COLS,
    PREPROCESSOR_FILE,
    TABNET_MODEL_FILE,
)
from src.explain import explain_single_applicant
from src.preprocessing import TabNetDataPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Cached model & preprocessor instances for high-performance inference
_CACHED_MODEL: Optional[TabNetClassifier] = None
_CACHED_PREPROCESSOR: Optional[TabNetDataPreprocessor] = None


def load_artifacts(
    model_path: Optional[Union[str, Path]] = None,
    preprocessor_path: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
) -> Tuple[TabNetClassifier, TabNetDataPreprocessor]:
    """Loads and caches the TabNet model and preprocessor artifacts."""
    global _CACHED_MODEL, _CACHED_PREPROCESSOR

    if _CACHED_MODEL is not None and _CACHED_PREPROCESSOR is not None and not force_reload:
        return _CACHED_MODEL, _CACHED_PREPROCESSOR

    m_path = Path(model_path or TABNET_MODEL_FILE)
    p_path = Path(preprocessor_path or PREPROCESSOR_FILE)

    if not m_path.exists():
        raise FileNotFoundError(
            f"Trained TabNet model not found at '{m_path}'. Please run 'python src/train.py' first."
        )
    if not p_path.exists():
        raise FileNotFoundError(
            f"Preprocessor artifact not found at '{p_path}'. Please run 'python src/train.py' first."
        )

    logger.info(f"Loading preprocessor from {p_path}...")
    preprocessor = TabNetDataPreprocessor.load(p_path)

    logger.info(f"Loading TabNet model from {m_path}...")
    model = TabNetClassifier()
    model.load_model(str(m_path))

    _CACHED_MODEL = model
    _CACHED_PREPROCESSOR = preprocessor
    return model, preprocessor


def predict_applicant(
    applicant_dict: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
    model: Optional[TabNetClassifier] = None,
    preprocessor: Optional[TabNetDataPreprocessor] = None,
) -> Dict[str, Any]:
    """Runs complete end-to-end inference and explainability for a single loan applicant."""
    if model is None or preprocessor is None:
        model, preprocessor = load_artifacts()

    # Create 1-row DataFrame
    applicant_df = pd.DataFrame([applicant_dict])

    # Run explanation and inference
    result = explain_single_applicant(
        model=model,
        preprocessor=preprocessor,
        applicant_data=applicant_df,
        thresholds=thresholds,
    )
    return result


def predict_batch(
    df: pd.DataFrame,
    model: Optional[TabNetClassifier] = None,
    preprocessor: Optional[TabNetDataPreprocessor] = None,
) -> pd.DataFrame:
    """Predicts default probability and class for a batch of applicants."""
    if model is None or preprocessor is None:
        model, preprocessor = load_artifacts()

    X = preprocessor.transform(df)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    result_df = df.copy()
    result_df["default_probability"] = probs
    result_df["predicted_default"] = preds
    return result_df
