"""Preprocessing pipeline for LoanGuard AI.

Implements leakage-safe data transformation, stratified train/val/test splits,
integer encoding for TabNet categorical features with embedding metadata,
and robust numerical feature scaling.
"""

from dataclasses import dataclass, field
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    ALL_FEATURE_COLS,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    PREPROCESSOR_FILE,
    TARGET_COL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PreprocessorArtifacts:
    """Stores fitted preprocessing state to ensure 100% reproducible inference."""
    categorical_cols: List[str]
    numerical_cols: List[str]
    all_feature_cols: List[str]
    cat_idxs: List[int]
    cat_dims: List[int]
    cat_emb_dims: List[int]
    category_encoders: Dict[str, Dict[any, int]]  # original value -> integer index
    scaler: StandardScaler
    numerical_medians: Dict[str, float]
    categorical_modes: Dict[str, any]


class TabNetDataPreprocessor:
    """Leakage-safe preprocessor tailored for PyTorch-TabNet and baseline models."""

    def __init__(
        self,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
    ):
        self.categorical_cols = categorical_cols or CATEGORICAL_COLS
        self.numerical_cols = numerical_cols or NUMERICAL_COLS
        # Strict order: categoricals first, then numericals
        self.all_feature_cols = self.categorical_cols + self.numerical_cols

        # Categorical column positions in X
        self.cat_idxs = list(range(len(self.categorical_cols)))
        self.cat_dims: List[int] = []
        self.cat_emb_dims: List[int] = []

        self.category_encoders: Dict[str, Dict[any, int]] = {}
        self.scaler = StandardScaler()
        self.numerical_medians: Dict[str, float] = {}
        self.categorical_modes: Dict[str, any] = {}
        self.is_fitted = False

    def split_data(
        self,
        df: pd.DataFrame,
        train_size: float = 0.70,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Performs a stratified split into train, validation, and test sets.

        Ensures no data leakage occurs between splits.
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-5, "Split fractions must sum to 1.0"

        X = df.drop(columns=[TARGET_COL]) if TARGET_COL in df.columns else df
        y = df[TARGET_COL] if TARGET_COL in df.columns else None

        # Step 1: Split into train and temp (val + test)
        temp_size = val_size + test_size
        df_train, df_temp = train_test_split(
            df,
            test_size=temp_size,
            random_state=random_state,
            stratify=y,
        )

        # Step 2: Split temp into val and test
        relative_test_size = test_size / temp_size
        df_val, df_test = train_test_split(
            df_temp,
            test_size=relative_test_size,
            random_state=random_state,
            stratify=df_temp[TARGET_COL] if y is not None else None,
        )

        logger.info(
            f"Stratified data split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}"
        )
        return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

    def fit(self, df_train: pd.DataFrame) -> "TabNetDataPreprocessor":
        """Fits imputer statistics, category encodings, and numerical scaler on train data ONLY."""
        logger.info("Fitting TabNetDataPreprocessor on training data...")

        # 1. Compute missing value fallbacks
        for col in self.categorical_cols:
            mode_series = df_train[col].dropna().mode()
            self.categorical_modes[col] = mode_series.iloc[0] if not mode_series.empty else 1

        for col in self.numerical_cols:
            self.numerical_medians[col] = float(df_train[col].dropna().median() if not df_train[col].empty else 0.0)

        # 2. Build Category Encoders (mapped to contiguous integers 0, 1, ... K-1)
        self.cat_dims = []
        self.cat_emb_dims = []
        for col in self.categorical_cols:
            clean_series = df_train[col].fillna(self.categorical_modes[col])
            unique_vals = sorted(clean_series.unique().tolist())
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            self.category_encoders[col] = mapping

            cardinality = len(unique_vals)
            self.cat_dims.append(cardinality)
            # Embedding dimension heuristic: min(50, max(2, (cardinality + 1) // 2))
            emb_dim = max(1, min(16, int(np.ceil(np.log2(cardinality + 1)))))
            self.cat_emb_dims.append(emb_dim)

        # 3. Fit numerical scaler on training set
        num_data = df_train[self.numerical_cols].fillna(self.numerical_medians)
        self.scaler.fit(num_data)

        self.is_fitted = True
        logger.info(
            f"Preprocessor fitted successfully. Categorical dims: {self.cat_dims}, "
            f"Embedding dims: {self.cat_emb_dims}"
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms input DataFrame into the exact 2D numpy array required by TabNet.

        Columns are strictly ordered: [Categoricals (integer-encoded)] + [Numericals (standard-scaled)]
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted. Call fit() first or load() a saved preprocessor.")

        df = df.copy()

        # Handle missing columns or reordering
        for col in self.all_feature_cols:
            if col not in df.columns:
                if col in self.categorical_cols:
                    df[col] = self.categorical_modes[col]
                else:
                    df[col] = self.numerical_medians[col]

        # 1. Transform Categorical columns to contiguous integer indices
        cat_transformed = []
        for col in self.categorical_cols:
            series = df[col].fillna(self.categorical_modes[col])
            encoder = self.category_encoders[col]
            # Handle unseen categories gracefully by falling back to 0 or mode index
            default_idx = encoder.get(self.categorical_modes[col], 0)
            encoded = series.map(lambda v: encoder.get(v, default_idx)).values.astype(np.int64)
            cat_transformed.append(encoded)

        cat_matrix = np.column_stack(cat_transformed) if cat_transformed else np.empty((len(df), 0))

        # 2. Transform Numerical columns (impute + scale)
        num_clean = df[self.numerical_cols].fillna(self.numerical_medians)
        num_scaled = self.scaler.transform(num_clean).astype(np.float32)

        # 3. Stack into single array: [Categoricals, Numericals]
        X = np.hstack([cat_matrix, num_scaled])
        return X

    def fit_transform(self, df_train: pd.DataFrame) -> np.ndarray:
        """Fits preprocessor and returns transformed training feature matrix."""
        return self.fit(df_train).transform(df_train)

    def get_feature_names(self) -> List[str]:
        """Returns the exact ordered feature names corresponding to columns of X."""
        return list(self.all_feature_cols)

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Saves fitted preprocessor artifacts to disk using joblib."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted preprocessor.")
        save_path = Path(filepath or PREPROCESSOR_FILE)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = PreprocessorArtifacts(
            categorical_cols=self.categorical_cols,
            numerical_cols=self.numerical_cols,
            all_feature_cols=self.all_feature_cols,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dims=self.cat_emb_dims,
            category_encoders=self.category_encoders,
            scaler=self.scaler,
            numerical_medians=self.numerical_medians,
            categorical_modes=self.categorical_modes,
        )
        joblib.dump(state, save_path)
        logger.info(f"Preprocessor artifacts successfully saved to {save_path}")
        return save_path

    @classmethod
    def load(cls, filepath: Optional[Union[str, Path]] = None) -> "TabNetDataPreprocessor":
        """Loads preprocessor artifacts from disk."""
        load_path = Path(filepath or PREPROCESSOR_FILE)
        if not load_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found at {load_path}")

        state: PreprocessorArtifacts = joblib.load(load_path)
        preprocessor = cls(
            categorical_cols=state.categorical_cols,
            numerical_cols=state.numerical_cols,
        )
        preprocessor.all_feature_cols = state.all_feature_cols
        preprocessor.cat_idxs = state.cat_idxs
        preprocessor.cat_dims = state.cat_dims
        preprocessor.cat_emb_dims = state.cat_emb_dims
        preprocessor.category_encoders = state.category_encoders
        preprocessor.scaler = state.scaler
        preprocessor.numerical_medians = state.numerical_medians
        preprocessor.categorical_modes = state.categorical_modes
        preprocessor.is_fitted = True
        logger.info(f"Loaded preprocessor from {load_path}")
        return preprocessor
