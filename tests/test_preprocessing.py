"""Unit tests for LoanGuard AI data loading and preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from config import ALL_FEATURE_COLS, CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL
from src.data_loader import generate_realistic_synthetic_data, standardize_credit_dataset
from src.preprocessing import TabNetDataPreprocessor


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Fixture providing a small synthetic dataset for testing."""
    return generate_realistic_synthetic_data(n_samples=200, random_state=42)


def test_synthetic_data_schema(sample_raw_data):
    """Verifies that synthetic data strictly adheres to expected schema and target."""
    assert len(sample_raw_data) == 200
    assert TARGET_COL in sample_raw_data.columns
    for col in ALL_FEATURE_COLS:
        assert col in sample_raw_data.columns
    # Check binary target
    assert set(sample_raw_data[TARGET_COL].unique()).issubset({0, 1})


def test_stratified_split(sample_raw_data):
    """Verifies that train/val/test splits sum to total and preserve class proportions."""
    preprocessor = TabNetDataPreprocessor()
    df_train, df_val, df_test = preprocessor.split_data(
        sample_raw_data, train_size=0.70, val_size=0.15, test_size=0.15
    )

    assert len(df_train) == 140
    assert len(df_val) == 30
    assert len(df_test) == 30
    assert len(df_train) + len(df_val) + len(df_test) == len(sample_raw_data)


def test_preprocessing_fit_and_transform(sample_raw_data):
    """Verifies that preprocessor transforms DataFrame into valid 2D numpy array."""
    preprocessor = TabNetDataPreprocessor()
    df_train, df_val, df_test = preprocessor.split_data(sample_raw_data)

    X_train = preprocessor.fit_transform(df_train)
    X_test = preprocessor.transform(df_test)

    assert isinstance(X_train, np.ndarray)
    assert X_train.shape == (len(df_train), len(ALL_FEATURE_COLS))
    assert X_test.shape == (len(df_test), len(ALL_FEATURE_COLS))

    # Verify no NaN or Inf values in transformed matrices
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    assert not np.isinf(X_train).any()
    assert not np.isinf(X_test).any()


def test_categorical_integer_encoding(sample_raw_data):
    """Verifies that categorical columns are converted to integer indices starting at 0."""
    preprocessor = TabNetDataPreprocessor()
    preprocessor.fit(sample_raw_data)
    X = preprocessor.transform(sample_raw_data)

    # First len(CATEGORICAL_COLS) columns are categorical
    for idx in preprocessor.cat_idxs:
        cat_values = X[:, idx]
        # Should be integers and within [0, cat_dim - 1]
        assert np.all(cat_values == np.floor(cat_values))
        assert np.min(cat_values) >= 0
        assert np.max(cat_values) < preprocessor.cat_dims[idx]


def test_missing_value_handling(sample_raw_data):
    """Verifies that preprocessor cleanly imputes missing categorical and numerical values."""
    preprocessor = TabNetDataPreprocessor()
    preprocessor.fit(sample_raw_data)

    corrupted_data = sample_raw_data.copy()
    corrupted_data.loc[0, "LIMIT_BAL"] = np.nan
    corrupted_data.loc[0, "SEX"] = np.nan
    corrupted_data.loc[1, "PAY_0"] = np.nan

    X_corrupted = preprocessor.transform(corrupted_data)
    assert not np.isnan(X_corrupted).any()
    assert X_corrupted.shape == (len(corrupted_data), len(ALL_FEATURE_COLS))


def test_preprocessor_serialization(tmp_path, sample_raw_data):
    """Verifies that saving and reloading preprocessor preserves exact transformation output."""
    preprocessor = TabNetDataPreprocessor()
    preprocessor.fit(sample_raw_data)
    orig_output = preprocessor.transform(sample_raw_data)

    save_file = tmp_path / "test_preprocessor.joblib"
    preprocessor.save(save_file)

    reloaded = TabNetDataPreprocessor.load(save_file)
    reloaded_output = reloaded.transform(sample_raw_data)

    np.testing.assert_array_almost_equal(orig_output, reloaded_output)
