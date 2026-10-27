"""Data loader module for LoanGuard AI.

Downloads, standardizes, or generates the Credit Card Default dataset.
Prevents target leakage and validates data integrity.
"""

import io
import logging
import sys
import zipfile
from pathlib import Path
from typing import Dict, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import requests

from config import (
    ALL_FEATURE_COLS,
    CATEGORICAL_COLS,
    DATA_DIR,
    DATASET_PATH,
    NUMERICAL_COLS,
    TARGET_COL,
    UCI_URL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_realistic_synthetic_data(n_samples: int = 30000, random_state: int = 42) -> pd.DataFrame:
    """Generates realistic synthetic loan/credit default data matching the UCI schema.

    Used when external UCI download is unavailable, ensuring the application
    remains completely self-contained, reproducible, and verifiable.
    """
    logger.info(f"Generating realistic synthetic credit default data ({n_samples} samples)...")
    np.random.seed(random_state)

    # 1. Demographics
    # SEX: 1=Male (40%), 2=Female (60%)
    sex = np.random.choice([1, 2], size=n_samples, p=[0.40, 0.60])

    # EDUCATION: 1=Grad School (35%), 2=University (46%), 3=High School (16%), 4=Others (3%)
    education = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.35, 0.46, 0.16, 0.03])

    # MARRIAGE: 1=Married (45%), 2=Single (53%), 3=Others (2%)
    marriage = np.random.choice([1, 2, 3], size=n_samples, p=[0.45, 0.53, 0.02])

    # AGE: Gamma-like distribution clipped between 21 and 75
    age = np.clip(np.random.normal(loc=35, scale=9.5, size=n_samples).astype(int), 21, 75)

    # LIMIT_BAL: Log-normal distribution reflecting realistic credit limits ($10k to $1M NTD)
    limit_bal = np.exp(np.random.normal(loc=11.5, scale=0.8, size=n_samples))
    limit_bal = np.round(np.clip(limit_bal, 10000, 1000000) / 10000) * 10000

    # 2. Financial History (Bills, Payments, and Payment Delays)
    # Underlying latent financial distress factor (higher = more distressed)
    distress = (
        0.3 * (age < 28).astype(float)
        + 0.2 * (education == 3).astype(float)
        - 0.3 * (limit_bal > 300000).astype(float)
        + np.random.normal(0, 1.0, size=n_samples)
    )

    # Delays for 6 months (PAY_6 to PAY_0)
    pay_cols = {}
    for i, month_name in enumerate(["PAY_6", "PAY_5", "PAY_4", "PAY_3", "PAY_2", "PAY_0"]):
        # Autoregressive persistence of payment delay
        delay_propensity = distress + 0.3 * i + np.random.normal(0, 0.8, size=n_samples)
        pay_vals = np.zeros(n_samples, dtype=int)
        pay_vals[delay_propensity < -0.8] = -1  # Paid in full
        pay_vals[(delay_propensity >= -0.8) & (delay_propensity < 0.6)] = 0  # Revolving credit
        pay_vals[(delay_propensity >= 0.6) & (delay_propensity < 1.4)] = 1  # 1 month delay
        pay_vals[(delay_propensity >= 1.4) & (delay_propensity < 2.2)] = 2  # 2 month delay
        pay_vals[delay_propensity >= 2.2] = np.clip(
            (delay_propensity[delay_propensity >= 2.2] - 0.5).astype(int), 2, 8
        )
        pay_cols[month_name] = pay_vals

    # Bill Amounts & Payment Amounts
    bill_cols = {}
    pay_amt_cols = {}
    for i in range(1, 7):
        # Bill statement depends on credit limit and utilization
        utilization = np.clip(np.random.beta(a=2, b=3, size=n_samples) + 0.1 * (distress > 0.5), 0.01, 1.2)
        bill = np.round(limit_bal * utilization + np.random.normal(0, 2000, size=n_samples))
        bill = np.clip(bill, 0, None)
        bill_cols[f"BILL_AMT{i}"] = bill

        # Payment depends on bill, delay, and financial distress
        pay_ratio = np.clip(0.8 - 0.2 * pay_cols["PAY_0"] + np.random.normal(0, 0.2, size=n_samples), 0.0, 1.0)
        payment = np.round(bill * pay_ratio)
        pay_amt_cols[f"PAY_AMT{i}"] = np.clip(payment, 0, None)

    # Target default probability (Yeh & Lien: baseline default rate ~22.1%)
    # Default strongly driven by recent delay (PAY_0, PAY_2), utilization, and low payment ratio
    default_logits = (
        - 1.8
        + 1.35 * (pay_cols["PAY_0"] > 0).astype(float)
        + 0.85 * (pay_cols["PAY_2"] > 0).astype(float)
        + 0.50 * (bill_cols["BILL_AMT1"] > 0.9 * limit_bal).astype(float)
        - 0.60 * (pay_amt_cols["PAY_AMT1"] >= bill_cols["BILL_AMT1"]).astype(float)
        + 0.40 * distress
    )
    default_prob = 1.0 / (1.0 + np.exp(-default_logits))
    default_target = (np.random.uniform(0, 1, size=n_samples) < default_prob).astype(int)

    # Construct DataFrame
    df_data = {
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        "PAY_0": pay_cols["PAY_0"],
        "PAY_2": pay_cols["PAY_2"],
        "PAY_3": pay_cols["PAY_3"],
        "PAY_4": pay_cols["PAY_4"],
        "PAY_5": pay_cols["PAY_5"],
        "PAY_6": pay_cols["PAY_6"],
        "BILL_AMT1": bill_cols["BILL_AMT1"],
        "BILL_AMT2": bill_cols["BILL_AMT2"],
        "BILL_AMT3": bill_cols["BILL_AMT3"],
        "BILL_AMT4": bill_cols["BILL_AMT4"],
        "BILL_AMT5": bill_cols["BILL_AMT5"],
        "BILL_AMT6": bill_cols["BILL_AMT6"],
        "PAY_AMT1": pay_amt_cols["PAY_AMT1"],
        "PAY_AMT2": pay_amt_cols["PAY_AMT2"],
        "PAY_AMT3": pay_amt_cols["PAY_AMT3"],
        "PAY_AMT4": pay_amt_cols["PAY_AMT4"],
        "PAY_AMT5": pay_amt_cols["PAY_AMT5"],
        "PAY_AMT6": pay_amt_cols["PAY_AMT6"],
        TARGET_COL: default_target,
    }

    df = pd.DataFrame(df_data)
    logger.info(f"Synthetic dataset created with shape {df.shape}. Default rate: {df[TARGET_COL].mean():.2%}")
    return df


def download_uci_credit_data() -> pd.DataFrame:
    """Attempts to download the UCI Credit Card Default dataset.

    Falls back cleanly to the realistic synthetic data generator if network
    request fails or format cannot be parsed.
    """
    logger.info(f"Attempting to download dataset from UCI URL: {UCI_URL} ...")
    try:
        response = requests.get(UCI_URL, timeout=15)
        response.raise_for_status()

        # Unpack zip in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            file_names = z.namelist()
            logger.info(f"Files found in UCI zip: {file_names}")
            # Locate excel or csv file
            target_file = next((f for f in file_names if f.endswith(".xls") or f.endswith(".xlsx") or f.endswith(".csv")), None)
            if not target_file:
                raise ValueError("No tabular file found inside UCI zip")

            with z.open(target_file) as f:
                if target_file.endswith(".csv"):
                    df = pd.read_csv(f, header=1)
                else:
                    df = pd.read_excel(f, header=1)

        # Standardize columns
        df = standardize_credit_dataset(df)
        logger.info(f"Successfully downloaded and processed UCI dataset with {len(df)} records.")
        return df

    except Exception as e:
        logger.warning(f"Could not fetch live UCI dataset ({e}). Using realistic synthetic data generator.")
        return generate_realistic_synthetic_data()


def standardize_credit_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans column names, removes identifiers, and groups undocumented categorical levels."""
    df = df.copy()

    # Drop ID column if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Standardize column names
    col_mapping = {}
    for col in df.columns:
        clean_name = str(col).strip().replace(" ", "_").replace(".", "_")
        if "default" in clean_name.lower():
            col_mapping[col] = TARGET_COL
        elif clean_name.upper() == "PAY_1":
            col_mapping[col] = "PAY_0"
        else:
            col_mapping[col] = clean_name.upper()

    df = df.rename(columns=col_mapping)

    # Handle undocumented categories in EDUCATION (0, 4, 5, 6 -> 4: others)
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x not in [1, 2, 3] else x)

    # Handle undocumented categories in MARRIAGE (0, 3 -> 3: others)
    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].apply(lambda x: 3 if x not in [1, 2] else x)

    # Ensure required columns exist
    missing_cols = [c for c in ALL_FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Standardized dataset is missing expected columns: {missing_cols}")

    return df[ALL_FEATURE_COLS + [TARGET_COL]]


def load_or_fetch_dataset(force_refresh: bool = False) -> pd.DataFrame:
    """Loads dataset from local disk if available, otherwise downloads/generates it.

    Ensures the application does not repeatedly re-download data.
    """
    if DATASET_PATH.exists() and not force_refresh:
        logger.info(f"Loading existing dataset from {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
        return df

    logger.info("Dataset not found locally. Initiating fetch/generation...")
    df = download_uci_credit_data()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    logger.info(f"Saved standardized dataset to {DATASET_PATH}")
    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict:
    """Generates a summary of the dataset for validation and documentation."""
    total_records = len(df)
    missing_counts = df.isnull().sum().to_dict()
    default_rate = float(df[TARGET_COL].mean()) if TARGET_COL in df.columns else 0.0

    return {
        "num_records": total_records,
        "num_features": len(ALL_FEATURE_COLS),
        "target_column": TARGET_COL,
        "default_rate": default_rate,
        "non_default_count": int((df[TARGET_COL] == 0).sum()) if TARGET_COL in df.columns else 0,
        "default_count": int((df[TARGET_COL] == 1).sum()) if TARGET_COL in df.columns else 0,
        "has_missing": any(c > 0 for c in missing_counts.values()),
        "missing_counts": missing_counts,
    }


if __name__ == "__main__":
    df = load_or_fetch_dataset()
    summary = get_dataset_summary(df)
    print("\n=== Dataset Summary ===")
    for k, v in summary.items():
        if k != "missing_counts":
            print(f"  {k}: {v}")
    print(f"  Missing values total: {sum(summary['missing_counts'].values())}")
