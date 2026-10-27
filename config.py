"""LoanGuard AI - Configuration and Hyperparameters.

Defines directory paths, feature schemas, model hyperparameters,
risk thresholds, and dataset metadata.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset details
DATASET_FILENAME = "default_of_credit_card_clients.csv"
DATASET_PATH = DATA_DIR / DATASET_FILENAME
UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"

TARGET_COL = "default_payment_next_month"

# Categorical and numerical feature definitions
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]

NUMERICAL_COLS = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

# Strict ordering of features for train and inference
ALL_FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS

# Feature labels and readable descriptions for UI and Explainability
FEATURE_DESCRIPTIONS = {
    "LIMIT_BAL": "Amount of given credit / Credit Limit (NT$)",
    "SEX": "Gender (1 = Male, 2 = Female)",
    "EDUCATION": "Education Level (1 = Grad School, 2 = University, 3 = High School, 4 = Others)",
    "MARRIAGE": "Marital Status (1 = Married, 2 = Single, 3 = Others)",
    "AGE": "Age of Applicant (Years)",
    "PAY_0": "Repayment Status in Sept (-2=No consumption, -1=Paid in full, 0=Revolving, 1-8=Months delay)",
    "PAY_2": "Repayment Status in Aug (-2=No consumption, -1=Paid in full, 0=Revolving, 1-8=Months delay)",
    "PAY_3": "Repayment Status in July (-2=No consumption, -1=Paid in full, 0=Revolving, 1-8=Months delay)",
    "PAY_4": "Repayment Status in June (-2=No consumption, -1=Paid in full, 0=Revolving, 1-8=Months delay)",
    "PAY_5": "Repayment Status in May (-2=No consumption, -1=Paid in full, 0=Revolving, 1-8=Months delay)",
    "PAY_6": "Repayment Status in April (-2=No consumption, -1=Paid in full, 0=Revolving, 1-8=Months delay)",
    "BILL_AMT1": "Bill Statement Amount in Sept (NT$)",
    "BILL_AMT2": "Bill Statement Amount in Aug (NT$)",
    "BILL_AMT3": "Bill Statement Amount in July (NT$)",
    "BILL_AMT4": "Bill Statement Amount in June (NT$)",
    "BILL_AMT5": "Bill Statement Amount in May (NT$)",
    "BILL_AMT6": "Bill Statement Amount in April (NT$)",
    "PAY_AMT1": "Amount Paid in Sept (NT$)",
    "PAY_AMT2": "Amount Paid in Aug (NT$)",
    "PAY_AMT3": "Amount Paid in July (NT$)",
    "PAY_AMT4": "Amount Paid in June (NT$)",
    "PAY_AMT5": "Amount Paid in May (NT$)",
    "PAY_AMT6": "Amount Paid in April (NT$)",
}

# Categorical value mappings for user interface and display
CATEGORICAL_MAPPINGS = {
    "SEX": {1: "Male", 2: "Female"},
    "EDUCATION": {
        1: "Graduate School",
        2: "University",
        3: "High School",
        4: "Others / Unknown",
    },
    "MARRIAGE": {
        1: "Married",
        2: "Single",
        3: "Others / Divorced",
    },
    "PAY_STATUS": {
        -2: "No consumption / Inactive",
        -1: "Paid in full",
        0: "Revolving credit (on-time minimum paid)",
        1: "Payment delay for 1 month",
        2: "Payment delay for 2 months",
        3: "Payment delay for 3 months",
        4: "Payment delay for 4 months",
        5: "Payment delay for 5 months",
        6: "Payment delay for 6 months",
        7: "Payment delay for 7 months",
        8: "Payment delay for 8+ months",
    },
}

# TabNet Hyperparameters
TABNET_PARAMS = {
    "n_d": 16,
    "n_a": 16,
    "n_steps": 4,
    "gamma": 1.3,
    "lambda_sparse": 1e-3,
    "cat_emb_dim": [1, 2, 2],
    "mask_type": "entmax",
    "seed": 42,
    "verbose": 1,
}

TRAINING_PARAMS = {
    "max_epochs": 40,
    "patience": 10,
    "batch_size": 1024,
    "virtual_batch_size": 128,
    "learning_rate": 0.02,
    "weight_decay": 1e-5,
    "num_workers": 0,
    "drop_last": False,
}

# Risk Rating Bands
DEFAULT_RISK_THRESHOLDS = {
    "low": 0.20,
    "medium": 0.50,
}

# Model artifacts file names
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"
TABNET_MODEL_FILE = MODELS_DIR / "tabnet_model.zip"
BASELINE_LR_FILE = MODELS_DIR / "baseline_lr.joblib"
METRICS_REPORT_FILE = REPORTS_DIR / "evaluation_metrics.json"
FEATURE_IMPORTANCE_FILE = REPORTS_DIR / "global_feature_importance.json"
TEST_DATA_CACHE = DATA_DIR / "test_set_cache.parquet"
