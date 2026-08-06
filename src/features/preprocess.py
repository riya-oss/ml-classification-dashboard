"""Raw data cleaning and feature engineering — no sklearn, no side effects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Column taxonomy ────────────────────────────────────────────────────────────

TARGET = "Churn Label"

# Dropped before any ML step (identifiers / geo noise)
DROP_ALWAYS = ["CustomerID", "Count", "Lat Long", "Latitude", "Longitude",
               "Zip Code", "City", "State", "Country"]

# Dropped to prevent target leakage
DROP_LEAKAGE = ["Churn Value", "Churn Score", "CLTV", "Churn Reason", "Tenure Band"]

# Binary yes/no columns → mapped to 0/1 before pipeline
BINARY_COLS = [
    "Partner", "Dependents", "Phone Service", "Paperless Billing",
    "Senior Citizen",
]

# Categorical columns with >2 levels → OneHotEncoded
OHE_COLS = [
    "Gender",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Payment Method",
]

# Numeric columns → StandardScaled
NUMERIC_COLS = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    # engineered features added by engineer_features()
    "Services Count",
    "Charge Per Tenure",
    "Avg Monthly Spend",
]


# ── Step 1: Raw cleaning ───────────────────────────────────────────────────────

def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes, drop identifier/leakage columns, handle known dirty values."""
    df = df.copy()

    # Fix Total Charges stored as object (blanks = 0-tenure new customers)
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df["Total Charges"] = df["Total Charges"].fillna(0.0)

    # Standardise Senior Citizen to Yes/No if it arrives as 0/1
    if df["Senior Citizen"].dtype != object:
        df["Senior Citizen"] = df["Senior Citizen"].map({1: "Yes", 0: "No"})

    drop_cols = [c for c in DROP_ALWAYS + DROP_LEAKAGE if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df


# ── Step 2: Feature engineering ───────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that improve signal without leaking target info."""
    df = df.copy()

    # Count number of active add-on services per customer
    addon_cols = [
        "Online Security", "Online Backup", "Device Protection",
        "Tech Support", "Streaming TV", "Streaming Movies",
    ]
    df["Services Count"] = (
        df[addon_cols]
        .apply(lambda col: col.map({"Yes": 1}).fillna(0))
        .sum(axis=1)
        .astype(int)
    )

    # Average monthly spend proxy (handles tenure=0 safely)
    df["Charge Per Tenure"] = np.where(
        df["Tenure Months"] > 0,
        df["Total Charges"] / df["Tenure Months"],
        df["Monthly Charges"],
    )

    # Normalised monthly spend relative to service bundle size
    df["Avg Monthly Spend"] = np.where(
        df["Services Count"] > 0,
        df["Monthly Charges"] / df["Services Count"],
        df["Monthly Charges"],
    )

    return df


# ── Step 3: Binary encoding ────────────────────────────────────────────────────

def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Map Yes/No columns to 1/0 in-place before the sklearn pipeline."""
    df = df.copy()
    yes_no = {"Yes": 1, "No": 0}
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(yes_no).fillna(0).astype(int)
    return df


# ── Step 4: Target extraction ─────────────────────────────────────────────────

def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (features, binary target). Target: Yes=1, No=0."""
    y = (df[TARGET] == "Yes").astype(int).rename("churn")
    X = df.drop(columns=[TARGET])
    return X, y


# ── Convenience: full cleaning chain ─────────────────────────────────────────

ROOT           = Path(__file__).resolve().parents[2]
PROCESSED_DIR  = ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "telco_processed.csv"


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Run clean → engineer → encode_binary → split. Returns (X, y)."""
    df = clean_raw(df)
    df = engineer_features(df)
    df = encode_binary(df)
    return split_X_y(df)


def save_processed(df: pd.DataFrame) -> None:
    """Persist the cleaned+engineered DataFrame (with target) to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"[preprocess] Processed data saved -> {PROCESSED_PATH}  "
          f"({df.shape[0]:,} rows x {df.shape[1]} cols)")


def load_processed() -> pd.DataFrame:
    """Load the saved processed CSV; raises FileNotFoundError if not yet generated."""
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Processed file not found at {PROCESSED_PATH}. "
            "Run run_pipeline() once to generate it."
        )
    return pd.read_csv(PROCESSED_PATH)
