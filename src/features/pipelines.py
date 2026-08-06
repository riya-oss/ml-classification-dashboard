"""sklearn Pipeline + ColumnTransformer for the full preprocessing flow."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features.preprocess import (
    BINARY_COLS,
    NUMERIC_COLS,
    OHE_COLS,
    TARGET,
    prepare,
    save_processed,
)

# ── Column sets resolved at runtime ───────────────────────────────────────────
# Binary cols are already 0/1 after preprocess.encode_binary(),
# so they only need scaling alongside numerics.
_SCALE_COLS = NUMERIC_COLS + BINARY_COLS


def build_preprocessor(
    numeric_cols: list[str] | None = None,
    ohe_cols: list[str] | None = None,
) -> ColumnTransformer:
    """
    Return an unfitted ColumnTransformer.

    numeric_cols : columns to StandardScale (defaults to _SCALE_COLS)
    ohe_cols     : columns to OneHotEncode  (defaults to OHE_COLS)
    """
    numeric_cols = numeric_cols if numeric_cols is not None else _SCALE_COLS
    ohe_cols     = ohe_cols     if ohe_cols     is not None else OHE_COLS

    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_cols,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="if_binary",        # avoid redundant columns for 2-level features
                ),
                ohe_cols,
            ),
        ],
        remainder="drop",       # silently drop any columns not listed above
        verbose_feature_names_out=False,
    )


def build_pipeline(
    numeric_cols: list[str] | None = None,
    ohe_cols: list[str] | None = None,
) -> Pipeline:
    """Return an unfitted end-to-end preprocessing Pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_cols, ohe_cols)),
        ]
    )


def run_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple:
    """
    Full preprocessing run from raw DataFrame to train/test arrays.

    Returns
    -------
    X_train, X_test, y_train, y_test, pipeline, feature_names
        X_*         : np.ndarray  (scaled + encoded)
        y_*         : pd.Series   (0 / 1)
        pipeline    : fitted Pipeline (use for inference)
        feature_names: list[str]  (column names matching X_* columns)
    """
    # ── 1. Clean, engineer, encode binary, extract target ─────────────────────
    X, y = prepare(df)

    # Save the full cleaned+engineered frame (features + target) before splitting
    save_processed(pd.concat([X, y.rename(TARGET)], axis=1))

    # ── 2. Guard: keep only columns the pipeline knows about ──────────────────
    scale_cols = [c for c in _SCALE_COLS if c in X.columns]
    ohe_cols   = [c for c in OHE_COLS    if c in X.columns]

    # ── 3. Stratified train / test split (on raw, before fit) ─────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )

    # ── 4. Fit pipeline on train only, transform both splits ──────────────────
    pipeline = build_pipeline(scale_cols, ohe_cols)
    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc  = pipeline.transform(X_test)

    feature_names = pipeline.get_feature_names_out().tolist()

    print(f"Train : {X_train_proc.shape[0]:,} rows x {X_train_proc.shape[1]} features")
    print(f"Test  : {X_test_proc.shape[0]:,}  rows x {X_test_proc.shape[1]} features")
    print(f"Churn rate — train: {y_train.mean():.3f}  test: {y_test.mean():.3f}")

    return X_train_proc, X_test_proc, y_train, y_test, pipeline, feature_names


# ── SMOTE oversampling ────────────────────────────────────────────────────────

def apply_smote(
    X_train: np.ndarray,
    y_train: pd.Series | np.ndarray,
    sampling_strategy: float | str = "auto",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Oversample the minority class on the TRAINING set only using SMOTE.

    Must be called AFTER train/test split and AFTER preprocessing (needs numeric data).
    Never apply to the test set.
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    original_pos  = int((y_train == 1).sum())
    resampled_pos = int((y_res   == 1).sum())
    print(f"[SMOTE] Minority class: {original_pos:,} → {resampled_pos:,}  "
          f"| Total rows: {len(y_train):,} → {len(y_res):,}")
    return X_res, y_res


# ── CLI smoke-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "raw",
        "Telco_customer_churn.xlsx",
    )
    raw = pd.read_excel(DATA_PATH)
    X_tr, X_te, y_tr, y_te, pipe, names = run_pipeline(raw)
    print("\nFirst 5 feature names:", names[:5])
