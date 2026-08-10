"""Reusable evaluation metrics and reporting utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# -- Paths ----------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "reports" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def _model_dir(model_name: str) -> Path:
    """Return (and create) the per-model metrics subdirectory."""
    d = METRICS_DIR / f"{model_name}_metrics"
    d.mkdir(parents=True, exist_ok=True)
    return d


# -- Core metric computation ----------------------------------------------------

def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series | None = None,
    prefix: str = "",
) -> dict[str, float]:
    """
    Compute all standard binary classification metrics.

    Parameters
    ----------
    y_true  : ground-truth labels (0 / 1)
    y_pred  : predicted binary labels (0 / 1)
    y_proba : predicted probability for class=1 (required for AUC)
    prefix  : prepended to each key (e.g. "train_" or "test_")

    Returns
    -------
    dict of metric_name -> float
    """
    metrics: dict[str, float] = {
        f"{prefix}accuracy":  round(accuracy_score(y_true, y_pred), 4),
        f"{prefix}precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        f"{prefix}recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        f"{prefix}f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        f"{prefix}mcc":       round(matthews_corrcoef(y_true, y_pred), 4),
    }

    if y_proba is not None:
        metrics[f"{prefix}auc"] = round(roc_auc_score(y_true, y_proba), 4)

    return metrics


def confusion_matrix_df(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Return confusion matrix as a labelled DataFrame."""
    labels = labels or ["Retained (0)", "Churned (1)"]
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm, index=[f"Actual: {l}" for l in labels],
                        columns=[f"Pred: {l}" for l in labels])


# -- Comparison across multiple runs -------------------------------------------

def load_all_metrics() -> pd.DataFrame:
    """
    Load every metrics JSON from all per-model subdirectories and return a comparison DataFrame.
    Useful for comparing runs across different models / hyperparameters.
    """
    records = []
    for path in sorted(METRICS_DIR.glob("**/*.json")):
        with open(path) as f:
            data = json.load(f)
        row = {"model_name": data.get("model_name"), "timestamp": data.get("timestamp")}
        row.update(data.get("metrics", {}))
        records.append(row)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Keep only the latest run per model when multiple runs exist
    if "timestamp" in df.columns:
        df = (df.sort_values("timestamp", ascending=False)
                .drop_duplicates(subset=["model_name"])
                .reset_index(drop=True))

    return df.sort_values("test_auc", ascending=False).reset_index(drop=True)


def save_comparison_csv() -> Path:
    """Write a single CSV comparing all saved metric runs."""
    df = load_all_metrics()
    out = METRICS_DIR / "all_runs_comparison.csv"
    df.to_csv(out, index=False)
    print(f"[evaluate] Comparison table saved -> {out}")
    return out


# -- Per-run metric save --------------------------------------------------------

def save_metrics(
    metrics: dict[str, float],
    model_name: str,
    timestamp: str,
    extra: dict | None = None,
) -> Path:
    """Persist a single run's metrics dict to the model's own subdirectory."""
    record = {
        "model_name": model_name,
        "timestamp":  timestamp,
        "metrics":    metrics,
        **(extra or {}),
    }
    out = _model_dir(model_name) / f"{model_name}_{timestamp}.json"
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    return out


# -- Pretty print --------------------------------------------------------------

def find_optimal_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    metric: str = "f1",
) -> tuple[float, dict[str, float]]:
    """
    Sweep probability thresholds and return the one that maximises `metric`.

    Parameters
    ----------
    y_true  : ground-truth binary labels
    y_proba : predicted probabilities for class=1
    metric  : one of 'f1' | 'precision' | 'recall'

    Returns
    -------
    (best_threshold, metrics_at_best_threshold)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    best_thresh, best_score = 0.5, 0.0
    for thresh, prec, rec in zip(thresholds, precisions[:-1], recalls[:-1]):
        if metric == "f1":
            score = 2 * prec * rec / (prec + rec + 1e-9)
        elif metric == "precision":
            score = prec
        else:
            score = rec

        if score > best_score:
            best_score, best_thresh = score, thresh

    y_pred_tuned = (y_proba >= best_thresh).astype(int)
    tuned_metrics = compute_metrics(y_true, y_pred_tuned, y_proba, prefix="tuned_")
    tuned_metrics["tuned_threshold"] = round(float(best_thresh), 4)

    print(f"[threshold] Optimal threshold ({metric}): {best_thresh:.4f}  "
          f"-> F1 {tuned_metrics['tuned_f1']:.4f}  "
          f"Precision {tuned_metrics['tuned_precision']:.4f}  "
          f"Recall {tuned_metrics['tuned_recall']:.4f}")
    return best_thresh, tuned_metrics


def print_metrics(metrics: dict[str, float], title: str = "") -> None:
    width = 45
    print(f"\n{'-' * width}")
    if title:
        print(f"  {title}")
        print(f"{'-' * width}")
    for k, v in metrics.items():
        print(f"  {k:<28}: {v:.4f}")
    print(f"{'-' * width}\n")
