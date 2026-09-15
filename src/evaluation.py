"""
Evaluation metrics: the technical ones (MAE/RMSE/R2 + bootstrap
confidence intervals) and the business metric (WMAE, weighted by
consumption concentration).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def weighted_mae(y_true, y_pred, weights):
    """
    WMAE: gives more weight to the error in the countries that account
    for 80% of world consumption (weights=2) versus the rest
    (weights=1). Same pattern as an ABC/Pareto-style business metric:
    getting the big markets right matters more than the small ones.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, ci=0.95, random_state=42):
    """Bootstrap (resampling with replacement) confidence interval for a metric."""
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        scores[i] = metric_fn(y_true[idx], y_pred[idx])
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return float(np.mean(scores)), float(lower), float(upper)


def evaluate(y_true, y_pred, weights=None, n_boot=1000):
    """
    Returns a dict with MAE, RMSE, R2 (point estimate and 95%
    bootstrap interval) and, if weights are passed, the WMAE too.
    """
    mae, mae_lo, mae_hi = bootstrap_ci(y_true, y_pred, mean_absolute_error, n_boot)
    rmse, rmse_lo, rmse_hi = bootstrap_ci(
        y_true, y_pred, lambda a, b: np.sqrt(mean_squared_error(a, b)), n_boot
    )
    r2, r2_lo, r2_hi = bootstrap_ci(y_true, y_pred, r2_score, n_boot)

    result = {
        "MAE": mae, "MAE_ci_low": mae_lo, "MAE_ci_high": mae_hi,
        "RMSE": rmse, "RMSE_ci_low": rmse_lo, "RMSE_ci_high": rmse_hi,
        "R2": r2, "R2_ci_low": r2_lo, "R2_ci_high": r2_hi,
    }
    if weights is not None:
        result["WMAE"] = weighted_mae(y_true, y_pred, weights)
    return result


def results_table(results_dict):
    """results_dict: {model_name: evaluate(...) output} -> tidy summary DataFrame, one row per model."""
    rows = []
    for model_name, res in results_dict.items():
        rows.append({
            "model": model_name,
            "MAE": round(res["MAE"], 2),
            "MAE_CI95": f"[{res['MAE_ci_low']:.2f}, {res['MAE_ci_high']:.2f}]",
            "RMSE": round(res["RMSE"], 2),
            "R2": round(res["R2"], 3),
            "R2_CI95": f"[{res['R2_ci_low']:.3f}, {res['R2_ci_high']:.3f}]",
            "WMAE": round(res["WMAE"], 2) if "WMAE" in res else None,
        })
    return pd.DataFrame(rows)
