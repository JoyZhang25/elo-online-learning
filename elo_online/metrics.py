"""Probability-forecast diagnostics for sequential match predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def per_game_log_loss(outcome: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """Binary cross-entropy, allowing a draw to be encoded as outcome 0.5."""

    y = np.asarray(outcome, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0 - 1e-12)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def calibration_table(
    outcome: np.ndarray,
    probability: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return equal-width reliability bins with empty bins omitted."""

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    frame = pd.DataFrame(
        {
            "outcome": np.asarray(outcome, dtype=float),
            "probability": np.asarray(probability, dtype=float),
        }
    )
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    frame["bin"] = pd.cut(
        frame["probability"], edges, include_lowest=True, labels=False
    )
    grouped = (
        frame.dropna(subset=["bin"])
        .groupby("bin", observed=True)
        .agg(
            n=("outcome", "size"),
            mean_prediction=("probability", "mean"),
            empirical_score=("outcome", "mean"),
        )
        .reset_index()
    )
    grouped["bin"] = grouped["bin"].astype(int)
    return grouped


def forecast_metrics(outcome: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Compute proper scores and an interpretable calibration error."""

    y = np.asarray(outcome, dtype=float)
    p = np.asarray(probability, dtype=float)
    if y.size == 0 or y.shape != p.shape:
        raise ValueError("outcome and probability must be nonempty with equal shape")
    calibration = calibration_table(y, p)
    calibration_error = np.average(
        np.abs(calibration["empirical_score"] - calibration["mean_prediction"]),
        weights=calibration["n"],
    )
    return {
        "n_games": int(y.size),
        "log_loss": float(np.mean(per_game_log_loss(y, p))),
        "brier_score": float(np.mean((y - p) ** 2)),
        "calibration_error": float(calibration_error),
        "mean_prediction": float(np.mean(p)),
        "mean_outcome": float(np.mean(y)),
    }
