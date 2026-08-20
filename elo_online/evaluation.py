"""Leakage-free walk-forward Elo evaluation on chronological match data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import forecast_metrics, per_game_log_loss
from .model import EloModel, expected_score


@dataclass
class RealDataResult:
    """All auditable outputs from model selection and final evaluation."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    candidate_metrics: pd.DataFrame
    selected_k: float
    selected_white_advantage: float
    validation_start: int
    test_start: int


def walk_forward_predictions(
    games: pd.DataFrame,
    *,
    k_factor: float,
    white_advantage: float = 0.0,
) -> pd.DataFrame:
    """Predict each game before using its outcome to update the model."""

    required = {
        "game_id",
        "created_at",
        "white_id",
        "black_id",
        "white_platform_rating",
        "black_platform_rating",
        "white_score",
    }
    missing = required.difference(games.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    ordered = games.sort_values(["created_at", "game_id"], kind="stable").reset_index(
        drop=True
    )
    model = EloModel(k_factor=k_factor, advantage_a=white_advantage)
    rows: list[dict] = []
    for event_index, row in ordered.iterrows():
        white = str(row["white_id"])
        black = str(row["black_id"])
        prediction = model.predict(
            white,
            black,
            seed_a=float(row["white_platform_rating"]),
            seed_b=float(row["black_platform_rating"]),
        )
        rating_white_before = model.rating(white)
        rating_black_before = model.rating(black)
        rows.append(
            {
                "event_index": int(event_index),
                "game_id": row["game_id"],
                "created_at": row["created_at"],
                "white_id": white,
                "black_id": black,
                "white_score": float(row["white_score"]),
                "elo_probability": prediction,
                "platform_probability": float(
                    expected_score(
                        float(row["white_platform_rating"]),
                        float(row["black_platform_rating"]),
                        advantage_a=white_advantage,
                    )
                ),
                "white_rating_before": rating_white_before,
                "black_rating_before": rating_black_before,
                "white_platform_rating": float(row["white_platform_rating"]),
                "black_platform_rating": float(row["black_platform_rating"]),
            }
        )
        model.update(
            white,
            black,
            float(row["white_score"]),
            prediction_a=prediction,
        )
    return pd.DataFrame(rows)


def _metric_row(
    name: str,
    outcome: np.ndarray,
    probability: np.ndarray,
    *,
    split: str,
) -> dict[str, float | str]:
    return {"model": name, "split": split, **forecast_metrics(outcome, probability)}


def evaluate_real_data(
    games: pd.DataFrame,
    *,
    k_grid: tuple[float, ...] = (
        0.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
        96.0,
        128.0,
    ),
    white_advantage_grid: tuple[float, ...] = (0.0, 25.0, 50.0, 75.0),
    validation_start_fraction: float = 0.10,
    test_start_fraction: float = 0.60,
) -> RealDataResult:
    """Select hyperparameters on an early block and report a later test block.

    Every candidate runs forward from the first game.  The first 10% is a state
    warm-up, the next 50% selects ``K`` and white advantage, and the final 40%
    is held out for the reported comparison.
    """

    n_games = len(games)
    validation_start = int(np.floor(validation_start_fraction * n_games))
    test_start = int(np.floor(test_start_fraction * n_games))
    if not 0 < validation_start < test_start < n_games:
        raise ValueError("split fractions produce an invalid chronological split")

    candidate_rows: list[dict] = []
    candidate_predictions: dict[tuple[float, float], pd.DataFrame] = {}
    for advantage in white_advantage_grid:
        for k_factor in k_grid:
            predictions = walk_forward_predictions(
                games, k_factor=k_factor, white_advantage=advantage
            )
            candidate_predictions[(k_factor, advantage)] = predictions
            validation = predictions.iloc[validation_start:test_start]
            scores = forecast_metrics(
                validation["white_score"].to_numpy(),
                validation["elo_probability"].to_numpy(),
            )
            candidate_rows.append(
                {
                    "k_factor": k_factor,
                    "white_advantage": advantage,
                    "split": "validation",
                    **scores,
                }
            )
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["log_loss", "brier_score", "k_factor", "white_advantage"], kind="stable"
    )
    best = candidates.iloc[0]
    selected_k = float(best["k_factor"])
    selected_advantage = float(best["white_advantage"])
    predictions = candidate_predictions[(selected_k, selected_advantage)].copy()

    frozen = walk_forward_predictions(
        games, k_factor=0.0, white_advantage=selected_advantage
    )
    predictions["frozen_probability"] = frozen["elo_probability"]
    predictions["naive_probability"] = 0.5
    split = np.full(n_games, "warmup", dtype=object)
    split[validation_start:test_start] = "validation"
    split[test_start:] = "test"
    predictions["split"] = split

    test = predictions.iloc[test_start:].copy()
    y_test = test["white_score"].to_numpy()
    metric_rows = [
        _metric_row(
            "Constant-step Elo",
            y_test,
            test["elo_probability"].to_numpy(),
            split="test",
        ),
        _metric_row(
            "Frozen initial ratings",
            y_test,
            test["frozen_probability"].to_numpy(),
            split="test",
        ),
        _metric_row(
            "Lichess pre-game ratings",
            y_test,
            test["platform_probability"].to_numpy(),
            split="test",
        ),
        _metric_row(
            "No-skill 50%",
            y_test,
            test["naive_probability"].to_numpy(),
            split="test",
        ),
    ]
    metrics = pd.DataFrame(metric_rows).sort_values("log_loss", kind="stable")

    selected_loss = per_game_log_loss(y_test, test["elo_probability"].to_numpy())
    frozen_loss = per_game_log_loss(y_test, test["frozen_probability"].to_numpy())
    predictions.loc[
        predictions.index[test_start:], "cumulative_log_loss_vs_frozen"
    ] = np.cumsum(selected_loss - frozen_loss)
    return RealDataResult(
        predictions=predictions,
        metrics=metrics,
        candidate_metrics=candidates.reset_index(drop=True),
        selected_k=selected_k,
        selected_white_advantage=selected_advantage,
        validation_start=validation_start,
        test_start=test_start,
    )
