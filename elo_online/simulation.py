"""Controlled experiments for stability, drift tracking, and change points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import expected_score


@dataclass
class ChangePointResult:
    trajectories: pd.DataFrame
    summary: pd.DataFrame


@dataclass
class InvariantFluctuationResult:
    """Stationary samples and diagnostics for the small-step experiment."""

    samples: pd.DataFrame
    summary: pd.DataFrame


def _center(values: np.ndarray, target_mean: float = 1500.0) -> np.ndarray:
    return values - np.mean(values) + target_mean


def _game_stream(
    *,
    n_players: int,
    n_games: int,
    seed: int,
    drift_sd: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    true_ratings = _center(rng.normal(1500.0, 180.0, size=n_players))
    player_a = rng.integers(0, n_players, size=n_games)
    offsets = rng.integers(1, n_players, size=n_games)
    player_b = (player_a + offsets) % n_players
    outcomes = np.empty(n_games, dtype=float)
    truth = np.empty((n_games, n_players), dtype=np.float32)
    for t in range(n_games):
        if drift_sd > 0:
            true_ratings = _center(
                true_ratings + rng.normal(0.0, drift_sd, size=n_players)
            )
        probability = expected_score(
            true_ratings[player_a[t]], true_ratings[player_b[t]]
        )
        outcomes[t] = float(rng.random() < probability)
        truth[t] = true_ratings
    return player_a, player_b, outcomes, truth


def invariant_fluctuation_experiment(
    *,
    k_grid: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0),
    n_games: int = 200_000,
    burn_in: int = 40_000,
    sample_every: int = 20,
    true_rating: float = 1500.0,
    opponent_sd: float = 180.0,
    seed: int = 20260819,
) -> InvariantFluctuationResult:
    """Sample stationary Elo errors against a stream of known opponents.

    One focal player has a fixed latent rating and repeatedly faces opponents
    whose ratings are observed.  Each candidate ``K`` sees the same opponents
    and outcomes.  After burn-in, the function records the focal rating error
    so that the ``sqrt(K)`` scale and approximate Gaussian shape can be checked
    directly.
    """

    if not k_grid or any(k <= 0 for k in k_grid):
        raise ValueError("k_grid must contain positive values")
    if not 0 <= burn_in < n_games:
        raise ValueError("burn_in must lie in [0, n_games)")
    if sample_every <= 0:
        raise ValueError("sample_every must be positive")

    rng = np.random.default_rng(seed)
    opponent_ratings = rng.normal(true_rating, opponent_sd, size=n_games)
    true_probabilities = expected_score(true_rating, opponent_ratings)
    outcomes = (rng.random(n_games) < true_probabilities).astype(float)

    sample_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int]] = []
    for k_factor in k_grid:
        estimate = float(true_rating)
        errors: list[float] = []
        for t in range(n_games):
            prediction = expected_score(estimate, opponent_ratings[t])
            estimate += k_factor * (outcomes[t] - prediction)
            if t >= burn_in and (t - burn_in) % sample_every == 0:
                errors.append(estimate - true_rating)

        error = np.asarray(errors, dtype=float)
        if error.size < 2:
            raise ValueError("experiment must retain at least two samples per K")
        mean_error = float(np.mean(error))
        variance = float(np.var(error, ddof=1))
        standard_deviation = float(np.sqrt(variance))
        standardized = (error - mean_error) / standard_deviation
        sample_frames.append(
            pd.DataFrame(
                {
                    "k_factor": k_factor,
                    "sample_index": np.arange(error.size),
                    "rating_error": error,
                    "centered_error_over_sqrt_k": (error - mean_error)
                    / np.sqrt(k_factor),
                    "standardized_error": standardized,
                }
            )
        )
        summary_rows.append(
            {
                "k_factor": k_factor,
                "n_samples": int(error.size),
                "mean_error": mean_error,
                "variance": variance,
                "variance_over_k": variance / k_factor,
                "skewness": float(np.mean(standardized**3)),
                "excess_kurtosis": float(np.mean(standardized**4) - 3.0),
            }
        )

    return InvariantFluctuationResult(
        samples=pd.concat(sample_frames, ignore_index=True),
        summary=pd.DataFrame(summary_rows),
    )


def stability_adaptation_experiment(
    *,
    k_grid: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
    drift_grid: tuple[float, ...] = (0.0, 0.6, 1.0),
    n_players: int = 24,
    n_games: int = 24_000,
    burn_fraction: float = 0.5,
    seed: int = 20260818,
) -> pd.DataFrame:
    """Compare steady-state estimation error across stepsizes and drift rates."""

    rows: list[dict] = []
    burn = int(burn_fraction * n_games)
    for drift_index, drift_sd in enumerate(drift_grid):
        player_a, player_b, outcomes, truth = _game_stream(
            n_players=n_players,
            n_games=n_games,
            seed=seed + 1000 * drift_index,
            drift_sd=drift_sd,
        )
        for k_factor in k_grid:
            # Start at the initial latent state so this experiment isolates
            # stationary/tracking error rather than a one-off cold-start bias.
            estimates = truth[0].astype(float).copy()
            squared_errors: list[float] = []
            for t in range(n_games):
                i = player_a[t]
                j = player_b[t]
                probability = expected_score(estimates[i], estimates[j])
                change = k_factor * (outcomes[t] - probability)
                estimates[i] += change
                estimates[j] -= change
                if t >= burn and t % 20 == 0:
                    aligned = _center(estimates, float(np.mean(truth[t])))
                    squared_errors.append(float(np.mean((aligned - truth[t]) ** 2)))
            rows.append(
                {
                    "drift_sd_per_event": drift_sd,
                    "k_factor": k_factor,
                    "tracking_rmse": float(np.sqrt(np.mean(squared_errors))),
                    "n_players": n_players,
                    "n_games": n_games,
                }
            )
    return pd.DataFrame(rows)


def change_point_experiment(
    *,
    k_values: tuple[float, ...] = (8.0, 24.0, 64.0),
    n_games: int = 4_000,
    change_at: int = 2_000,
    jump: float = 240.0,
    seed: int = 731,
) -> ChangePointResult:
    """Track one player's sudden skill change against known-rating opponents."""

    rng = np.random.default_rng(seed)
    opponent_ratings = rng.normal(1500.0, 180.0, size=n_games)
    truth = np.full(n_games, 1500.0)
    truth[change_at:] += jump
    probabilities = expected_score(truth, opponent_ratings)
    outcomes = (rng.random(n_games) < probabilities).astype(float)
    trajectories = pd.DataFrame(
        {
            "event_index": np.arange(n_games),
            "true_rating": truth,
            "opponent_rating": opponent_ratings,
            "outcome": outcomes,
        }
    )
    summary_rows: list[dict] = []
    for k_factor in k_values:
        estimate = 1500.0
        path = np.empty(n_games)
        for t in range(n_games):
            prediction = expected_score(estimate, opponent_ratings[t])
            estimate += k_factor * (outcomes[t] - prediction)
            path[t] = estimate
        label = f"constant K={k_factor:g}"
        trajectories[label] = path
        post_rmse = np.sqrt(np.mean((path[change_at:] - truth[change_at:]) ** 2))
        target = truth[change_at] - 0.1 * jump
        hits = np.flatnonzero(path[change_at:] >= target)
        delay = float(hits[0]) if hits.size else np.nan
        summary_rows.append(
            {
                "method": label,
                "post_change_rmse": float(post_rmse),
                "events_to_90pct_of_jump": delay,
            }
        )

    estimate = 1500.0
    decay_path = np.empty(n_games)
    for t in range(n_games):
        prediction = expected_score(estimate, opponent_ratings[t])
        effective_k = 64.0 / np.sqrt(1.0 + t / 100.0)
        estimate += effective_k * (outcomes[t] - prediction)
        decay_path[t] = estimate
    label = "decaying K"
    trajectories[label] = decay_path
    post_rmse = np.sqrt(
        np.mean((decay_path[change_at:] - truth[change_at:]) ** 2)
    )
    target = truth[change_at] - 0.1 * jump
    hits = np.flatnonzero(decay_path[change_at:] >= target)
    summary_rows.append(
        {
            "method": label,
            "post_change_rmse": float(post_rmse),
            "events_to_90pct_of_jump": float(hits[0]) if hits.size else np.nan,
        }
    )
    return ChangePointResult(
        trajectories=trajectories,
        summary=pd.DataFrame(summary_rows),
    )
