"""Controlled constant-step experiments in Bradley-Terry coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import alpha_to_k, expected_score, k_to_alpha, rating_to_skill


@dataclass
class InvariantFluctuationResult:
    samples: pd.DataFrame
    summary: pd.DataFrame


@dataclass
class DynamicTrackingResult:
    grid: pd.DataFrame
    optima: pd.DataFrame
    scaling: pd.DataFrame


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    value = np.asarray(value, dtype=float)
    probability = np.exp(-np.logaddexp(0.0, -value))
    if probability.ndim == 0:
        return float(probability)
    return probability


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
    """Sample stationary Elo errors against a stream of known opponents."""

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

    sample_frames = []
    summary_rows = []
    for k_factor in k_grid:
        estimate = float(true_rating)
        errors = []
        for t in range(n_games):
            prediction = expected_score(estimate, opponent_ratings[t])
            estimate += k_factor * (outcomes[t] - prediction)
            if t >= burn_in and (t - burn_in) % sample_every == 0:
                errors.append(estimate - true_rating)

        rating_error = np.asarray(errors, dtype=float)
        skill_error = np.asarray(rating_to_skill(rating_error), dtype=float)
        mean_error = float(np.mean(skill_error))
        variance = float(np.var(skill_error, ddof=1))
        standard_deviation = float(np.sqrt(variance))
        standardized = (skill_error - mean_error) / standard_deviation
        alpha = k_to_alpha(k_factor)
        sample_frames.append(
            pd.DataFrame(
                {
                    "k_factor": k_factor,
                    "alpha": alpha,
                    "sample_index": np.arange(skill_error.size),
                    "skill_error": skill_error,
                    "centered_error_over_sqrt_alpha": (skill_error - mean_error)
                    / np.sqrt(alpha),
                    "standardized_error": standardized,
                }
            )
        )
        summary_rows.append(
            {
                "k_factor": k_factor,
                "alpha": alpha,
                "n_samples": int(skill_error.size),
                "mean_skill_error": mean_error,
                "skill_error_variance": variance,
                "variance_over_alpha": variance / alpha,
                "skewness": float(np.mean(standardized**3)),
                "excess_kurtosis": float(np.mean(standardized**4) - 3.0),
            }
        )
    return InvariantFluctuationResult(
        samples=pd.concat(sample_frames, ignore_index=True),
        summary=pd.DataFrame(summary_rows),
    )


def _latent_path(
    dynamics: str,
    tau: int,
    n_games: int,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if dynamics == "smooth":
        phase = rng.uniform(0.0, 2.0 * np.pi)
        return amplitude * np.sin(phase + np.pi * np.arange(n_games) / (2.0 * tau))

    path = np.empty(n_games, dtype=float)
    path[0] = rng.normal(0.0, amplitude)
    if dynamics == "ou":
        persistence = np.exp(-1.0 / tau)
        innovation_sd = amplitude * np.sqrt(1.0 - persistence**2)
        for t in range(1, n_games):
            path[t] = persistence * path[t - 1] + rng.normal(0.0, innovation_sd)
        return path
    if dynamics == "jump":
        for t in range(1, n_games):
            path[t] = (
                rng.normal(0.0, amplitude)
                if rng.random() < 1.0 / tau
                else path[t - 1]
            )
        return path
    raise ValueError(f"unknown dynamics: {dynamics}")


def _refined_optimal_alpha(alphas: np.ndarray, mse: np.ndarray) -> float:
    """Interpolate the local MSE minimum on the log-step scale."""

    optimum = int(np.argmin(mse))
    if optimum == 0 or optimum == len(alphas) - 1:
        return float(alphas[optimum])
    local_log_alpha = np.log(alphas[optimum - 1 : optimum + 2])
    quadratic, linear, _ = np.polyfit(
        local_log_alpha, mse[optimum - 1 : optimum + 2], 2
    )
    if quadratic <= 0:
        return float(alphas[optimum])
    vertex = np.clip(
        -linear / (2.0 * quadratic),
        local_log_alpha[0],
        local_log_alpha[-1],
    )
    return float(np.exp(vertex))


def dynamic_tracking_experiment(
    *,
    dynamics_grid: tuple[str, ...] = ("smooth", "ou", "jump"),
    tau_grid: tuple[int, ...] = (50, 100, 200, 400, 800),
    alpha_grid: tuple[float, ...] | None = None,
    amplitude: float = 1.25,
    minimum_games: int = 40_000,
    games_per_tau: int = 80,
    burn_fraction: float = 0.20,
    n_replications: int = 10,
    n_mc_resamples: int = 2_000,
    seed: int = 20260820,
) -> DynamicTrackingResult:
    """Estimate the step size that best tracks several latent-skill dynamics.

    The smooth and Ornstein-Uhlenbeck cases reproduce the asymptotic tracking
    questions studied for Elo: optimal ``alpha`` should scale respectively as
    ``tau^(-2/3)`` and ``tau^(-1/2)``. No reference exponent is assigned to the
    jump model. Independent replications
    quantify simulation uncertainty; within each replication, every candidate
    step sees the same latent path and outcomes.
    """

    if alpha_grid is None:
        alpha_grid = tuple(np.geomspace(0.0025, 0.50, 29))
    alphas = np.sort(np.asarray(alpha_grid, dtype=float))
    if np.any(alphas <= 0) or np.any(alphas >= 1):
        raise ValueError("alpha_grid must lie in (0, 1)")
    if len(np.unique(alphas)) != len(alphas):
        raise ValueError("alpha_grid values must be unique")
    if not dynamics_grid:
        raise ValueError("dynamics_grid must be nonempty")
    if not tau_grid or any(tau <= 0 for tau in tau_grid):
        raise ValueError("tau_grid must contain positive integers")
    if n_replications < 2:
        raise ValueError("n_replications must be at least two")
    if n_mc_resamples <= 0:
        raise ValueError("n_mc_resamples must be positive")

    grid_rows = []
    replication_mse: dict[tuple[str, int], np.ndarray] = {}
    seed_sequences = np.random.SeedSequence(seed).spawn(
        len(dynamics_grid) * len(tau_grid) * n_replications
    )
    seed_index = 0
    for dynamics in dynamics_grid:
        for tau in tau_grid:
            n_games = max(minimum_games, games_per_tau * tau)
            burn_in = int(burn_fraction * n_games)
            mse_by_replication = np.empty(
                (n_replications, len(alphas)), dtype=float
            )
            for replication in range(n_replications):
                rng = np.random.default_rng(seed_sequences[seed_index])
                seed_index += 1
                truth = _latent_path(dynamics, tau, n_games, amplitude, rng)
                outcomes = (rng.random(n_games) < _sigmoid(truth)).astype(float)
                estimates = np.zeros(len(alphas), dtype=float)
                squared_error = np.zeros(len(alphas), dtype=float)
                for t, outcome in enumerate(outcomes):
                    estimates += alphas * (outcome - _sigmoid(estimates))
                    if t >= burn_in:
                        squared_error += (estimates - truth[t]) ** 2
                mse_by_replication[replication] = squared_error / (
                    n_games - burn_in
                )

            replication_mse[(dynamics, tau)] = mse_by_replication
            mean_mse = mse_by_replication.mean(axis=0)
            mse_sd = mse_by_replication.std(axis=0, ddof=1)
            mse_se = mse_sd / np.sqrt(n_replications)
            for alpha, value, standard_deviation, standard_error in zip(
                alphas, mean_mse, mse_sd, mse_se
            ):
                grid_rows.append(
                    {
                        "dynamics": dynamics,
                        "tau": tau,
                        "alpha": alpha,
                        "k_equivalent": alpha_to_k(alpha),
                        "tracking_mse": value,
                        "tracking_rmse": np.sqrt(value),
                        "tracking_mse_sd": standard_deviation,
                        "tracking_mse_se": standard_error,
                        "n_games": n_games,
                        "n_replications": n_replications,
                    }
                )

    grid = pd.DataFrame(grid_rows)
    optima_rows = []
    for (dynamics, tau), group in grid.groupby(["dynamics", "tau"], sort=False):
        group = group.sort_values("alpha")
        row = group.loc[group["tracking_mse"].idxmin()].to_dict()
        row["refined_alpha"] = _refined_optimal_alpha(
            group["alpha"].to_numpy(dtype=float),
            group["tracking_mse"].to_numpy(dtype=float),
        )
        row["refined_k_equivalent"] = alpha_to_k(row["refined_alpha"])
        optima_rows.append(row)
    optima = pd.DataFrame(optima_rows)
    theoretical_slopes = {"smooth": -2.0 / 3.0, "ou": -0.5, "jump": np.nan}
    scaling_rows = []
    bootstrap_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    for dynamics, group in optima.groupby("dynamics", sort=False):
        group = group.sort_values("tau")
        x = np.log(group["tau"].to_numpy(dtype=float))
        y = np.log(group["refined_alpha"].to_numpy(dtype=float))
        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        denominator = np.sum((y - np.mean(y)) ** 2)
        r_squared = (
            1.0 - np.sum((y - fitted) ** 2) / denominator
            if denominator > 0
            else np.nan
        )

        bootstrap_optimal_alpha = np.empty(
            (n_mc_resamples, len(group)), dtype=float
        )
        for tau_index, tau in enumerate(group["tau"].astype(int)):
            mse = replication_mse[(dynamics, tau)]
            draws = bootstrap_rng.integers(
                0,
                n_replications,
                size=(n_mc_resamples, n_replications),
            )
            bootstrap_mse = mse[draws].mean(axis=1)
            bootstrap_optimal_alpha[:, tau_index] = np.array(
                [
                    _refined_optimal_alpha(alphas, replicate_mse)
                    for replicate_mse in bootstrap_mse
                ]
            )
        bootstrap_slopes = np.array(
            [
                np.polyfit(x, np.log(optimal_alpha), 1)[0]
                for optimal_alpha in bootstrap_optimal_alpha
            ]
        )
        scaling_rows.append(
            {
                "dynamics": dynamics,
                "fitted_slope": slope,
                "mc_ci_2_5": float(np.quantile(bootstrap_slopes, 0.025)),
                "mc_ci_97_5": float(np.quantile(bootstrap_slopes, 0.975)),
                "mc_standard_error": float(bootstrap_slopes.std(ddof=1)),
                "theoretical_slope": theoretical_slopes.get(dynamics, np.nan),
                "r_squared": r_squared,
                "n_tau_values": len(group),
                "n_replications": n_replications,
                "n_mc_resamples": n_mc_resamples,
                "interval_type": "replication-bootstrap Monte Carlo uncertainty",
            }
        )
    return DynamicTrackingResult(
        grid=grid,
        optima=optima,
        scaling=pd.DataFrame(scaling_rows),
    )
