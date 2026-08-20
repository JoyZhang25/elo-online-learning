"""Chronological ATP evaluation with locked model and strategy selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .metrics import forecast_metrics, per_game_log_loss
from .model import EloModel, k_to_alpha
from .tennis import MixtureElo, SurfaceElo


@dataclass
class TennisEvaluationResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    candidate_metrics: pd.DataFrame
    overlay_validation: pd.DataFrame
    uncertainty: pd.DataFrame
    model_comparison_uncertainty: pd.DataFrame
    yearly_stability: pd.DataFrame
    strategy_validation: pd.DataFrame
    strategy_test: pd.DataFrame
    selected_model: str
    selected_params: dict[str, Any]
    validation_start: pd.Timestamp
    test_start: pd.Timestamp


def _ordered_matches(matches: pd.DataFrame) -> pd.DataFrame:
    required = {
        "match_id",
        "date",
        "player_a",
        "player_b",
        "surface",
        "outcome_a",
    }
    missing = required.difference(matches.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    ordered = matches.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="raise").dt.normalize()
    if ordered.empty:
        raise ValueError("matches must be nonempty")
    if ordered["date"].isna().any():
        raise ValueError("match dates must be nonmissing")

    match_id = ordered["match_id"].astype("string").str.strip()
    if match_id.isna().any() or match_id.eq("").any() or match_id.duplicated().any():
        raise ValueError("match_id values must be nonempty and unique")
    ordered["match_id"] = match_id

    player_a = ordered["player_a"].astype("string").str.strip()
    player_b = ordered["player_b"].astype("string").str.strip()
    invalid_players = (
        player_a.isna()
        | player_b.isna()
        | player_a.eq("")
        | player_b.eq("")
        | player_a.str.casefold().eq(player_b.str.casefold())
    )
    if invalid_players.any():
        raise ValueError("each match must contain two distinct nonempty player names")
    ordered["player_a"] = player_a
    ordered["player_b"] = player_b

    surface = ordered["surface"].astype("string").str.strip()
    if surface.isna().any() or surface.eq("").any():
        raise ValueError("surface values must be nonempty")
    ordered["surface"] = surface

    outcome = pd.to_numeric(ordered["outcome_a"], errors="coerce")
    if not np.isfinite(outcome).all() or not outcome.isin([0.0, 1.0]).all():
        raise ValueError("outcome_a values must be binary")
    ordered["outcome_a"] = outcome.astype(float)

    ordered = ordered.sort_values(["date", "match_id"], kind="stable").reset_index(
        drop=True
    )
    return ordered


def walk_forward_overall(matches: pd.DataFrame, *, k_factor: float) -> np.ndarray:
    """Predict every date as a batch, then update a standard Elo model."""

    ordered = _ordered_matches(matches)
    model = EloModel(k_factor=k_factor)
    predictions = np.empty(len(ordered), dtype=float)
    for _, day in ordered.groupby("date", sort=False):
        pending: list[tuple[int, str, str, float, float]] = []
        for index, row in day.iterrows():
            player_a = str(row["player_a"])
            player_b = str(row["player_b"])
            probability = model.predict(player_a, player_b)
            predictions[index] = probability
            pending.append(
                (index, player_a, player_b, float(row["outcome_a"]), probability)
            )
        for _, player_a, player_b, outcome, probability in pending:
            model.update(
                player_a,
                player_b,
                outcome,
                prediction_a=probability,
            )
    return predictions


def walk_forward_surface(
    matches: pd.DataFrame,
    *,
    k_global: float,
    k_surface: float,
) -> np.ndarray:
    """Walk forward a global-plus-surface Elo model."""

    ordered = _ordered_matches(matches)
    model = SurfaceElo(k_global=k_global, k_surface=k_surface)
    predictions = np.empty(len(ordered), dtype=float)
    for _, day in ordered.groupby("date", sort=False):
        pending: list[tuple[str, str, str, float, float]] = []
        for index, row in day.iterrows():
            player_a = str(row["player_a"])
            player_b = str(row["player_b"])
            surface = str(row["surface"])
            probability = model.predict(player_a, player_b, surface)
            predictions[index] = probability
            pending.append(
                (player_a, player_b, surface, float(row["outcome_a"]), probability)
            )
        for player_a, player_b, surface, outcome, probability in pending:
            model.update(player_a, player_b, surface, outcome, probability)
    return predictions


def walk_forward_mixture(
    matches: pd.DataFrame,
    *,
    k_grid: tuple[float, ...],
    learning_rate: float,
) -> np.ndarray:
    """Walk forward an online mixture of multiple Elo response horizons."""

    ordered = _ordered_matches(matches)
    model = MixtureElo(k_grid=k_grid, learning_rate=learning_rate)
    predictions = np.empty(len(ordered), dtype=float)
    for _, day in ordered.groupby("date", sort=False):
        pending: list[tuple[str, str, float, np.ndarray]] = []
        for index, row in day.iterrows():
            player_a = str(row["player_a"])
            player_b = str(row["player_b"])
            probability, expert_probabilities = model.predict(player_a, player_b)
            predictions[index] = probability
            pending.append(
                (
                    player_a,
                    player_b,
                    float(row["outcome_a"]),
                    expert_probabilities,
                )
            )
        for player_a, player_b, outcome, expert_probabilities in pending:
            model.update(player_a, player_b, outcome, expert_probabilities)
    return predictions


def _attach_market(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    pinnacle = (
        result.get("pinnacle_odds_a", pd.Series(np.nan, index=result.index)).gt(1.0)
        & result.get("pinnacle_odds_b", pd.Series(np.nan, index=result.index)).gt(1.0)
    )
    average = (
        result.get("average_odds_a", pd.Series(np.nan, index=result.index)).gt(1.0)
        & result.get("average_odds_b", pd.Series(np.nan, index=result.index)).gt(1.0)
    )
    result["market_source"] = np.where(
        pinnacle, "Pinnacle", np.where(average, "bookmaker average", pd.NA)
    )
    result["market_odds_a"] = result.get(
        "pinnacle_odds_a", pd.Series(np.nan, index=result.index)
    ).where(pinnacle, result.get("average_odds_a", np.nan))
    result["market_odds_b"] = result.get(
        "pinnacle_odds_b", pd.Series(np.nan, index=result.index)
    ).where(pinnacle, result.get("average_odds_b", np.nan))
    valid = pinnacle | average
    inverse_a = 1.0 / result["market_odds_a"]
    inverse_b = 1.0 / result["market_odds_b"]
    result["market_overround"] = (inverse_a + inverse_b).where(valid)
    result["market_probability_a"] = (
        inverse_a / (inverse_a + inverse_b)
    ).where(valid)
    return result


def _candidate_row(
    model: str,
    outcome: np.ndarray,
    probability: np.ndarray,
    **parameters: float | str,
) -> dict[str, float | str]:
    return {"model": model, **parameters, **forecast_metrics(outcome, probability)}


def _logit(probability: np.ndarray | pd.Series) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-8, 1.0 - 1e-8)
    return np.log(clipped / (1.0 - clipped))


def _inverse_logit(value: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -np.asarray(value, dtype=float)))


def _metric_rows(
    frame: pd.DataFrame,
    probability_columns: dict[str, str],
    *,
    sample: str,
) -> list[dict[str, float | str]]:
    rows = []
    for model, column in probability_columns.items():
        subset = frame.loc[frame[column].notna()]
        if subset.empty:
            continue
        rows.append(
            {
                "model": model,
                "sample": sample,
                **forecast_metrics(
                    subset["outcome_a"].to_numpy(), subset[column].to_numpy()
                ),
            }
        )
    return rows


def paired_month_block_bootstrap(
    frame: pd.DataFrame,
    left_probability_column: str,
    right_probability_column: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap the mean paired log-loss difference using month blocks.

    The reported estimand is left-column loss minus right-column loss. Both
    losses are evaluated on exactly the same matches before calendar months
    are resampled with replacement.
    """

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    required = {
        "date",
        "outcome_a",
        left_probability_column,
        right_probability_column,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing bootstrap columns: {sorted(missing)}")
    subset = frame.loc[
        frame[left_probability_column].notna()
        & frame[right_probability_column].notna()
    ].copy()
    if subset.empty:
        raise ValueError("bootstrap sample must contain paired forecasts")
    subset["date"] = pd.to_datetime(subset["date"], errors="raise")
    left_loss = per_game_log_loss(
        subset["outcome_a"].to_numpy(),
        subset[left_probability_column].to_numpy(),
    )
    right_loss = per_game_log_loss(
        subset["outcome_a"].to_numpy(),
        subset[right_probability_column].to_numpy(),
    )
    subset["loss_difference"] = left_loss - right_loss
    subset["month"] = subset["date"].dt.to_period("M")
    blocks = subset.groupby("month")["loss_difference"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(blocks), size=(n_bootstrap, len(blocks)))
    block_sums = blocks["sum"].to_numpy()
    block_counts = blocks["count"].to_numpy()
    bootstrap_means = block_sums[draws].sum(axis=1) / block_counts[draws].sum(axis=1)
    return {
        "n_matches": int(len(subset)),
        "mean_log_loss_difference": float(subset["loss_difference"].mean()),
        "ci_2_5": float(np.quantile(bootstrap_means, 0.025)),
        "ci_97_5": float(np.quantile(bootstrap_means, 0.975)),
        "n_month_blocks": int(len(blocks)),
    }


def _yearly_model_stability(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in test.groupby(test["date"].dt.year, sort=True):
        outcome = group["outcome_a"].to_numpy()
        overall_loss = float(
            per_game_log_loss(outcome, group["overall_probability"].to_numpy()).mean()
        )
        surface_loss = float(
            per_game_log_loss(outcome, group["surface_probability"].to_numpy()).mean()
        )
        rows.append(
            {
                "year": int(year),
                "n_matches": int(len(group)),
                "overall_log_loss": overall_loss,
                "surface_log_loss": surface_loss,
                "surface_minus_overall": surface_loss - overall_loss,
            }
        )
    return pd.DataFrame(rows)


def _strategy_summary(
    frame: pd.DataFrame,
    probability_column: str,
    *,
    threshold: float,
    split: str,
    kelly_multiplier: float = 0.10,
    max_stake: float = 0.02,
    abstain: bool = False,
) -> dict[str, float | int | str]:
    path_note = (
        "order-dependent; no intraday order or simultaneous-exposure model"
    )
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    if not np.isfinite(kelly_multiplier) or not 0.0 <= kelly_multiplier <= 1.0:
        raise ValueError("kelly_multiplier must lie in [0, 1]")
    if not np.isfinite(max_stake) or not 0.0 <= max_stake <= 1.0:
        raise ValueError("max_stake must lie in [0, 1]")
    if abstain:
        return {
            "split": split,
            "policy": "abstain",
            "threshold": np.nan,
            "n_bets": 0,
            "hit_rate": np.nan,
            "unit_roi": np.nan,
            "roi_ci_2_5": np.nan,
            "roi_ci_97_5": np.nan,
            "final_bankroll": 1.0,
            "max_drawdown": 0.0,
            "mean_edge": np.nan,
            "path_statistics_note": path_note,
        }
    subset = frame.loc[
        (frame["split"] == split)
        & frame[probability_column].notna()
        & frame["market_probability_a"].notna()
    ].copy()
    model_p = subset[probability_column].to_numpy()
    market_p = subset["market_probability_a"].to_numpy()
    bet_a = model_p >= market_p
    side_probability = np.where(bet_a, model_p, 1.0 - model_p)
    side_market_probability = np.where(bet_a, market_p, 1.0 - market_p)
    side_odds = np.where(
        bet_a, subset["market_odds_a"].to_numpy(), subset["market_odds_b"].to_numpy()
    )
    side_outcome = np.where(
        bet_a, subset["outcome_a"].to_numpy(), 1.0 - subset["outcome_a"].to_numpy()
    )
    edge = side_probability - side_market_probability
    expected_return = side_probability * side_odds - 1.0
    selected = (edge >= threshold) & (expected_return > 0.0)
    unit_return = side_outcome[selected] * side_odds[selected] - 1.0
    if unit_return.size == 0:
        return {
            "split": split,
            "policy": "trade",
            "threshold": threshold,
            "n_bets": 0,
            "hit_rate": np.nan,
            "unit_roi": np.nan,
            "roi_ci_2_5": np.nan,
            "roi_ci_97_5": np.nan,
            "final_bankroll": 1.0,
            "max_drawdown": 0.0,
            "mean_edge": np.nan,
            "path_statistics_note": path_note,
        }
    chosen_probability = side_probability[selected]
    chosen_odds = side_odds[selected]
    full_kelly = np.maximum(
        (chosen_probability * chosen_odds - 1.0) / (chosen_odds - 1.0), 0.0
    )
    stake_fraction = np.minimum(max_stake, kelly_multiplier * full_kelly)
    wealth = np.cumprod(1.0 + stake_fraction * unit_return)
    running_peak = np.maximum.accumulate(np.r_[1.0, wealth])
    drawdown = 1.0 - np.r_[1.0, wealth] / running_peak
    bet_month = subset.loc[selected, "date"].dt.to_period("M")
    return_blocks = pd.DataFrame(
        {"month": bet_month.to_numpy(), "unit_return": unit_return}
    ).groupby("month")["unit_return"].agg(["sum", "count"])
    rng = np.random.default_rng(
        20260821 + int(round(threshold * 10_000)) + (0 if split == "validation" else 1)
    )
    draws = rng.integers(
        0, len(return_blocks), size=(2_000, len(return_blocks))
    )
    bootstrap_roi = (
        return_blocks["sum"].to_numpy()[draws].sum(axis=1)
        / return_blocks["count"].to_numpy()[draws].sum(axis=1)
    )
    return {
        "split": split,
        "policy": "trade",
        "threshold": threshold,
        "n_bets": int(unit_return.size),
        "hit_rate": float(np.mean(side_outcome[selected])),
        "unit_roi": float(np.mean(unit_return)),
        "roi_ci_2_5": float(np.quantile(bootstrap_roi, 0.025)),
        "roi_ci_97_5": float(np.quantile(bootstrap_roi, 0.975)),
        "final_bankroll": float(wealth[-1]),
        "max_drawdown": float(np.max(drawdown)),
        "mean_edge": float(np.mean(edge[selected])),
        "path_statistics_note": path_note,
    }


def evaluate_tennis(
    matches: pd.DataFrame,
    *,
    validation_start: str | pd.Timestamp = "2018-01-01",
    test_start: str | pd.Timestamp = "2022-01-01",
    k_grid: tuple[float, ...] = (4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0),
    surface_k_grid: tuple[float, ...] = (0.0, 4.0, 8.0, 16.0, 24.0),
    mixture_learning_rates: tuple[float, ...] = (0.02, 0.05, 0.10, 0.25),
    overlay_weights: tuple[float, ...] = (
        -0.50,
        -0.25,
        0.0,
        0.10,
        0.20,
        0.35,
        0.50,
        0.75,
        1.0,
    ),
    overlay_biases: tuple[float, ...] = (-0.10, -0.05, 0.0, 0.05, 0.10),
    strategy_thresholds: tuple[float, ...] = (0.00, 0.02, 0.04, 0.06, 0.08),
    n_bootstrap: int = 2_000,
) -> TennisEvaluationResult:
    """Select on a historical block and report a later held-out ATP block."""

    ordered = _attach_market(_ordered_matches(matches))
    validation_start = pd.Timestamp(validation_start).normalize()
    test_start = pd.Timestamp(test_start).normalize()
    if validation_start >= test_start:
        raise ValueError("validation_start must precede test_start")
    if not k_grid:
        raise ValueError("k_grid must be nonempty")
    if not surface_k_grid:
        raise ValueError("surface_k_grid must be nonempty")
    if not mixture_learning_rates:
        raise ValueError("mixture_learning_rates must be nonempty")
    if not overlay_weights or not np.isfinite(overlay_weights).all():
        raise ValueError("overlay_weights must contain finite values")
    if not overlay_biases or not np.isfinite(overlay_biases).all():
        raise ValueError("overlay_biases must contain finite values")
    if not strategy_thresholds or any(
        not np.isfinite(threshold) or threshold < 0
        for threshold in strategy_thresholds
    ):
        raise ValueError("strategy_thresholds must contain nonnegative values")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    validation_mask = (ordered["date"] >= validation_start) & (
        ordered["date"] < test_start
    )
    test_mask = ordered["date"] >= test_start
    if (
        not (ordered["date"] < validation_start).any()
        or not validation_mask.any()
        or not test_mask.any()
    ):
        raise ValueError("warm-up, validation, and test periods must all be nonempty")
    y_validation = ordered.loc[validation_mask, "outcome_a"].to_numpy()

    candidate_rows: list[dict[str, float | str]] = []
    prediction_cache: dict[tuple[Any, ...], np.ndarray] = {}

    for k_factor in k_grid:
        probability = walk_forward_overall(ordered, k_factor=k_factor)
        prediction_cache[("overall", k_factor)] = probability
        candidate_rows.append(
            _candidate_row(
                "Overall Elo",
                y_validation,
                probability[validation_mask],
                k_factor=k_factor,
                alpha=k_to_alpha(k_factor),
            )
        )

    for k_global in k_grid:
        for k_surface in surface_k_grid:
            probability = walk_forward_surface(
                ordered, k_global=k_global, k_surface=k_surface
            )
            prediction_cache[("surface", k_global, k_surface)] = probability
            candidate_rows.append(
                _candidate_row(
                    "Surface Elo",
                    y_validation,
                    probability[validation_mask],
                    k_factor=k_global,
                    alpha=k_to_alpha(k_global),
                    k_surface=k_surface,
                )
            )

    for learning_rate in mixture_learning_rates:
        probability = walk_forward_mixture(
            ordered, k_grid=k_grid, learning_rate=learning_rate
        )
        prediction_cache[("mixture", learning_rate)] = probability
        candidate_rows.append(
            _candidate_row(
                "Multi-timescale Elo",
                y_validation,
                probability[validation_mask],
                learning_rate=learning_rate,
                expert_k_grid="|".join(f"{k:g}" for k in k_grid),
            )
        )

    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["log_loss", "brier_score", "model"], kind="stable"
    ).reset_index(drop=True)
    family_best = candidates.groupby("model", sort=False, as_index=False).first()
    selected = family_best.sort_values(
        ["log_loss", "brier_score"], kind="stable"
    ).iloc[0]

    best_overall = family_best.loc[family_best["model"] == "Overall Elo"].iloc[0]
    best_surface = family_best.loc[family_best["model"] == "Surface Elo"].iloc[0]
    best_mixture = family_best.loc[
        family_best["model"] == "Multi-timescale Elo"
    ].iloc[0]
    overall_probability = prediction_cache[("overall", float(best_overall["k_factor"]))]
    surface_probability = prediction_cache[
        (
            "surface",
            float(best_surface["k_factor"]),
            float(best_surface["k_surface"]),
        )
    ]
    mixture_probability = prediction_cache[
        ("mixture", float(best_mixture["learning_rate"]))
    ]
    ordered["overall_probability"] = overall_probability
    ordered["surface_probability"] = surface_probability
    ordered["mixture_probability"] = mixture_probability
    ordered["no_skill_probability"] = 0.5

    selected_model = str(selected["model"])
    selected_column = {
        "Overall Elo": "overall_probability",
        "Surface Elo": "surface_probability",
        "Multi-timescale Elo": "mixture_probability",
    }[selected_model]
    ordered["selected_probability"] = ordered[selected_column]
    ordered["split"] = np.where(
        ordered["date"] < validation_start,
        "warmup",
        np.where(ordered["date"] < test_start, "validation", "test"),
    )

    overlay_mask = validation_mask & ordered["market_probability_a"].notna()
    if not overlay_mask.any():
        raise ValueError("validation period must contain market-covered matches")
    overlay_rows = []
    market_logit = _logit(ordered.loc[overlay_mask, "market_probability_a"])
    elo_logit = _logit(ordered.loc[overlay_mask, selected_column])
    overlay_outcome = ordered.loc[overlay_mask, "outcome_a"].to_numpy()
    for weight in overlay_weights:
        for bias in overlay_biases:
            probability = _inverse_logit(
                market_logit + bias + weight * (elo_logit - market_logit)
            )
            overlay_rows.append(
                _candidate_row(
                    "Market + Elo overlay",
                    overlay_outcome,
                    probability,
                    overlay_weight=weight,
                    overlay_bias=bias,
                )
            )
    overlay_validation = pd.DataFrame(overlay_rows).sort_values(
        ["log_loss", "brier_score", "overlay_weight", "overlay_bias"],
        kind="stable",
    ).reset_index(drop=True)
    best_overlay = overlay_validation.iloc[0]
    market_available = ordered["market_probability_a"].notna()
    ordered["overlay_probability"] = np.nan
    ordered.loc[market_available, "overlay_probability"] = _inverse_logit(
        _logit(ordered.loc[market_available, "market_probability_a"])
        + float(best_overlay["overlay_bias"])
        + float(best_overlay["overlay_weight"])
        * (
            _logit(ordered.loc[market_available, selected_column])
            - _logit(ordered.loc[market_available, "market_probability_a"])
        )
    )

    test = ordered.loc[test_mask]
    elo_probability_columns = {
        "Overall Elo": "overall_probability",
        "Surface Elo": "surface_probability",
        "Multi-timescale Elo": "mixture_probability",
        "No-skill 50%": "no_skill_probability",
    }
    metric_rows = _metric_rows(test, elo_probability_columns, sample="all test matches")
    market_test = test.loc[test["market_probability_a"].notna()]
    if market_test.empty:
        raise ValueError("test period must contain market-covered matches")
    market_probability_columns = {
        **elo_probability_columns,
        "De-vigged market": "market_probability_a",
        "Market + Elo overlay": "overlay_probability",
    }
    metric_rows.extend(
        _metric_rows(
            market_test,
            market_probability_columns,
            sample="market-covered test matches",
        )
    )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["sample", "log_loss"], kind="stable"
    )

    uncertainty_rows = []
    bootstrap_columns = {
        **elo_probability_columns,
        "Market + Elo overlay": "overlay_probability",
    }
    for index, (model, column) in enumerate(bootstrap_columns.items()):
        interval = paired_month_block_bootstrap(
            market_test,
            column,
            "market_probability_a",
            n_bootstrap=n_bootstrap,
            seed=20260820 + index,
        )
        uncertainty_rows.append(
            {"model": model, "benchmark": "De-vigged market", **interval}
        )
    uncertainty = pd.DataFrame(uncertainty_rows)

    surface_overall_interval = paired_month_block_bootstrap(
        test,
        "surface_probability",
        "overall_probability",
        n_bootstrap=n_bootstrap,
        seed=20260901,
    )
    model_comparison_uncertainty = pd.DataFrame(
        [
            {
                "model": "Surface Elo",
                "benchmark": "Overall Elo",
                "difference": "Surface minus Overall log loss",
                **surface_overall_interval,
            }
        ]
    )
    yearly_stability = _yearly_model_stability(test)

    validation_strategy = pd.DataFrame(
        [
            _strategy_summary(
                ordered,
                "overlay_probability",
                threshold=threshold,
                split="validation",
            )
            for threshold in strategy_thresholds
        ]
    )
    validation_strategy = pd.concat(
        [
            validation_strategy,
            pd.DataFrame(
                [
                    _strategy_summary(
                        ordered,
                        "overlay_probability",
                        threshold=0.0,
                        split="validation",
                        abstain=True,
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    eligible = validation_strategy.loc[
        (validation_strategy["policy"] == "abstain")
        | (validation_strategy["n_bets"] >= 100)
    ]
    chosen_policy = eligible.sort_values(
        ["final_bankroll", "max_drawdown"],
        ascending=[False, True],
        kind="stable",
    ).iloc[0]
    should_abstain = str(chosen_policy["policy"]) == "abstain"
    chosen_threshold = (
        np.nan if should_abstain else float(chosen_policy["threshold"])
    )
    strategy_test = pd.DataFrame(
        [
            _strategy_summary(
                ordered,
                "overlay_probability",
                threshold=0.0 if should_abstain else chosen_threshold,
                split="test",
                abstain=should_abstain,
            )
        ]
    )

    if selected_model == "Overall Elo":
        selected_params = {
            "k_factor": float(selected["k_factor"]),
            "alpha": float(selected["alpha"]),
        }
    elif selected_model == "Surface Elo":
        selected_params = {
            "k_global": float(selected["k_factor"]),
            "k_surface": float(selected["k_surface"]),
            "alpha_global": float(selected["alpha"]),
        }
    else:
        selected_params = {
            "learning_rate": float(selected["learning_rate"]),
            "expert_k_grid": tuple(k_grid),
        }
    selected_params.update(
        {
            "overlay_weight": float(best_overlay["overlay_weight"]),
            "overlay_bias": float(best_overlay["overlay_bias"]),
            "strategy_policy": "abstain" if should_abstain else "trade",
            "strategy_threshold": chosen_threshold,
        }
    )

    return TennisEvaluationResult(
        predictions=ordered,
        metrics=metrics.reset_index(drop=True),
        candidate_metrics=candidates,
        overlay_validation=overlay_validation,
        uncertainty=uncertainty,
        model_comparison_uncertainty=model_comparison_uncertainty,
        yearly_stability=yearly_stability,
        strategy_validation=validation_strategy,
        strategy_test=strategy_test,
        selected_model=selected_model,
        selected_params=selected_params,
        validation_start=validation_start,
        test_start=test_start,
    )
