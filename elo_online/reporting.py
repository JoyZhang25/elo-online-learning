"""Build the figures and tables used by the repository."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from statistics import NormalDist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import DEFAULT_TOURNAMENT_ID, load_tournament_games
from .evaluation import RealDataResult, evaluate_real_data
from .metrics import calibration_table
from .simulation import (
    change_point_experiment,
    invariant_fluctuation_experiment,
    stability_adaptation_experiment,
)


COLORS = ["#276FBF", "#F18F01", "#2A9D8F", "#8E5EA2", "#C44536"]


def _output_dirs(root: Path) -> tuple[Path, Path]:
    figures = root / "docs" / "figures"
    tables = root / "docs" / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return figures, tables


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_simulation_outputs(
    root: str | Path = PROJECT_ROOT,
) -> dict[str, pd.DataFrame]:
    """Run the controlled experiments and save their figures and tables."""

    root = Path(root)
    figures, tables = _output_dirs(root)
    invariant = invariant_fluctuation_experiment()
    tradeoff = stability_adaptation_experiment()
    change = change_point_experiment()
    invariant.summary.to_csv(
        tables / "invariant_fluctuation_summary.csv", index=False
    )
    tradeoff.to_csv(tables / "stability_adaptation.csv", index=False)
    change.summary.to_csv(tables / "change_point_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.1))
    ax = axes[0]
    true_score_probability = 0.70
    rating_difference = np.linspace(-1200.0, 1200.0, 600)
    logistic_scale = np.log(10.0) / 400.0
    population_loss = np.logaddexp(
        0.0, logistic_scale * rating_difference
    ) - true_score_probability * logistic_scale * rating_difference
    optimum = 400.0 * np.log10(
        true_score_probability / (1.0 - true_score_probability)
    )
    minimum_loss = -(
        true_score_probability * np.log(true_score_probability)
        + (1.0 - true_score_probability)
        * np.log(1.0 - true_score_probability)
    )
    curvature = (
        logistic_scale**2
        * true_score_probability
        * (1.0 - true_score_probability)
    )
    local_quadratic = minimum_loss + 0.5 * curvature * (
        rating_difference - optimum
    ) ** 2
    ax.plot(
        rating_difference,
        population_loss,
        color=COLORS[0],
        linewidth=2.3,
        label="logistic population loss",
    )
    ax.plot(
        rating_difference,
        local_quadratic,
        color="0.35",
        linestyle="--",
        linewidth=1.7,
        label="local quadratic approximation",
    )
    ax.axvline(optimum, color="0.55", linestyle=":", linewidth=1)
    ax.set_xlabel("rating difference")
    ax.set_ylabel("expected logistic loss")
    ax.set_title("A. Linear tails, quadratic minimum")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    summary = invariant.summary.sort_values("k_factor")
    reference_slope = float(summary["variance_over_k"].median())
    ax.plot(
        summary["k_factor"],
        summary["variance"],
        marker="o",
        linewidth=2.2,
        color=COLORS[1],
        label="simulated variance",
    )
    ax.plot(
        summary["k_factor"],
        reference_slope * summary["k_factor"],
        linestyle="--",
        linewidth=1.6,
        color="0.35",
        label="reference proportional to K",
    )
    ax.set_xlabel("Elo step size K")
    ax.set_ylabel("stationary rating-error variance")
    ax.set_title(r"B. Fluctuation scale is $\sqrt{K}$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    probability_grid = (np.arange(1, 200) - 0.5) / 199.0
    normal_quantiles = np.array(
        [NormalDist().inv_cdf(probability) for probability in probability_grid]
    )
    shown_k = (2.0, 8.0, 32.0)
    for color, k_factor in zip(COLORS, shown_k):
        standardized = invariant.samples.loc[
            invariant.samples["k_factor"] == k_factor, "standardized_error"
        ].to_numpy()
        empirical_quantiles = np.quantile(standardized, probability_grid)
        ax.plot(
            normal_quantiles,
            empirical_quantiles,
            color=color,
            linewidth=1.8,
            label=f"K={k_factor:g}",
        )
    limits = (-2.8, 2.8)
    ax.plot(limits, limits, color="0.35", linestyle="--", linewidth=1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("standard-normal quantile")
    ax.set_ylabel("empirical quantile")
    ax.set_title("C. Standardized errors are near Gaussian")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "The Elo objective links global tail shape to a local Gaussian regime",
        y=1.03,
        fontsize=14,
    )
    fig.tight_layout()
    _save_figure(fig, figures / "theory_bridge.png")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    ax = axes[0]
    for color, (drift, group) in zip(
        COLORS, tradeoff.groupby("drift_sd_per_event", sort=True)
    ):
        label = "stationary skill" if drift == 0 else f"skill-drift SD={drift:g}"
        ax.plot(
            group["k_factor"],
            group["tracking_rmse"],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
        best = group.loc[group["tracking_rmse"].idxmin()]
        ax.scatter(
            best["k_factor"],
            best["tracking_rmse"],
            s=85,
            facecolors="none",
            edgecolors=color,
            linewidths=2,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(tradeoff["k_factor"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Elo step size K")
    ax.set_ylabel("post-burn-in rating RMSE")
    ax.set_title("A. Stability–adaptation trade-off")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    plot_columns = [
        column
        for column in change.trajectories.columns
        if column.startswith("constant") or column == "decaying K"
    ]
    ax.plot(
        change.trajectories["event_index"],
        change.trajectories["true_rating"],
        color="black",
        linewidth=2.5,
        label="latent skill",
    )
    for color, column in zip(COLORS, plot_columns):
        smooth = change.trajectories[column].ewm(span=60, adjust=False).mean()
        ax.plot(
            change.trajectories["event_index"],
            smooth,
            color=color,
            linewidth=1.8,
            label=column,
        )
    ax.axvline(2000, color="0.45", linestyle="--", linewidth=1)
    ax.set_xlabel("match involving focal player")
    ax.set_ylabel("rating")
    ax.set_title("B. Tracking a sudden skill change")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(
        "Constant stepsize controls variance and adaptation speed",
        y=1.02,
        fontsize=14,
    )
    fig.tight_layout()
    _save_figure(fig, figures / "simulation_tradeoff.png")
    return {
        "invariant_fluctuation_summary": invariant.summary,
        "stability_adaptation": tradeoff,
        "change_point_summary": change.summary,
    }


def _anonymous_player(identifier: str) -> str:
    digest = hashlib.sha256(
        f"{DEFAULT_TOURNAMENT_ID}:{identifier}".encode("utf-8")
    ).hexdigest()
    return f"player_{digest[:10]}"


def _calibration_outputs(result: RealDataResult) -> pd.DataFrame:
    test = result.predictions.loc[result.predictions["split"] == "test"]
    frames = []
    columns = {
        "Constant-step Elo": "elo_probability",
        "Frozen initial ratings": "frozen_probability",
        "Lichess pre-game ratings": "platform_probability",
    }
    for name, column in columns.items():
        table = calibration_table(
            test["white_score"].to_numpy(), test[column].to_numpy()
        )
        table.insert(0, "model", name)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def _real_data_overview(games: pd.DataFrame) -> pd.DataFrame:
    appearances = pd.concat([games["white_id"], games["black_id"]]).value_counts()
    return pd.DataFrame(
        {
            "statistic": [
                "eligible games",
                "unique players",
                "draw rate",
                "median games per player",
                "90th percentile games per player",
                "first game (UTC)",
                "last game (UTC)",
            ],
            "value": [
                len(games),
                appearances.size,
                f"{(games['white_score'] == 0.5).mean():.3%}",
                f"{appearances.median():.1f}",
                f"{appearances.quantile(0.9):.1f}",
                games["created_at"].min().isoformat(),
                games["created_at"].max().isoformat(),
            ],
        }
    )


def build_real_data_outputs(
    root: str | Path = PROJECT_ROOT,
    *,
    refresh: bool = False,
) -> tuple[pd.DataFrame, RealDataResult]:
    """Download the fixed sample, run model selection, and save audit outputs."""

    root = Path(root)
    figures, tables = _output_dirs(root)
    games = load_tournament_games(
        cache_dir=root / "data" / "cache", refresh=refresh
    )
    result = evaluate_real_data(games)
    calibration = _calibration_outputs(result)
    overview = _real_data_overview(games)

    result.metrics.to_csv(tables / "real_test_metrics.csv", index=False)
    result.candidate_metrics.to_csv(
        tables / "real_validation_grid.csv", index=False
    )
    calibration.to_csv(tables / "real_calibration.csv", index=False)
    overview.to_csv(tables / "real_data_overview.csv", index=False)

    public_predictions = result.predictions.copy()
    public_predictions["white_id"] = public_predictions["white_id"].map(
        _anonymous_player
    )
    public_predictions["black_id"] = public_predictions["black_id"].map(
        _anonymous_player
    )
    public_predictions.to_csv(tables / "walk_forward_predictions.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))
    ax = axes[0]
    best_by_k = (
        result.candidate_metrics.sort_values("log_loss")
        .groupby("k_factor", as_index=False)
        .first()
        .sort_values("k_factor")
    )
    ax.plot(
        best_by_k["k_factor"],
        best_by_k["log_loss"],
        marker="o",
        linewidth=2,
        color=COLORS[1],
    )
    chosen = best_by_k.loc[best_by_k["k_factor"] == result.selected_k].iloc[0]
    ax.scatter(
        [chosen["k_factor"]],
        [chosen["log_loss"]],
        s=110,
        facecolors="none",
        edgecolors="black",
        linewidths=2,
        label=f"selected K={result.selected_k:g}",
    )
    ax.set_xlabel("Elo step size K")
    ax.set_ylabel("validation log loss")
    ax.set_title("A. Validation model selection")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    for color, (model, group) in zip(COLORS, calibration.groupby("model")):
        ax.plot(
            group["mean_prediction"],
            group["empirical_score"],
            marker="o",
            linewidth=1.8,
            color=color,
            label=model,
        )
    ax.plot([0, 1], [0, 1], color="0.35", linestyle="--", linewidth=1)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.05, 0.95)
    ax.set_xlabel("mean predicted score")
    ax.set_ylabel("empirical white score")
    ax.set_title("B. Held-out calibration")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    test = result.predictions.loc[result.predictions["split"] == "test"]
    ax.plot(
        np.arange(len(test)),
        test["cumulative_log_loss_vs_frozen"],
        color=COLORS[2],
        linewidth=2,
    )
    ax.axhline(0, color="0.35", linestyle="--", linewidth=1)
    ax.set_xlabel("held-out game")
    ax.set_ylabel("cumulative loss: online − frozen")
    ax.set_title("C. Value of online updating")
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Walk-forward Elo on a fixed Lichess Daily Blitz Arena",
        y=1.01,
        fontsize=14,
    )
    fig.tight_layout()
    _save_figure(fig, figures / "real_data_evaluation.png")
    return games, result
