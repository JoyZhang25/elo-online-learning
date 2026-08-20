"""Build the figures and aggregate tables committed to the repository."""

from __future__ import annotations

import os
from pathlib import Path
from statistics import NormalDist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import (
    DEFAULT_END_YEAR,
    DEFAULT_START_YEAR,
    load_tennis_matches,
)
from .evaluation import TennisEvaluationResult, evaluate_tennis
from .metrics import calibration_table, per_game_log_loss
from .simulation import dynamic_tracking_experiment, invariant_fluctuation_experiment


COLORS = ["#176B87", "#D1495B", "#2E8B57", "#6F4E7C", "#D17A22"]


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
    """Run the stationary and dynamic-skill experiments."""

    root = Path(root)
    figures, tables = _output_dirs(root)
    invariant = invariant_fluctuation_experiment()
    tracking = dynamic_tracking_experiment()
    invariant.summary.to_csv(
        tables / "invariant_fluctuation_summary.csv", index=False
    )
    tracking.grid.to_csv(tables / "dynamic_tracking_grid.csv", index=False)
    tracking.optima.to_csv(tables / "dynamic_tracking_optima.csv", index=False)
    tracking.scaling.to_csv(tables / "dynamic_tracking_scaling.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.1))
    true_probability = 0.70
    skill_difference = np.linspace(-7.0, 7.0, 600)
    population_loss = (
        np.logaddexp(0.0, skill_difference)
        - true_probability * skill_difference
    )
    optimum = np.log(true_probability / (1.0 - true_probability))
    minimum_loss = -(
        true_probability * np.log(true_probability)
        + (1.0 - true_probability) * np.log(1.0 - true_probability)
    )
    curvature = true_probability * (1.0 - true_probability)
    local_quadratic = minimum_loss + 0.5 * curvature * (skill_difference - optimum) ** 2

    ax = axes[0]
    ax.plot(skill_difference, population_loss, color=COLORS[0], linewidth=2.3)
    ax.plot(
        skill_difference,
        local_quadratic,
        color="0.35",
        linestyle="--",
        linewidth=1.6,
        label="local quadratic",
    )
    ax.axvline(optimum, color="0.55", linestyle=":", linewidth=1)
    ax.set_ylim(bottom=0.0, top=4.2)
    ax.set_xlabel(r"Bradley-Terry skill difference $\theta_i-\theta_j$")
    ax.set_ylabel("expected logistic loss")
    ax.set_title("A. Quadratic minimum, linear tails")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    summary = invariant.summary.sort_values("alpha")
    ax.plot(
        summary["alpha"],
        summary["skill_error_variance"],
        marker="o",
        color=COLORS[1],
        linewidth=2.1,
        label="simulation",
    )
    ax.plot(
        summary["alpha"],
        0.5 * summary["alpha"],
        color="0.35",
        linestyle="--",
        linewidth=1.5,
        label=r"theory: $\mathrm{Var}=\alpha/2$",
    )
    ax.set_xlabel(r"SGD step $\alpha=(\log 10/400)K$")
    ax.set_ylabel("stationary skill-error variance")
    ax.set_title("B. Stationary variance versus step size")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    quantile_probability = (np.arange(1, 200) - 0.5) / 199.0
    normal_quantiles = np.array(
        [NormalDist().inv_cdf(value) for value in quantile_probability]
    )
    for color, k_factor in zip(COLORS, (2.0, 8.0, 32.0)):
        standardized = invariant.samples.loc[
            invariant.samples["k_factor"] == k_factor, "standardized_error"
        ].to_numpy()
        ax.plot(
            normal_quantiles,
            np.quantile(standardized, quantile_probability),
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
    ax.set_title("C. Standardized stationary errors")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        r"Stationary logistic Elo with $(m,\beta)=(2,1)$", y=1.03, fontsize=14
    )
    fig.tight_layout()
    _save_figure(fig, figures / "theory_bridge.png")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    ax = axes[0]
    shown_tau = 200
    for color, (dynamics, group) in zip(
        COLORS, tracking.grid.loc[tracking.grid["tau"] == shown_tau].groupby("dynamics")
    ):
        ax.plot(
            group["alpha"],
            group["tracking_rmse"],
            color=color,
            linewidth=2,
            label=dynamics,
        )
        best = group.loc[group["tracking_rmse"].idxmin()]
        ax.scatter(
            best["alpha"],
            best["tracking_rmse"],
            s=72,
            facecolors="white",
            edgecolors=color,
            linewidths=2,
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"constant step $\alpha$")
    ax.set_ylabel("tracking RMSE")
    ax.set_title(rf"A. Tracking RMSE at $\tau={shown_tau}$")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)

    ax = axes[1]
    for color, (dynamics, group) in zip(COLORS, tracking.optima.groupby("dynamics")):
        scaling = tracking.scaling.loc[
            tracking.scaling["dynamics"] == dynamics
        ].iloc[0]
        ax.plot(
            group["tau"],
            group["refined_alpha"],
            marker="o",
            color=color,
            linewidth=2,
            label=(
                f"{dynamics}: {scaling['fitted_slope']:.2f} "
                f"[{scaling['mc_ci_2_5']:.2f}, {scaling['mc_ci_97_5']:.2f}]"
            ),
        )
        if dynamics in {"smooth", "ou"}:
            exponent = -2.0 / 3.0 if dynamics == "smooth" else -0.5
            anchor = float(group.iloc[len(group) // 2]["refined_alpha"])
            anchor_tau = float(group.iloc[len(group) // 2]["tau"])
            reference = anchor * (group["tau"] / anchor_tau) ** exponent
            ax.plot(
                group["tau"],
                reference,
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
            )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sorted(tracking.optima["tau"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(r"latent-skill time scale $\tau$")
    ax.set_ylabel(r"empirical optimal $\alpha$")
    ax.set_title(r"B. Optimal $\alpha$ versus $\tau$" "\nSlope [95% MC interval]")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.suptitle("Dynamic-skill tracking simulation", y=1.02, fontsize=14)
    fig.tight_layout()
    _save_figure(fig, figures / "simulation_tradeoff.png")
    return {
        "invariant_fluctuation_summary": invariant.summary,
        "dynamic_tracking_grid": tracking.grid,
        "dynamic_tracking_optima": tracking.optima,
        "dynamic_tracking_scaling": tracking.scaling,
    }


def _tennis_overview(
    matches: pd.DataFrame,
    result: TennisEvaluationResult,
) -> pd.DataFrame:
    appearances = pd.concat([matches["player_a"], matches["player_b"]]).value_counts()
    market_coverage = (
        matches["pinnacle_odds_a"].notna()
        | matches["average_odds_a"].notna()
    )
    return pd.DataFrame(
        {
            "statistic": [
                "completed matches",
                "unique players",
                "first match date",
                "last match date",
                "median matches per player",
                "market odds coverage",
                "validation period begins",
                "held-out test begins",
                "selected model",
            ],
            "value": [
                len(matches),
                appearances.size,
                matches["date"].min().date().isoformat(),
                matches["date"].max().date().isoformat(),
                f"{appearances.median():.1f}",
                f"{market_coverage.mean():.2%}",
                result.validation_start.date().isoformat(),
                result.test_start.date().isoformat(),
                result.selected_model,
            ],
        }
    )


def _tennis_calibration(result: TennisEvaluationResult) -> pd.DataFrame:
    test = result.predictions.loc[result.predictions["split"] == "test"]
    columns = {
        "Overall Elo": "overall_probability",
        "Surface Elo": "surface_probability",
        "Multi-timescale Elo": "mixture_probability",
        "De-vigged market": "market_probability_a",
        "Market + Elo overlay": "overlay_probability",
    }
    frames = []
    for model, column in columns.items():
        subset = test.loc[test[column].notna()]
        calibration = calibration_table(
            subset["outcome_a"].to_numpy(), subset[column].to_numpy()
        )
        calibration.insert(0, "model", model)
        frames.append(calibration)
    return pd.concat(frames, ignore_index=True)


def build_tennis_outputs(
    root: str | Path = PROJECT_ROOT,
    *,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    refresh: bool = False,
) -> tuple[pd.DataFrame, TennisEvaluationResult]:
    """Run the ATP walk-forward evaluation and write aggregate outputs."""

    root = Path(root)
    figures, tables = _output_dirs(root)
    matches = load_tennis_matches(
        start_year,
        end_year,
        cache_dir=root / "data" / "cache",
        refresh=refresh,
    )
    result = evaluate_tennis(matches)
    calibration = _tennis_calibration(result)
    overview = _tennis_overview(matches, result)

    overview.to_csv(tables / "atp_data_overview.csv", index=False)
    result.metrics.to_csv(tables / "tennis_test_metrics.csv", index=False)
    result.candidate_metrics.to_csv(tables / "tennis_validation_grid.csv", index=False)
    result.overlay_validation.to_csv(
        tables / "tennis_overlay_validation.csv", index=False
    )
    result.uncertainty.to_csv(tables / "tennis_market_uncertainty.csv", index=False)
    result.model_comparison_uncertainty.to_csv(
        tables / "tennis_model_comparison_uncertainty.csv", index=False
    )
    result.yearly_stability.to_csv(
        tables / "tennis_yearly_stability.csv", index=False
    )
    result.strategy_validation.to_csv(
        tables / "tennis_strategy_validation.csv", index=False
    )
    result.strategy_test.to_csv(tables / "tennis_strategy_test.csv", index=False)
    calibration.to_csv(tables / "tennis_calibration.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))
    ax = axes[0]
    surface_grid = result.candidate_metrics.loc[
        result.candidate_metrics["model"] == "Surface Elo"
    ].copy()
    global_steps = np.sort(surface_grid["k_factor"].unique())
    surface_steps = np.sort(surface_grid["k_surface"].unique())
    loss_grid = (
        surface_grid.pivot(
            index="k_surface", columns="k_factor", values="log_loss"
        )
        .reindex(index=surface_steps, columns=global_steps)
        .to_numpy()
    )
    image = ax.imshow(loss_grid, origin="lower", aspect="auto", cmap="viridis")
    chosen = surface_grid.loc[surface_grid["log_loss"].idxmin()]
    chosen_x = int(np.flatnonzero(global_steps == chosen["k_factor"])[0])
    chosen_y = int(np.flatnonzero(surface_steps == chosen["k_surface"])[0])
    ax.scatter(
        chosen_x,
        chosen_y,
        s=115,
        facecolors="none",
        edgecolors="white",
        linewidths=2,
    )
    ax.set_xticks(
        np.arange(len(global_steps)), [f"{value:g}" for value in global_steps]
    )
    ax.set_yticks(
        np.arange(len(surface_steps)), [f"{value:g}" for value in surface_steps]
    )
    ax.set_xlabel(r"global step $K_{\mathrm{global}}$")
    ax.set_ylabel(r"surface step $K_{\mathrm{surface}}$")
    ax.set_title("A. Surface Elo validation loss")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.046)
    colorbar.set_label("log loss", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    ax = axes[1]
    test_metrics = result.metrics.loc[
        result.metrics["sample"] == "all test matches"
    ]
    test_metrics = test_metrics.sort_values("log_loss", ascending=True)
    y_position = np.arange(len(test_metrics))
    ax.scatter(
        test_metrics["log_loss"],
        y_position,
        s=60,
        color=[COLORS[index % len(COLORS)] for index in range(len(test_metrics))],
        zorder=3,
    )
    for y_value, loss in zip(y_position, test_metrics["log_loss"]):
        ax.text(float(loss) + 0.0015, y_value, f"{loss:.4f}", va="center", fontsize=7)
    ax.set_yticks(y_position, test_metrics["model"])
    ax.set_xlim(
        float(test_metrics["log_loss"].min()) - 0.008,
        float(test_metrics["log_loss"].max()) + 0.018,
    )
    ax.set_xlabel("held-out log loss")
    comparison = result.model_comparison_uncertainty.iloc[0]
    ax.set_title(
        "B. 2022-2025 locked test\n"
        f"Surface - Overall: {comparison['mean_log_loss_difference']:.4f} "
        f"[{comparison['ci_2_5']:.4f}, {comparison['ci_97_5']:.4f}]"
    )
    ax.grid(axis="x", alpha=0.22)

    ax = axes[2]
    for color, (model, group) in zip(COLORS, calibration.groupby("model")):
        ax.plot(
            group["mean_prediction"],
            group["empirical_score"],
            marker="o",
            color=color,
            linewidth=1.7,
            label=model,
        )
    ax.plot([0, 1], [0, 1], color="0.35", linestyle="--", linewidth=1)
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0.08, 0.92)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("empirical win rate")
    ax.set_title("C. Test-period calibration")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Chronological ATP test results", y=1.02, fontsize=14)
    fig.tight_layout()
    _save_figure(fig, figures / "tennis_walk_forward.png")

    selected_column = {
        "Overall Elo": "overall_probability",
        "Surface Elo": "surface_probability",
        "Multi-timescale Elo": "mixture_probability",
    }[result.selected_model]
    market_test = result.predictions.loc[
        (result.predictions["split"] == "test")
        & result.predictions["market_probability_a"].notna()
    ].copy()
    selected_loss = per_game_log_loss(
        market_test["outcome_a"].to_numpy(), market_test[selected_column].to_numpy()
    )
    market_loss = per_game_log_loss(
        market_test["outcome_a"].to_numpy(),
        market_test["market_probability_a"].to_numpy(),
    )
    overlay_loss = per_game_log_loss(
        market_test["outcome_a"].to_numpy(),
        market_test["overlay_probability"].to_numpy(),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    ax = axes[0]
    ax.plot(
        market_test["date"],
        np.cumsum(selected_loss - market_loss),
        color=COLORS[1],
        linewidth=1.8,
        label=f"{result.selected_model} minus market",
    )
    ax.plot(
        market_test["date"],
        np.cumsum(overlay_loss - market_loss),
        color=COLORS[0],
        linewidth=1.8,
        label="locked overlay minus market",
    )
    ax.axhline(0, color="0.35", linestyle="--", linewidth=1)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("match date")
    ax.set_ylabel("cumulative model loss minus market loss")
    ax.set_title("A. Cumulative excess log loss")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    validation_strategy = result.strategy_validation.loc[
        result.strategy_validation["policy"] == "trade"
    ].sort_values("threshold")
    validation_roi = validation_strategy["unit_roi"].to_numpy(dtype=float)
    validation_lower = validation_strategy["roi_ci_2_5"].to_numpy(dtype=float)
    validation_upper = validation_strategy["roi_ci_97_5"].to_numpy(dtype=float)
    ax.errorbar(
        validation_strategy["threshold"],
        validation_roi,
        yerr=np.vstack(
            [validation_roi - validation_lower, validation_upper - validation_roi]
        ),
        marker="o",
        color=COLORS[2],
        linewidth=2,
        capsize=3,
        label="validation",
    )
    selected_policy = str(result.strategy_test.iloc[0]["policy"])
    selected_threshold = result.strategy_test.iloc[0]["threshold"]
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1, label="zero return")
    if selected_policy == "trade":
        test_strategy = result.strategy_test.iloc[0]
        test_roi = float(test_strategy["unit_roi"])
        selected_validation = validation_strategy.loc[
            validation_strategy["threshold"] == float(selected_threshold)
        ].iloc[0]
        ax.axvline(float(selected_threshold), color="0.35", linestyle=":", linewidth=1)
        ax.scatter(
            [float(selected_threshold)],
            [float(selected_validation["unit_roi"])],
            s=120,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="selected on validation",
            zorder=4,
        )
        ax.errorbar(
            [float(selected_threshold)],
            [test_roi],
            yerr=[
                [test_roi - float(test_strategy["roi_ci_2_5"])],
                [float(test_strategy["roi_ci_97_5"]) - test_roi],
            ],
            marker="D",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor=COLORS[1],
            color=COLORS[1],
            linewidth=1.8,
            capsize=4,
            label="locked test",
        )
    else:
        ax.text(
            0.98,
            0.93,
            "validation decision: abstain",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
    ax.set_xlabel("minimum probability edge")
    ax.set_ylabel("unit ROI")
    ax.set_title("B. Unit ROI by validation threshold")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.suptitle("Market comparison and betting uncertainty", y=1.02, fontsize=14)
    fig.tight_layout()
    _save_figure(fig, figures / "market_decision.png")
    return matches, result
