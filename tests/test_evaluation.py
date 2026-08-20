import unittest

import numpy as np
import pandas as pd

from elo_online.evaluation import (
    evaluate_tennis,
    paired_month_block_bootstrap,
    walk_forward_overall,
    walk_forward_surface,
)
from elo_online.metrics import per_game_log_loss


def toy_matches() -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2017-01-01"] * 4
        + ["2018-01-01"] * 4
        + ["2019-01-01"] * 4
        + ["2022-01-01"] * 4
        + ["2023-01-01"] * 4
    )
    n_matches = len(dates)
    outcomes = np.tile([1.0, 1.0, 0.0, 1.0], n_matches // 4)
    return pd.DataFrame(
        {
            "match_id": [f"m{i:02d}" for i in range(n_matches)],
            "date": dates,
            "player_a": ["alice"] * n_matches,
            "player_b": ["bob"] * n_matches,
            "surface": np.tile(["Hard", "Clay", "Hard", "Clay"], n_matches // 4),
            "outcome_a": outcomes,
            "pinnacle_odds_a": np.full(n_matches, 1.8),
            "pinnacle_odds_b": np.full(n_matches, 2.2),
        }
    )


class EvaluationTests(unittest.TestCase):
    def test_same_day_is_predicted_before_any_update(self) -> None:
        matches = toy_matches().iloc[:8].copy()
        predictions = walk_forward_overall(matches, k_factor=32.0)
        self.assertTrue(np.allclose(predictions[:4], 0.5))
        self.assertGreater(predictions[4], 0.5)

    def test_same_day_predictions_are_invariant_to_row_order(self) -> None:
        matches = toy_matches().iloc[:12].copy()
        shuffled = matches.sample(frac=1.0, random_state=19)
        expected = walk_forward_overall(matches, k_factor=32.0)
        actual = walk_forward_overall(shuffled, k_factor=32.0)
        np.testing.assert_allclose(actual, expected)

    def test_zero_surface_step_agrees_with_overall_elo(self) -> None:
        matches = toy_matches()
        overall = walk_forward_overall(matches, k_factor=16.0)
        surface = walk_forward_surface(
            matches, k_global=16.0, k_surface=0.0
        )
        np.testing.assert_allclose(surface, overall, rtol=0.0, atol=1e-15)

    def test_paired_month_bootstrap_is_deterministic(self) -> None:
        frame = toy_matches()[["date", "outcome_a"]].copy()
        frame["left"] = np.linspace(0.35, 0.65, len(frame))
        frame["right"] = 0.5
        first = paired_month_block_bootstrap(
            frame, "left", "right", n_bootstrap=200, seed=23
        )
        second = paired_month_block_bootstrap(
            frame, "left", "right", n_bootstrap=200, seed=23
        )
        self.assertEqual(first, second)
        expected_difference = np.mean(
            per_game_log_loss(frame["outcome_a"], frame["left"])
            - per_game_log_loss(frame["outcome_a"], frame["right"])
        )
        self.assertAlmostEqual(
            first["mean_log_loss_difference"], expected_difference
        )
        self.assertEqual(first["n_month_blocks"], 5)

    def test_chronological_split_and_selection(self) -> None:
        result = evaluate_tennis(
            toy_matches(),
            k_grid=(8.0, 16.0),
            surface_k_grid=(0.0, 4.0),
            mixture_learning_rates=(0.1,),
            strategy_thresholds=(0.0, 0.04),
            n_bootstrap=50,
        )
        self.assertEqual((result.predictions["split"] == "warmup").sum(), 4)
        self.assertEqual((result.predictions["split"] == "validation").sum(), 8)
        self.assertEqual((result.predictions["split"] == "test").sum(), 8)
        self.assertIn(
            result.selected_model,
            {"Overall Elo", "Surface Elo", "Multi-timescale Elo"},
        )
        self.assertEqual(len(result.strategy_test), 1)
        self.assertEqual(set(result.yearly_stability["year"]), {2022, 2023})
        self.assertEqual(
            list(result.model_comparison_uncertainty["difference"]),
            ["Surface minus Overall log loss"],
        )
        self.assertTrue(
            {
                "year",
                "n_matches",
                "overall_log_loss",
                "surface_log_loss",
                "surface_minus_overall",
            }.issubset(result.yearly_stability.columns)
        )

    def test_test_outcomes_do_not_change_validation_selection(self) -> None:
        kwargs = {
            "k_grid": (8.0, 16.0),
            "surface_k_grid": (0.0, 4.0),
            "mixture_learning_rates": (0.1,),
            "strategy_thresholds": (0.0, 0.04),
            "n_bootstrap": 30,
        }
        baseline = evaluate_tennis(toy_matches(), **kwargs)
        altered_matches = toy_matches()
        altered_matches.loc[
            altered_matches["date"] >= pd.Timestamp("2022-01-01"), "outcome_a"
        ] = 1.0 - altered_matches.loc[
            altered_matches["date"] >= pd.Timestamp("2022-01-01"), "outcome_a"
        ]
        altered = evaluate_tennis(altered_matches, **kwargs)
        self.assertEqual(altered.selected_model, baseline.selected_model)
        self.assertEqual(altered.selected_params, baseline.selected_params)
        pd.testing.assert_frame_equal(
            altered.candidate_metrics, baseline.candidate_metrics
        )
        pd.testing.assert_frame_equal(
            altered.overlay_validation, baseline.overlay_validation
        )

    def test_invalid_match_rows_are_rejected(self) -> None:
        duplicate_id = toy_matches()
        duplicate_id.loc[1, "match_id"] = duplicate_id.loc[0, "match_id"]
        with self.assertRaises(ValueError):
            walk_forward_overall(duplicate_id, k_factor=16.0)

        invalid_outcome = toy_matches()
        invalid_outcome.loc[0, "outcome_a"] = 0.5
        with self.assertRaises(ValueError):
            walk_forward_overall(invalid_outcome, k_factor=16.0)

    def test_invalid_split_order_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_tennis(
                toy_matches(),
                validation_start="2022-01-01",
                test_start="2018-01-01",
                k_grid=(8.0,),
                surface_k_grid=(0.0,),
                mixture_learning_rates=(0.1,),
                n_bootstrap=10,
            )


if __name__ == "__main__":
    unittest.main()
