import unittest

import pandas as pd

from elo_online.simulation import (
    dynamic_tracking_experiment,
    invariant_fluctuation_experiment,
)


class SimulationTests(unittest.TestCase):
    def test_invariant_experiment_is_deterministic(self) -> None:
        kwargs = {
            "k_grid": (2.0, 8.0),
            "n_games": 1_000,
            "burn_in": 200,
            "sample_every": 20,
            "seed": 11,
        }
        first = invariant_fluctuation_experiment(**kwargs)
        second = invariant_fluctuation_experiment(**kwargs)
        pd.testing.assert_frame_equal(first.summary, second.summary)
        pd.testing.assert_frame_equal(first.samples, second.samples)
        self.assertEqual(len(first.summary), 2)
        self.assertTrue((first.summary["skill_error_variance"] > 0).all())

    def test_dynamic_tracking_is_deterministic(self) -> None:
        kwargs = {
            "dynamics_grid": ("smooth", "ou"),
            "tau_grid": (10, 20, 40),
            "alpha_grid": (0.02, 0.08, 0.24),
            "minimum_games": 500,
            "games_per_tau": 5,
            "n_replications": 3,
            "n_mc_resamples": 100,
            "seed": 17,
        }
        first = dynamic_tracking_experiment(**kwargs)
        second = dynamic_tracking_experiment(**kwargs)
        pd.testing.assert_frame_equal(first.grid, second.grid)
        pd.testing.assert_frame_equal(first.optima, second.optima)
        pd.testing.assert_frame_equal(first.scaling, second.scaling)
        self.assertEqual(len(first.optima), 6)
        self.assertTrue((first.grid["n_replications"] == 3).all())
        self.assertTrue(
            {"mc_ci_2_5", "mc_ci_97_5", "interval_type"}.issubset(
                first.scaling.columns
            )
        )


if __name__ == "__main__":
    unittest.main()
