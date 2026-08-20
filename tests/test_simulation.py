import unittest

import pandas as pd

from elo_online.simulation import (
    change_point_experiment,
    invariant_fluctuation_experiment,
    stability_adaptation_experiment,
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
        self.assertTrue((first.summary["variance"] > 0).all())

    def test_tradeoff_is_deterministic(self) -> None:
        kwargs = {
            "k_grid": (4.0, 16.0),
            "drift_grid": (0.0, 0.2),
            "n_players": 8,
            "n_games": 300,
            "burn_fraction": 0.5,
            "seed": 17,
        }
        first = stability_adaptation_experiment(**kwargs)
        second = stability_adaptation_experiment(**kwargs)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 4)

    def test_change_point_outputs_all_methods(self) -> None:
        result = change_point_experiment(
            k_values=(8.0, 32.0), n_games=200, change_at=100, seed=3
        )
        self.assertEqual(len(result.trajectories), 200)
        self.assertEqual(
            set(result.summary["method"]),
            {"constant K=8", "constant K=32", "decaying K"},
        )


if __name__ == "__main__":
    unittest.main()
