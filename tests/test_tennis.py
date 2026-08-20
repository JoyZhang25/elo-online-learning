import unittest

import numpy as np

from elo_online.tennis import MixtureElo, SurfaceElo


class TennisModelTests(unittest.TestCase):
    def test_surface_model_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            SurfaceElo(k_global=16.0, k_surface=-1.0)

        model = SurfaceElo(k_global=16.0, k_surface=4.0)
        with self.assertRaises(ValueError):
            model.update("alice", "bob", "Hard", 1.0, np.nan)

    def test_mixture_requires_one_valid_probability_per_expert(self) -> None:
        model = MixtureElo(k_grid=(8.0, 16.0), learning_rate=0.1)
        with self.assertRaises(ValueError):
            model.update("alice", "bob", 1.0, np.array([0.5]))
        with self.assertRaises(ValueError):
            model.update("alice", "bob", 1.0, np.array([0.5, np.nan]))


if __name__ == "__main__":
    unittest.main()
