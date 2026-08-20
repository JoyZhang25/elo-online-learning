import unittest

import numpy as np

from elo_online.model import (
    EloModel,
    alpha_to_k,
    expected_score,
    k_to_alpha,
    rating_to_skill,
    skill_to_rating,
)


class EloModelTests(unittest.TestCase):
    def test_expected_score_symmetry(self) -> None:
        p_ab = expected_score(1650.0, 1500.0)
        p_ba = expected_score(1500.0, 1650.0)
        self.assertAlmostEqual(p_ab + p_ba, 1.0)
        self.assertGreater(p_ab, 0.5)

    def test_update_is_zero_sum(self) -> None:
        model = EloModel(k_factor=32.0)
        prediction = model.predict("a", "b")
        before = model.rating("a") + model.rating("b")
        model.update("a", "b", 1.0, prediction_a=prediction)
        after = model.rating("a") + model.rating("b")
        self.assertAlmostEqual(before, after)
        self.assertGreater(model.rating("a"), model.rating("b"))

    def test_draw_between_equal_players_has_no_update(self) -> None:
        model = EloModel(k_factor=20.0)
        prediction = model.predict("a", "b")
        change = model.update("a", "b", 0.5, prediction_a=prediction)
        self.assertAlmostEqual(change, 0.0)

    def test_invalid_step_rejected(self) -> None:
        for value in (-1.0, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                EloModel(k_factor=value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                k_to_alpha(value)

    def test_invalid_update_inputs_are_rejected(self) -> None:
        model = EloModel(k_factor=20.0)
        with self.assertRaises(ValueError):
            model.predict("alice", "Alice")
        with self.assertRaises(ValueError):
            model.update("alice", "bob", 1.0, prediction_a=np.nan)
        with self.assertRaises(ValueError):
            model.update("alice", "bob", 1.0, k_factor=np.inf)

    def test_k_alpha_and_rating_skill_mappings_are_inverses(self) -> None:
        k_factor = 32.0
        self.assertAlmostEqual(alpha_to_k(k_to_alpha(k_factor)), k_factor)
        rating = 173.71779276130073
        self.assertAlmostEqual(rating_to_skill(rating), 1.0)
        self.assertAlmostEqual(skill_to_rating(rating_to_skill(rating)), rating)


if __name__ == "__main__":
    unittest.main()
