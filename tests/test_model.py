import unittest

from elo_online.model import EloModel, expected_score


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
        with self.assertRaises(ValueError):
            EloModel(k_factor=-1.0)


if __name__ == "__main__":
    unittest.main()
