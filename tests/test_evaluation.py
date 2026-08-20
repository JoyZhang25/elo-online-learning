import unittest

import pandas as pd

from elo_online.evaluation import evaluate_real_data, walk_forward_predictions


def toy_games(n_games: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [f"g{i:02d}" for i in range(n_games)],
            "created_at": pd.date_range(
                "2025-01-01", periods=n_games, freq="h", tz="UTC"
            ),
            "white_id": ["alice"] * n_games,
            "black_id": ["bob"] * n_games,
            "white_platform_rating": [1500.0] * n_games,
            "black_platform_rating": [1500.0] * n_games,
            "white_score": [1.0] * n_games,
        }
    )


class EvaluationTests(unittest.TestCase):
    def test_prediction_precedes_update(self) -> None:
        predictions = walk_forward_predictions(toy_games(3), k_factor=32.0)
        self.assertAlmostEqual(predictions.loc[0, "elo_probability"], 0.5)
        self.assertGreater(predictions.loc[1, "elo_probability"], 0.5)
        self.assertAlmostEqual(predictions.loc[0, "white_rating_before"], 1500.0)

    def test_chronological_split_and_selection(self) -> None:
        result = evaluate_real_data(
            toy_games(),
            k_grid=(0.0, 16.0),
            white_advantage_grid=(0.0,),
            validation_start_fraction=0.1,
            test_start_fraction=0.4,
        )
        self.assertEqual(result.validation_start, 2)
        self.assertEqual(result.test_start, 8)
        self.assertEqual((result.predictions["split"] == "test").sum(), 12)
        self.assertEqual(result.selected_k, 16.0)


if __name__ == "__main__":
    unittest.main()
