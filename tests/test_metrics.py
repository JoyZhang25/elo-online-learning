import unittest

import numpy as np

from elo_online.metrics import calibration_table, forecast_metrics


class MetricTests(unittest.TestCase):
    def test_better_forecast_has_lower_scores(self) -> None:
        outcome = np.array([1.0, 0.0, 1.0, 0.0])
        good = forecast_metrics(outcome, np.array([0.9, 0.1, 0.8, 0.2]))
        poor = forecast_metrics(outcome, np.full(4, 0.5))
        self.assertLess(good["log_loss"], poor["log_loss"])
        self.assertLess(good["brier_score"], poor["brier_score"])

    def test_calibration_counts_all_rows(self) -> None:
        table = calibration_table(
            np.array([0.0, 0.5, 1.0]), np.array([0.1, 0.5, 0.9])
        )
        self.assertEqual(int(table["n"].sum()), 3)


if __name__ == "__main__":
    unittest.main()
