import tempfile
import unittest
from pathlib import Path

import pandas as pd

from elo_online.data import parse_tennis_workbook, tennis_data_url


class DataTests(unittest.TestCase):
    def test_parser_filters_orients_and_sorts(self) -> None:
        raw = pd.DataFrame(
            {
                "Date": ["2024-01-03", "2024-01-01", "2024-01-02"],
                "Winner": ["Zed", "Alice", "Carl"],
                "Loser": ["Bob", "Zoe", "Dana"],
                "Surface": ["Hard", "Clay", "Hard"],
                "Comment": ["Completed", "Completed", "Retired"],
                "Tournament": ["T1", "T1", "T1"],
                "Location": ["L", "L", "L"],
                "Round": ["1st Round", "1st Round", "1st Round"],
                "WRank": [10, 20, 30],
                "LRank": [40, 50, 60],
                "PSW": [1.5, 2.2, 1.8],
                "PSL": [2.8, 1.7, 2.1],
                "AvgW": [1.55, 2.25, 1.85],
                "AvgL": [2.75, 1.68, 2.05],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "2024.xlsx"
            raw.to_excel(path, index=False)
            frame = parse_tennis_workbook(path, year=2024)
        self.assertEqual(frame["player_a"].tolist(), ["Alice", "Bob"])
        self.assertEqual(frame["outcome_a"].tolist(), [1.0, 0.0])
        self.assertEqual(frame["pinnacle_odds_a"].tolist(), [2.2, 2.8])
        self.assertTrue(frame["date"].is_monotonic_increasing)

    def test_url_is_year_specific(self) -> None:
        self.assertEqual(
            tennis_data_url(2024),
            "http://www.tennis-data.co.uk/2024/2024.xlsx",
        )

    def test_parser_handles_missing_optional_metadata(self) -> None:
        raw = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "Winner": ["Alice", " ", "CARL"],
                "Loser": ["Bob", "Dana", "carl"],
                "Surface": ["hard", "clay", "grass"],
                "Comment": ["Completed", "Completed", "Completed"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "2024.xlsx"
            raw.to_excel(path, index=False)
            frame = parse_tennis_workbook(path, year=2024)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "tournament"], "Unknown")
        self.assertEqual(frame.loc[0, "location"], "Unknown")
        self.assertEqual(frame.loc[0, "round"], "Unknown")
        self.assertEqual(frame.loc[0, "surface"], "Hard")


if __name__ == "__main__":
    unittest.main()
