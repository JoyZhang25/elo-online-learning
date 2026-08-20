import json
import tempfile
import unittest
from pathlib import Path

from elo_online.data import parse_tournament_ndjson


def _game(game_id: str, created_at: int, *, variant: str = "standard") -> dict:
    return {
        "id": game_id,
        "rated": True,
        "variant": variant,
        "speed": "blitz",
        "createdAt": created_at,
        "status": "mate",
        "winner": "white",
        "players": {
            "white": {"user": {"id": "alice"}, "rating": 1600},
            "black": {"user": {"id": "bob"}, "rating": 1500},
        },
        "arenaTour": {"id": "sample", "name": "Sample"},
    }


class DataTests(unittest.TestCase):
    def test_parser_filters_and_sorts(self) -> None:
        records = [
            _game("late", 2_000),
            _game("skip", 1_500, variant="chess960"),
            _game("early", 1_000),
            _game("early", 1_000),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.ndjson"
            path.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )
            frame = parse_tournament_ndjson(path)
        self.assertEqual(frame["game_id"].tolist(), ["early", "late"])
        self.assertTrue(frame["created_at"].is_monotonic_increasing)
        self.assertEqual(frame["white_score"].tolist(), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
