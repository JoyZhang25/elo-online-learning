"""Download and clean a fixed public Lichess tournament sample."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


DEFAULT_TOURNAMENT_ID = "NQzyuRkI"
DEFAULT_TOURNAMENT_NAME = "Daily Blitz Arena, 18 September 2025"


def tournament_api_url(tournament_id: str) -> str:
    """Construct the official Lichess NDJSON export URL."""

    query = urlencode(
        {
            "moves": "false",
            "clocks": "false",
            "evals": "false",
            "opening": "false",
        }
    )
    return f"https://lichess.org/api/tournament/{tournament_id}/games?{query}"


def download_tournament(
    tournament_id: str = DEFAULT_TOURNAMENT_ID,
    *,
    cache_dir: str | Path = "data/cache",
    refresh: bool = False,
) -> Path:
    """Download a tournament once and return the local NDJSON cache path."""

    cache_path = Path(cache_dir) / f"lichess_{tournament_id}.ndjson"
    if cache_path.exists() and not refresh:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        tournament_api_url(tournament_id),
        headers={
            "Accept": "application/x-ndjson",
            "User-Agent": "elo-online-learning-research/0.1",
        },
    )
    with urlopen(request, timeout=120) as response:
        payload = response.read()
    if not payload.strip():
        raise RuntimeError("Lichess returned an empty tournament export")
    cache_path.write_bytes(payload)
    return cache_path


def _player_id(player: dict) -> str | None:
    user = player.get("user") or {}
    identifier = user.get("id") or user.get("name")
    return str(identifier).lower() if identifier else None


def parse_tournament_ndjson(path: str | Path) -> pd.DataFrame:
    """Parse, validate, deduplicate, and time-sort a Lichess game export."""

    rows: list[dict] = []
    excluded_statuses = {"aborted", "created", "started", "noStart"}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                game = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid NDJSON on line {line_number}") from exc
            if game.get("variant") != "standard" or not game.get("rated", False):
                continue
            if game.get("status") in excluded_statuses:
                continue
            players = game.get("players") or {}
            white = players.get("white") or {}
            black = players.get("black") or {}
            white_id = _player_id(white)
            black_id = _player_id(black)
            white_rating = white.get("rating")
            black_rating = black.get("rating")
            if (
                white_id is None
                or black_id is None
                or white_rating is None
                or black_rating is None
            ):
                continue
            winner = game.get("winner")
            if winner == "white":
                outcome = 1.0
            elif winner == "black":
                outcome = 0.0
            else:
                outcome = 0.5
            tournament = game.get("arenaTour") or {}
            rows.append(
                {
                    "game_id": str(game["id"]),
                    "created_at": pd.to_datetime(
                        game["createdAt"], unit="ms", utc=True
                    ),
                    "white_id": white_id,
                    "black_id": black_id,
                    "white_platform_rating": float(white_rating),
                    "black_platform_rating": float(black_rating),
                    "white_score": outcome,
                    "status": str(game.get("status", "unknown")),
                    "speed": str(game.get("speed", "unknown")),
                    "tournament_id": str(tournament.get("id", "")),
                    "tournament_name": str(tournament.get("name", "")),
                }
            )
    if not rows:
        raise ValueError("no eligible rated standard games found")
    frame = pd.DataFrame(rows)
    frame = (
        frame.drop_duplicates(subset="game_id", keep="first")
        .sort_values(["created_at", "game_id"], kind="stable")
        .reset_index(drop=True)
    )
    if not frame["created_at"].is_monotonic_increasing:
        raise AssertionError("games must be in chronological order")
    if not np.isfinite(
        frame[["white_platform_rating", "black_platform_rating"]].to_numpy()
    ).all():
        raise ValueError("ratings contain non-finite values")
    return frame


def load_tournament_games(
    tournament_id: str = DEFAULT_TOURNAMENT_ID,
    *,
    cache_dir: str | Path = "data/cache",
    refresh: bool = False,
) -> pd.DataFrame:
    """Download if needed, then return the cleaned chronological game table."""

    path = download_tournament(
        tournament_id, cache_dir=cache_dir, refresh=refresh
    )
    return parse_tournament_ndjson(path)
