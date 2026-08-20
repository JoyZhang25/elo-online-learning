"""Download and clean annual ATP results and pre-match odds."""

from __future__ import annotations

import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


DEFAULT_START_YEAR = 2010
DEFAULT_END_YEAR = 2025
TENNIS_DATA_BASE_URL = "http://www.tennis-data.co.uk"


def tennis_data_url(year: int) -> str:
    """Return the Tennis-Data URL for one annual ATP workbook."""

    if year < 2000 or year > 2100:
        raise ValueError("year must be between 2000 and 2100")
    return f"{TENNIS_DATA_BASE_URL}/{year}/{year}.xlsx"


def download_tennis_year(
    year: int,
    *,
    cache_dir: str | Path = "data/cache",
    refresh: bool = False,
) -> Path:
    """Download one annual workbook, retaining it only in the ignored cache."""

    cache_root = Path(cache_dir)
    candidates = [cache_root / f"atp_{year}.xlsx", cache_root / f"atp_{year}.xls"]
    if not refresh:
        for candidate in candidates:
            if candidate.exists():
                return candidate
    cache_root.mkdir(parents=True, exist_ok=True)
    request = Request(
        tennis_data_url(year),
        headers={"User-Agent": "elo-online-learning/0.2"},
    )
    payload: bytes | None = None
    last_error: HTTPError | URLError | TimeoutError | None = None
    for attempt in range(4):
        try:
            with urlopen(request, timeout=120) as response:
                payload = response.read()
            break
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2**attempt)
    if payload is None:
        raise RuntimeError(
            f"failed to download Tennis-Data file for {year}"
        ) from last_error
    if payload.startswith(b"PK"):
        cache_path = candidates[0]
    elif payload.startswith(bytes.fromhex("D0CF11E0")):
        cache_path = candidates[1]
    else:
        raise RuntimeError(f"Tennis-Data did not return an Excel file for {year}")
    cache_path.write_bytes(payload)
    return cache_path


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _string(
    frame: pd.DataFrame,
    column: str,
    *,
    default: str = "Unknown",
) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype="string")
    values = frame[column].astype("string").str.strip()
    return values.mask(values.eq("")).fillna(default)


def _oriented_pair(
    winner_value: pd.Series,
    loser_value: pd.Series,
    player_a_won: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    return (
        winner_value.where(player_a_won, loser_value),
        loser_value.where(player_a_won, winner_value),
    )


def parse_tennis_workbook(path: str | Path, *, year: int | None = None) -> pd.DataFrame:
    """Parse a Tennis-Data ATP workbook into a chronological match table.

    Player orientation is alphabetical rather than winner-first, so the target
    is not encoded in the row layout. Retirements and walkovers are excluded.
    """

    raw = pd.read_excel(path, sheet_name=0)
    required = {"Date", "Winner", "Loser", "Surface", "Comment"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"missing Tennis-Data columns: {sorted(missing)}")

    frame = raw.copy()
    frame["source_row"] = np.arange(len(frame), dtype=int)
    frame["date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    frame["winner"] = frame["Winner"].astype("string").str.strip()
    frame["loser"] = frame["Loser"].astype("string").str.strip()
    frame["comment"] = frame["Comment"].astype("string").str.strip()
    winner_key = frame["winner"].str.casefold()
    loser_key = frame["loser"].str.casefold()
    frame = frame.loc[
        frame["date"].notna()
        & frame["winner"].notna()
        & frame["loser"].notna()
        & frame["winner"].ne("")
        & frame["loser"].ne("")
        & winner_key.ne(loser_key)
        & frame["comment"].str.casefold().eq("completed")
    ].copy()
    if frame.empty:
        raise ValueError("workbook contains no completed ATP matches")

    player_a_won = frame["winner"].str.casefold() < frame["loser"].str.casefold()
    frame["player_a"] = frame["winner"].where(player_a_won, frame["loser"])
    frame["player_b"] = frame["loser"].where(player_a_won, frame["winner"])
    frame["outcome_a"] = player_a_won.astype(float)

    source_year = (
        int(year) if year is not None else int(frame["date"].dt.year.mode()[0])
    )
    frame["source_year"] = source_year
    frame["match_id"] = [f"{source_year}:{row}" for row in frame["source_row"]]
    frame["tournament"] = _string(frame, "Tournament")
    frame["location"] = _string(frame, "Location")
    frame["round"] = _string(frame, "Round")
    frame["surface"] = _string(frame, "Surface").str.title()

    for prefix, winner_column, loser_column in (
        ("pinnacle", "PSW", "PSL"),
        ("average", "AvgW", "AvgL"),
    ):
        winner_odds = _numeric(frame, winner_column)
        loser_odds = _numeric(frame, loser_column)
        odds_a, odds_b = _oriented_pair(winner_odds, loser_odds, player_a_won)
        valid = odds_a.gt(1.0) & odds_b.gt(1.0)
        frame[f"{prefix}_odds_a"] = odds_a.where(valid)
        frame[f"{prefix}_odds_b"] = odds_b.where(valid)

    rank_a, rank_b = _oriented_pair(
        _numeric(frame, "WRank"), _numeric(frame, "LRank"), player_a_won
    )
    frame["rank_a"] = rank_a
    frame["rank_b"] = rank_b

    columns = [
        "match_id",
        "date",
        "source_year",
        "source_row",
        "tournament",
        "location",
        "surface",
        "round",
        "player_a",
        "player_b",
        "outcome_a",
        "rank_a",
        "rank_b",
        "pinnacle_odds_a",
        "pinnacle_odds_b",
        "average_odds_a",
        "average_odds_b",
    ]
    return frame[columns].sort_values(
        ["date", "source_year", "source_row"], kind="stable"
    ).reset_index(drop=True)


def load_tennis_matches(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    *,
    cache_dir: str | Path = "data/cache",
    refresh: bool = False,
) -> pd.DataFrame:
    """Load a contiguous range of annual ATP workbooks."""

    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year")
    frames = []
    for year in range(start_year, end_year + 1):
        path = download_tennis_year(year, cache_dir=cache_dir, refresh=refresh)
        frames.append(parse_tennis_workbook(path, year=year))
    matches = pd.concat(frames, ignore_index=True).sort_values(
        ["date", "source_year", "source_row"], kind="stable"
    )
    matches = matches.drop_duplicates(subset="match_id", keep="first").reset_index(
        drop=True
    )
    if not matches["date"].is_monotonic_increasing:
        raise AssertionError("matches must be chronological")
    return matches
