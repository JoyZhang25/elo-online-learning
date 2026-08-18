# Data provenance

The real-data experiment uses the completed Lichess Daily Blitz Arena with
tournament ID `NQzyuRkI`, played on 18 September 2025.

- Tournament page: <https://lichess.org/tournament/NQzyuRkI>
- Official export endpoint: `https://lichess.org/api/tournament/NQzyuRkI/games`
- Lichess open-database terms: <https://database.lichess.org/>

The raw NDJSON response is cached under `data/cache/` and excluded from version
control. Running `python scripts/run_real_data.py` downloads it only when the
cache is absent. Committed prediction outputs hash player identifiers and retain
only the fields required to audit the chronological evaluation.
