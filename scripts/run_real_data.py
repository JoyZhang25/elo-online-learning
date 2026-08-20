#!/usr/bin/env python3
"""Run the fixed-sample, walk-forward Lichess evaluation."""

from elo_online.reporting import build_real_data_outputs


if __name__ == "__main__":
    games, result = build_real_data_outputs()
    print(f"Cleaned games: {len(games):,}")
    print(
        f"Selected K={result.selected_k:g}, "
        f"white advantage={result.selected_white_advantage:g} Elo points"
    )
    print(result.metrics.to_string(index=False))
