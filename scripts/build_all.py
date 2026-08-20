#!/usr/bin/env python3
"""Rebuild every committed figure and table."""

from elo_online.reporting import build_real_data_outputs, build_simulation_outputs


if __name__ == "__main__":
    build_simulation_outputs()
    games, result = build_real_data_outputs()
    print(
        f"Built all outputs from {len(games):,} real games; "
        f"selected K={result.selected_k:g}."
    )
