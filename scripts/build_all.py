#!/usr/bin/env python3
"""Rebuild every committed figure and table."""

from elo_online.reporting import build_simulation_outputs, build_tennis_outputs


if __name__ == "__main__":
    build_simulation_outputs()
    matches, result = build_tennis_outputs()
    print(
        f"Built all outputs from {len(matches):,} ATP matches; "
        f"selected {result.selected_model}."
    )
