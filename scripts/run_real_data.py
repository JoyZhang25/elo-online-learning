#!/usr/bin/env python3
"""Run the chronological ATP Elo and market evaluation."""

from elo_online.reporting import build_tennis_outputs


if __name__ == "__main__":
    matches, result = build_tennis_outputs()
    print(f"Completed ATP matches: {len(matches):,}")
    print(f"Selected model: {result.selected_model}")
    print(f"Selected parameters: {result.selected_params}")
    print(result.metrics.to_string(index=False))
    print("\nLocked test strategy")
    print(result.strategy_test.to_string(index=False))
