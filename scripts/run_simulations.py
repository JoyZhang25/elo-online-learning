#!/usr/bin/env python3
"""Run the controlled constant-step Elo experiments."""

from elo_online.reporting import build_simulation_outputs


if __name__ == "__main__":
    outputs = build_simulation_outputs()
    print("Invariant-fluctuation summary")
    print(outputs["invariant_fluctuation_summary"].to_string(index=False))
    print("\nStability-adaptation summary")
    print(outputs["stability_adaptation"].to_string(index=False))
    print("\nChange-point summary")
    print(outputs["change_point_summary"].to_string(index=False))
