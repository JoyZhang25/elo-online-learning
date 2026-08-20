#!/usr/bin/env python3
"""Run the controlled constant-step Elo experiments."""

from elo_online.reporting import build_simulation_outputs


if __name__ == "__main__":
    outputs = build_simulation_outputs()
    print("Invariant-fluctuation summary")
    print(outputs["invariant_fluctuation_summary"].to_string(index=False))
    print("\nDynamic tracking optima")
    print(outputs["dynamic_tracking_optima"].to_string(index=False))
    print("\nFitted scaling exponents")
    print(outputs["dynamic_tracking_scaling"].to_string(index=False))
