"""Core Elo model written as a constant-step stochastic-gradient update."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def expected_score(
    rating_a: float | np.ndarray,
    rating_b: float | np.ndarray,
    *,
    scale: float = 400.0,
    advantage_a: float = 0.0,
) -> float | np.ndarray:
    """Return the Bradley–Terry/Elo expected score for player A.

    ``scale=400`` gives the familiar base-10 Elo parameterization.  The
    implementation is algebraically equivalent to a logistic link.  Without
    draws, the expected score is also the win probability.  The exponent is
    clipped only to avoid floating-point overflow for extreme inputs.
    """

    if scale <= 0:
        raise ValueError("scale must be positive")
    exponent = np.clip(
        (np.asarray(rating_b) - np.asarray(rating_a) - advantage_a) / scale,
        -20,
        20,
    )
    probability = 1.0 / (1.0 + np.power(10.0, exponent))
    if np.ndim(probability) == 0:
        return float(probability)
    return probability


@dataclass
class EloModel:
    """A stateful online Elo model.

    Predictions are made from the current state.  Calling :meth:`update`
    applies ``K * (outcome - prediction)`` to player A and the opposite change
    to player B, so the sum of the two ratings is conserved.
    """

    k_factor: float
    initial_rating: float = 1500.0
    scale: float = 400.0
    advantage_a: float = 0.0
    ratings: dict[str, float] = field(default_factory=dict)
    games_played: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.k_factor < 0:
            raise ValueError("k_factor must be nonnegative")
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def rating(self, player: str, seed_rating: float | None = None) -> float:
        """Return a player's current rating, initializing it if necessary."""

        if player not in self.ratings:
            value = self.initial_rating if seed_rating is None else float(seed_rating)
            if not np.isfinite(value):
                value = self.initial_rating
            self.ratings[player] = value
            self.games_played[player] = 0
        return self.ratings[player]

    def predict(
        self,
        player_a: str,
        player_b: str,
        *,
        seed_a: float | None = None,
        seed_b: float | None = None,
    ) -> float:
        """Predict player A's score before observing the current outcome."""

        rating_a = self.rating(player_a, seed_a)
        rating_b = self.rating(player_b, seed_b)
        return float(
            expected_score(
                rating_a,
                rating_b,
                scale=self.scale,
                advantage_a=self.advantage_a,
            )
        )

    def update(
        self,
        player_a: str,
        player_b: str,
        outcome_a: float,
        *,
        prediction_a: float | None = None,
        seed_a: float | None = None,
        seed_b: float | None = None,
        k_factor: float | None = None,
    ) -> float:
        """Update both ratings and return the signed change for player A."""

        if not 0.0 <= outcome_a <= 1.0:
            raise ValueError("outcome_a must lie in [0, 1]")
        rating_a = self.rating(player_a, seed_a)
        rating_b = self.rating(player_b, seed_b)
        if prediction_a is None:
            prediction_a = float(
                expected_score(
                    rating_a,
                    rating_b,
                    scale=self.scale,
                    advantage_a=self.advantage_a,
                )
            )
        step = self.k_factor if k_factor is None else float(k_factor)
        if step < 0:
            raise ValueError("k_factor must be nonnegative")
        change = step * (float(outcome_a) - float(prediction_a))
        self.ratings[player_a] = rating_a + change
        self.ratings[player_b] = rating_b - change
        self.games_played[player_a] += 1
        self.games_played[player_b] += 1
        return change
