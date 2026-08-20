"""Core Elo model written as a constant-step stochastic-gradient update."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


DEFAULT_ELO_SCALE = 400.0


def k_to_alpha(k_factor: float, *, scale: float = DEFAULT_ELO_SCALE) -> float:
    """Convert the conventional Elo ``K`` to the logistic-SGD step ``alpha``."""

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive")
    k_factor = float(k_factor)
    if not np.isfinite(k_factor) or k_factor < 0:
        raise ValueError("k_factor must be finite and nonnegative")
    return k_factor * np.log(10.0) / scale


def alpha_to_k(alpha: float, *, scale: float = DEFAULT_ELO_SCALE) -> float:
    """Convert a normalized logistic-SGD step ``alpha`` to Elo ``K`` points."""

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive")
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    return alpha * scale / np.log(10.0)


def rating_to_skill(
    rating: float | np.ndarray,
    *,
    scale: float = DEFAULT_ELO_SCALE,
) -> float | np.ndarray:
    """Map Elo points to the natural-log Bradley-Terry skill coordinate."""

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive")
    skill = np.asarray(rating) * np.log(10.0) / scale
    if np.ndim(skill) == 0:
        return float(skill)
    return skill


def skill_to_rating(
    skill: float | np.ndarray,
    *,
    scale: float = DEFAULT_ELO_SCALE,
) -> float | np.ndarray:
    """Map natural-log Bradley-Terry skill back to Elo points."""

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive")
    rating = np.asarray(skill) * scale / np.log(10.0)
    if np.ndim(rating) == 0:
        return float(rating)
    return rating


def expected_score(
    rating_a: float | np.ndarray,
    rating_b: float | np.ndarray,
    *,
    scale: float = DEFAULT_ELO_SCALE,
    advantage_a: float = 0.0,
) -> float | np.ndarray:
    """Return the Bradley-Terry/Elo expected score for player A.

    ``scale=400`` gives the familiar base-10 Elo parameterization.  The
    implementation is algebraically equivalent to a logistic link.  Without
    draws, the expected score is also the win probability.  The exponent is
    clipped only to avoid floating-point overflow for extreme inputs.
    """

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive")
    if not np.isfinite(advantage_a):
        raise ValueError("advantage_a must be finite")
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
    scale: float = DEFAULT_ELO_SCALE
    advantage_a: float = 0.0
    ratings: dict[str, float] = field(default_factory=dict)
    games_played: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.k_factor) or self.k_factor < 0:
            raise ValueError("k_factor must be finite and nonnegative")
        if not np.isfinite(self.initial_rating):
            raise ValueError("initial_rating must be finite")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be positive")
        if not np.isfinite(self.advantage_a):
            raise ValueError("advantage_a must be finite")

    @staticmethod
    def _validate_players(player_a: str, player_b: str) -> None:
        if not player_a.strip() or not player_b.strip():
            raise ValueError("player names must be nonempty")
        if player_a.casefold() == player_b.casefold():
            raise ValueError("player_a and player_b must be distinct")

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

        self._validate_players(player_a, player_b)
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

        self._validate_players(player_a, player_b)
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
        if not np.isfinite(prediction_a) or not 0.0 <= prediction_a <= 1.0:
            raise ValueError("prediction_a must be finite and lie in [0, 1]")
        step = self.k_factor if k_factor is None else float(k_factor)
        if not np.isfinite(step) or step < 0:
            raise ValueError("k_factor must be finite and nonnegative")
        change = step * (float(outcome_a) - float(prediction_a))
        self.ratings[player_a] = rating_a + change
        self.ratings[player_b] = rating_b - change
        self.games_played[player_a] += 1
        self.games_played[player_b] += 1
        return change
