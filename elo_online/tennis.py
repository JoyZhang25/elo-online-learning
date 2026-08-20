"""Sequential Elo models used in the ATP evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import EloModel, expected_score


@dataclass
class SurfaceElo:
    """Elo with a global rating and a surface-specific player offset."""

    k_global: float
    k_surface: float
    initial_rating: float = 1500.0
    ratings: dict[str, float] = field(default_factory=dict)
    surface_offsets: dict[tuple[str, str], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.k_global) or self.k_global < 0:
            raise ValueError("k_global must be finite and nonnegative")
        if not np.isfinite(self.k_surface) or self.k_surface < 0:
            raise ValueError("k_surface must be finite and nonnegative")
        if not np.isfinite(self.initial_rating):
            raise ValueError("initial_rating must be finite")

    @staticmethod
    def _validate_players(player_a: str, player_b: str) -> None:
        if not player_a.strip() or not player_b.strip():
            raise ValueError("player names must be nonempty")
        if player_a.casefold() == player_b.casefold():
            raise ValueError("player_a and player_b must be distinct")

    def _global(self, player: str) -> float:
        return self.ratings.setdefault(player, self.initial_rating)

    def _offset(self, player: str, surface: str) -> float:
        return self.surface_offsets.setdefault((player, surface), 0.0)

    def predict(self, player_a: str, player_b: str, surface: str) -> float:
        self._validate_players(player_a, player_b)
        rating_a = self._global(player_a) + self._offset(player_a, surface)
        rating_b = self._global(player_b) + self._offset(player_b, surface)
        return float(expected_score(rating_a, rating_b))

    def update(
        self,
        player_a: str,
        player_b: str,
        surface: str,
        outcome_a: float,
        prediction_a: float,
    ) -> None:
        self._validate_players(player_a, player_b)
        if not np.isfinite(outcome_a) or not 0.0 <= outcome_a <= 1.0:
            raise ValueError("outcome_a must be finite and lie in [0, 1]")
        if not np.isfinite(prediction_a) or not 0.0 <= prediction_a <= 1.0:
            raise ValueError("prediction_a must be finite and lie in [0, 1]")
        residual = float(outcome_a) - float(prediction_a)
        global_change = self.k_global * residual
        surface_change = self.k_surface * residual
        self.ratings[player_a] = self._global(player_a) + global_change
        self.ratings[player_b] = self._global(player_b) - global_change
        key_a = (player_a, surface)
        key_b = (player_b, surface)
        self.surface_offsets[key_a] = self._offset(player_a, surface) + surface_change
        self.surface_offsets[key_b] = self._offset(player_b, surface) - surface_change


@dataclass
class MixtureElo:
    """Online log-loss mixture of Elo experts operating at different K values."""

    k_grid: tuple[float, ...]
    learning_rate: float
    models: list[EloModel] = field(init=False)
    log_weights: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not self.k_grid or any(
            not np.isfinite(k) or k < 0 for k in self.k_grid
        ):
            raise ValueError("k_grid must contain finite nonnegative values")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        self.models = [EloModel(k_factor=k) for k in self.k_grid]
        self.log_weights = np.full(len(self.models), -np.log(len(self.models)))

    @property
    def weights(self) -> np.ndarray:
        shifted = self.log_weights - np.max(self.log_weights)
        weights = np.exp(shifted)
        return weights / weights.sum()

    def predict(self, player_a: str, player_b: str) -> tuple[float, np.ndarray]:
        expert_probabilities = np.array(
            [model.predict(player_a, player_b) for model in self.models]
        )
        return float(self.weights @ expert_probabilities), expert_probabilities

    def update(
        self,
        player_a: str,
        player_b: str,
        outcome_a: float,
        expert_probabilities: np.ndarray,
    ) -> None:
        if not np.isfinite(outcome_a) or not 0.0 <= outcome_a <= 1.0:
            raise ValueError("outcome_a must be finite and lie in [0, 1]")
        raw_probabilities = np.asarray(expert_probabilities, dtype=float)
        if raw_probabilities.shape != (len(self.models),):
            raise ValueError("one expert probability is required per Elo model")
        if (
            not np.isfinite(raw_probabilities).all()
            or np.any(raw_probabilities < 0.0)
            or np.any(raw_probabilities > 1.0)
        ):
            raise ValueError("expert probabilities must be finite and lie in [0, 1]")
        probabilities = np.clip(raw_probabilities, 1e-12, 1.0 - 1e-12)
        losses = -(
            outcome_a * np.log(probabilities)
            + (1.0 - outcome_a) * np.log(1.0 - probabilities)
        )
        self.log_weights -= self.learning_rate * losses
        self.log_weights -= np.max(self.log_weights)
        for model, probability in zip(self.models, raw_probabilities):
            model.update(
                player_a,
                player_b,
                outcome_a,
                prediction_a=float(probability),
            )
