"""
OpenAI-ES: Evolution Strategies as a Scalable Alternative to RL.
Salimans et al., 2017 — https://arxiv.org/abs/1703.03864

Key design choices:
  - Antithetic sampling: for each noise vector ε we also evaluate −ε.
    This halves the variance of the gradient estimate at zero extra cost.
  - Fitness rank normalisation: raw fitnesses are replaced by their centred
    ranks in [−0.5, 0.5].  This removes the effect of fitness scale and
    outliers, acting like a robust gradient shaping.
  - The update is a pseudo-gradient step: θ ← θ + α · ĝ where ĝ is the
    finite-difference gradient estimate over the population.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rank_normalize(fitnesses: np.ndarray) -> np.ndarray:
    """Map raw fitnesses to centred ranks in [−0.5, 0.5]."""
    n = len(fitnesses)
    ranks = np.argsort(np.argsort(fitnesses)).astype(np.float32)
    return (ranks / (n - 1)) - 0.5


# ---------------------------------------------------------------------------
# State snapshot (useful for checkpointing)
# ---------------------------------------------------------------------------

@dataclass
class ESState:
    theta: np.ndarray          # current solution (centroid)
    generation: int
    best_fitness: float
    best_theta: np.ndarray     # weights that achieved best_fitness so far
    fitness_history: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

class OpenAIES:
    """
    Gradient-free optimiser operating on a flat numpy parameter vector.

    Usage
    -----
    es = OpenAIES(theta_init, ...)
    for _ in range(num_generations):
        population = es.ask()          # list[np.ndarray]
        fitnesses  = [f(p) for p in population]
        es.tell(fitnesses)
        print(es.state.best_fitness)
    """

    def __init__(
        self,
        theta_init: np.ndarray,
        sigma: float = 0.01,
        learning_rate: float = 0.01,
        population_size: int = 50,
        antithetic_sampling: bool = True,
        seed: int = 42,
    ) -> None:
        if antithetic_sampling and population_size % 2 != 0:
            raise ValueError(
                "population_size must be even when antithetic_sampling=True"
            )

        self.sigma = sigma
        self.lr = learning_rate
        self.pop_size = population_size
        self.antithetic = antithetic_sampling
        self.rng = np.random.default_rng(seed)

        self.theta: np.ndarray = theta_init.copy().astype(np.float32)
        self._perturbations: np.ndarray | None = None

        self.state = ESState(
            theta=self.theta.copy(),
            generation=0,
            best_fitness=-np.inf,
            best_theta=self.theta.copy(),
        )

        logger.info(
            f"OpenAI-ES initialised | dim={len(self.theta):,} | "
            f"pop={population_size} | σ={sigma} | lr={learning_rate} | "
            f"antithetic={antithetic_sampling}"
        )

    # ------------------------------------------------------------------
    # Core ES interface
    # ------------------------------------------------------------------

    def ask(self) -> list[np.ndarray]:
        """
        Sample a population of perturbed solutions around the current centroid.

        With antithetic sampling, half the population uses +ε and half uses −ε
        for the same base perturbation vectors.

        Returns
        -------
        List of ``population_size`` flat parameter vectors.
        """
        dim = len(self.theta)

        if self.antithetic:
            half = self.pop_size // 2
            eps = self.rng.standard_normal((half, dim)).astype(np.float32)
            # Stack positive and negative perturbations
            self._perturbations = np.concatenate([eps, -eps], axis=0)
        else:
            self._perturbations = self.rng.standard_normal(
                (self.pop_size, dim)
            ).astype(np.float32)

        return [self.theta + self.sigma * eps_i for eps_i in self._perturbations]

    def tell(self, fitnesses: list[float]) -> None:
        """
        Update the centroid using the population fitnesses.

        The pseudo-gradient is:
            ĝ = (1 / (N · σ)) · Σ_i F̃_i · ε_i
        where F̃_i are the centred-rank-normalised fitnesses and ε_i are the
        stored perturbations.

        With antithetic sampling this simplifies to:
            ĝ = (1 / (N · σ)) · Σ_{i=1}^{N/2} (F̃_i⁺ − F̃_i⁻) · ε_i
        which reduces variance further.
        """
        if self._perturbations is None:
            raise RuntimeError("Call ask() before tell().")
        if len(fitnesses) != self.pop_size:
            raise ValueError(
                f"Expected {self.pop_size} fitnesses, got {len(fitnesses)}."
            )

        F = np.array(fitnesses, dtype=np.float32)
        F_shaped = _rank_normalize(F)

        if self.antithetic:
            half = self.pop_size // 2
            pos_eps = self._perturbations[:half]  # [half, dim]
            gradient = np.sum(
                (F_shaped[:half] - F_shaped[half:])[:, np.newaxis] * pos_eps,
                axis=0,
            ) / (self.pop_size * self.sigma)
        else:
            gradient = np.sum(
                F_shaped[:, np.newaxis] * self._perturbations,
                axis=0,
            ) / (self.pop_size * self.sigma)

        self.theta = self.theta + self.lr * gradient

        # Track best solution seen across the whole population
        best_idx = int(np.argmax(F))
        if F[best_idx] > self.state.best_fitness:
            self.state.best_fitness = float(F[best_idx])
            self.state.best_theta = (
                self.theta - self.lr * gradient  # centroid before the update
                + self.sigma * self._perturbations[best_idx]
            )

        self.state.generation += 1
        self.state.theta = self.theta.copy()
        self.state.fitness_history.append(float(np.mean(F)))
        self._perturbations = None

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def get_checkpoint(self) -> dict:
        """Return a dict that fully captures the optimizer state for resume."""
        return {
            "theta": self.theta.copy(),
            "best_theta": self.state.best_theta.copy(),
            "best_fitness": self.state.best_fitness,
            "generation": self.state.generation,
            "fitness_history": list(self.state.fitness_history),
            "rng_state": self.rng.__getstate__(),
            "sigma": self.sigma,
            "lr": self.lr,
            "pop_size": self.pop_size,
            "antithetic": self.antithetic,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "OpenAIES":
        """Reconstruct an optimizer from a saved checkpoint."""
        dummy_theta = checkpoint["theta"]
        es = cls.__new__(cls)
        es.sigma = checkpoint["sigma"]
        es.lr = checkpoint["lr"]
        es.pop_size = checkpoint["pop_size"]
        es.antithetic = checkpoint["antithetic"]
        es.rng = np.random.default_rng()
        es.rng.__setstate__(checkpoint["rng_state"])
        es.theta = dummy_theta.copy()
        es._perturbations = None
        es.state = ESState(
            theta=dummy_theta.copy(),
            generation=checkpoint["generation"],
            best_fitness=checkpoint["best_fitness"],
            best_theta=checkpoint["best_theta"].copy(),
            fitness_history=list(checkpoint["fitness_history"]),
        )
        logger.info(
            f"OpenAI-ES restored from checkpoint | gen={es.state.generation} | "
            f"best={es.state.best_fitness:.4f} | dim={len(es.theta):,}"
        )
        return es

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return len(self.theta)
