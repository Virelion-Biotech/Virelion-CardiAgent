"""Adaptive phenotype-level challenge generation.

This module lets CardiAgent learn from downstream benchmark outcomes and produce
harder *abstract phenotype* challenges. It never constructs or optimizes
operational biological parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Iterable, Mapping, Sequence

from .models import ChallengeAgent, ChallengeDomain, PhenotypeProfile


@dataclass(frozen=True)
class DetectionOutcome:
    """A CardiVex evaluation for one blinded case."""

    case_id: str
    predicted_domain: str | None
    confidence: float = 0.0
    detected: bool = False
    characterization_correct: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")


@dataclass(frozen=True)
class AdaptiveScore:
    """Difficulty signal assigned to a challenge from downstream performance."""

    case_id: str
    hardness: float
    reason: str


@dataclass(frozen=True)
class CurriculumStage:
    """One controlled difficulty stage."""

    name: str
    difficulty: float
    overlap: float
    heterogeneity: float
    noise: float
    partial_observation: float


DEFAULT_CURRICULUM: tuple[CurriculumStage, ...] = (
    CurriculumStage("baseline", 0.25, 0.15, 0.15, 0.02, 0.00),
    CurriculumStage("moderate", 0.50, 0.35, 0.30, 0.06, 0.05),
    CurriculumStage("hard", 0.75, 0.60, 0.50, 0.12, 0.12),
    CurriculumStage("stress", 0.90, 0.78, 0.68, 0.18, 0.20),
    CurriculumStage("edge", 0.98, 0.88, 0.82, 0.24, 0.28),
)


class AdaptiveChallengeEngine:
    """Turn CardiVex outcomes into harder, reproducible phenotype challenges.

    The engine does not require access to CardiVex internals. It only consumes
    blinded-case outcomes, making the two systems independently testable.
    """

    VERSION = "0.4-adaptive"

    def __init__(self, *, seed: int = 0, curriculum: Sequence[CurriculumStage] = DEFAULT_CURRICULUM):
        self.seed = seed
        self.curriculum = tuple(curriculum)
        self._scores: dict[str, AdaptiveScore] = {}

    def score(self, outcomes: Iterable[DetectionOutcome]) -> list[AdaptiveScore]:
        """Score cases so misses and uncertain calls receive more attention."""
        scores: list[AdaptiveScore] = []
        for outcome in outcomes:
            miss = 1.0 if not outcome.detected else 0.0
            wrong = 1.0 if not outcome.characterization_correct else 0.0
            uncertainty = 1.0 - outcome.confidence
            hardness = min(1.0, 0.60 * miss + 0.30 * wrong + 0.10 * uncertainty)
            reason = "miss" if miss else "mischaracterization" if wrong else "low-confidence"
            item = AdaptiveScore(outcome.case_id, hardness, reason)
            self._scores[outcome.case_id] = item
            scores.append(item)
        return scores

    def next_stage(self, mean_hardness: float) -> CurriculumStage:
        """Choose a curriculum stage from observed downstream performance."""
        if not 0.0 <= mean_hardness <= 1.0:
            raise ValueError("mean_hardness must be within [0, 1]")
        # If CardiVex is already struggling, do not jump immediately to the
        # hardest regime; this keeps the benchmark diagnostically useful.
        target = min(0.98, 0.25 + 0.75 * mean_hardness)
        return min(self.curriculum, key=lambda stage: abs(stage.difficulty - target))

    def evolve(
        self,
        parents: Sequence[ChallengeAgent],
        *,
        count: int,
        stage: CurriculumStage | None = None,
        mutation_rate: float = 0.18,
        mutation_scale: float = 0.14,
    ) -> list[ChallengeAgent]:
        """Create a new generation by safe phenotype-space recombination/mutation."""
        if not parents:
            raise ValueError("At least one parent challenge is required")
        if count < 1:
            raise ValueError("count must be positive")
        if not 0.0 <= mutation_rate <= 1.0 or mutation_scale < 0.0:
            raise ValueError("invalid mutation parameters")
        stage = stage or self.curriculum[-1]
        rng = random.Random(self.seed + len(self._scores) + count)
        output: list[ChallengeAgent] = []

        for index in range(count):
            a = rng.choice(parents)
            b = rng.choice(parents)
            pa = a.phenotype.to_dict()
            pb = b.phenotype.to_dict()
            values: dict[str, float] = {}
            for field in pa:
                value = (pa[field] + pb[field]) / 2.0
                if rng.random() < mutation_rate:
                    value += rng.gauss(0.0, mutation_scale)
                values[field] = max(0.0, min(1.0, value))

            # Stage controls challenge-level observability, not any operational
            # biological parameter.
            metadata = dict(a.metadata)
            metadata.update({
                "generator": "virelion-cardiagent-adaptive",
                "generator_version": self.VERSION,
                "ml_generated": bool(metadata.get("ml_generated", False)),
                "adaptive_generation": True,
                "curriculum_stage": stage.name,
                "difficulty": stage.difficulty,
                "phenotype_overlap": stage.overlap,
                "measurement_noise": stage.noise,
                "partial_observation_rate": stage.partial_observation,
                "parent_ids": [a.agent_id, b.agent_id],
            })
            agent_id = self._id(a, b, index)
            output.append(
                ChallengeAgent(
                    agent_id=agent_id,
                    domain=a.domain,
                    version=self.VERSION,
                    seed=self.seed,
                    severity=max(0.0, min(1.0, (a.severity + b.severity) / 2.0)),
                    onset=max(0.0, min(1.0, (a.onset + b.onset) / 2.0)),
                    persistence=max(0.0, min(1.0, (a.persistence + b.persistence) / 2.0)),
                    heterogeneity=stage.heterogeneity,
                    phenotype=PhenotypeProfile(**values),
                    metadata=metadata,
                )
            )
        return output

    def hard_cases(self, cases: Sequence[ChallengeAgent], *, top_k: int = 10) -> list[ChallengeAgent]:
        """Return cases whose IDs have the highest downstream hardness."""
        ranked = sorted(cases, key=lambda c: self._scores.get(c.agent_id, AdaptiveScore(c.agent_id, 0.0, "unscored")).hardness, reverse=True)
        return ranked[:top_k]

    def _id(self, a: ChallengeAgent, b: ChallengeAgent, index: int) -> str:
        raw = f"{self.seed}|{a.agent_id}|{b.agent_id}|{index}".encode()
        return f"ADAPT-{hashlib.sha256(raw).hexdigest()[:16]}"
