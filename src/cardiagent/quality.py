"""Quality gates for phenotype-level challenge populations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from statistics import mean
from typing import Sequence

from .models import ChallengeAgent


@dataclass(frozen=True)
class PopulationReport:
    count: int
    domain_balance: dict[str, int]
    mean_severity: float
    mean_difficulty: float
    diversity: float
    duplicate_rate: float
    quality_score: float
    warnings: tuple[str, ...]


def _vector(agent: ChallengeAgent) -> tuple[float, ...]:
    return tuple(agent.phenotype.to_dict().values()) + (
        agent.onset,
        agent.persistence,
        agent.heterogeneity,
    )


def _distance(a: ChallengeAgent, b: ChallengeAgent) -> float:
    av, bv = _vector(a), _vector(b)
    return sum(abs(x - y) for x, y in zip(av, bv)) / len(av)


def assess_population(challenges: Sequence[ChallengeAgent]) -> PopulationReport:
    """Assess coverage, duplication, and phenotype-space diversity."""
    if not challenges:
        raise ValueError("At least one challenge is required")
    counts: dict[str, int] = {}
    for challenge in challenges:
        counts[challenge.domain.value] = counts.get(challenge.domain.value, 0) + 1

    vectors = [_vector(c) for c in challenges]
    unique = len(set(vectors))
    duplicate_rate = 1.0 - unique / len(vectors)
    pair_distances = [_distance(a, b) for a, b in combinations(challenges, 2)]
    diversity = mean(pair_distances) if pair_distances else 0.0
    difficulty = mean(float(c.metadata.get("difficulty", 0.5)) for c in challenges)
    severity = mean(c.severity for c in challenges)

    warnings: list[str] = []
    if duplicate_rate > 0.15:
        warnings.append("high duplicate rate")
    if diversity < 0.08 and len(challenges) > 1:
        warnings.append("low phenotype diversity")
    if len(counts) < 2:
        warnings.append("single-domain population")
    if any(v / len(challenges) > 0.75 for v in counts.values()):
        warnings.append("domain imbalance")

    balance = min(1.0, len(counts) / 4.0)
    quality = max(0.0, min(1.0, 0.35 * min(1.0, diversity / 0.30) + 0.25 * (1 - duplicate_rate) + 0.20 * balance + 0.20 * difficulty))
    return PopulationReport(
        count=len(challenges),
        domain_balance=counts,
        mean_severity=severity,
        mean_difficulty=difficulty,
        diversity=diversity,
        duplicate_rate=duplicate_rate,
        quality_score=quality,
        warnings=tuple(warnings),
    )
