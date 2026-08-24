"""Model-agnostic evaluation metrics for CardiAgent populations.

The metrics operate only on phenotype-level challenge objects. They are
intended to make generator quality measurable before introducing heavier ML
models or external datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean
from typing import Iterable, Sequence

from .models import ChallengeAgent, ChallengeDomain


PHENOTYPE_FIELDS = (
    "stress",
    "inflammation",
    "electrical_instability",
    "contractile_impairment",
    "viability_loss",
    "oxidative_stress",
    "metabolic_disruption",
    "remodeling_signal",
)


@dataclass(frozen=True)
class PopulationMetrics:
    """Compact summary of distributional and conditional generator quality."""

    count: int
    domain_coverage: float
    mean_severity: float
    severity_std: float
    mean_difficulty: float
    phenotype_diversity: float
    duplicate_rate: float
    domain_balance: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "domain_coverage": self.domain_coverage,
            "mean_severity": self.mean_severity,
            "severity_std": self.severity_std,
            "mean_difficulty": self.mean_difficulty,
            "phenotype_diversity": self.phenotype_diversity,
            "duplicate_rate": self.duplicate_rate,
            "domain_balance": self.domain_balance,
        }


def _phenotype_vector(agent: ChallengeAgent) -> tuple[float, ...]:
    values = agent.phenotype.to_dict()
    return tuple(values[name] for name in PHENOTYPE_FIELDS)


def _distance(a: ChallengeAgent, b: ChallengeAgent) -> float:
    av = _phenotype_vector(a)
    bv = _phenotype_vector(b)
    return sqrt(sum((x - y) ** 2 for x, y in zip(av, bv)) / len(PHENOTYPE_FIELDS))


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def population_metrics(challenges: Iterable[ChallengeAgent]) -> PopulationMetrics:
    """Calculate reproducible quality metrics for a challenge population.

    ``domain_balance`` is one for a perfectly even distribution and approaches
    zero as the population collapses onto one domain. ``domain_coverage`` is
    the fraction of the seven supported domains represented.
    """
    items = list(challenges)
    if not items:
        raise ValueError("At least one challenge is required")

    domain_counts = {domain: 0 for domain in ChallengeDomain}
    for challenge in items:
        domain_counts[challenge.domain] += 1

    nonzero = [count for count in domain_counts.values() if count]
    max_count = max(domain_counts.values())
    ideal = len(items) / len(domain_counts)
    imbalance = mean(abs(count - ideal) for count in domain_counts.values()) / max(ideal, 1.0)

    vectors = [_phenotype_vector(challenge) for challenge in items]
    unique = len(set(vectors))
    duplicate_rate = 1.0 - unique / len(vectors)

    if len(items) > 1:
        distances = [_distance(a, b) for index, a in enumerate(items) for b in items[index + 1 :]]
        diversity = mean(distances)
    else:
        diversity = 0.0

    severities = [challenge.severity for challenge in items]
    difficulties = [float(challenge.metadata.get("difficulty", challenge.severity)) for challenge in items]

    return PopulationMetrics(
        count=len(items),
        domain_coverage=len(nonzero) / len(domain_counts),
        mean_severity=mean(severities),
        severity_std=_std(severities),
        mean_difficulty=mean(difficulties),
        phenotype_diversity=diversity,
        duplicate_rate=duplicate_rate,
        domain_balance=max(0.0, 1.0 - imbalance),
    )


def phenotype_mae(reference: Iterable[ChallengeAgent], candidate: Iterable[ChallengeAgent]) -> float:
    """Mean absolute phenotype error between two aligned populations."""
    ref = list(reference)
    pred = list(candidate)
    if not ref or not pred:
        raise ValueError("Both populations must contain at least one challenge")
    if len(ref) != len(pred):
        raise ValueError("Reference and candidate populations must be aligned and equal in length")
    errors = []
    for left, right in zip(ref, pred):
        errors.extend(abs(a - b) for a, b in zip(_phenotype_vector(left), _phenotype_vector(right)))
    return mean(errors)


def conditional_domain_fidelity(challenges: Iterable[ChallengeAgent]) -> float:
    """Measure whether generated cases span all requested domains.

    This metric intentionally does not assess biological correctness; it only
    checks that the conditional generator actually honored its domain target.
    """
    items = list(challenges)
    if not items:
        raise ValueError("At least one challenge is required")
    requested = [challenge.metadata.get("requested_domain", challenge.domain.value) for challenge in items]
    matches = sum(request == challenge.domain.value for request, challenge in zip(requested, items))
    return matches / len(items)
