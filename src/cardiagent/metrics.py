"""Evaluation metrics for CardiAgent challenge populations and CardiVex outcomes.

The metrics are dependency-light and operate on the abstract phenotype-level
objects already used by CardiAgent. They are intended for reproducible
benchmark reports rather than model training.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean, pstdev
from typing import Iterable, Sequence

from .adaptive import DetectionOutcome
from .models import ChallengeAgent, ChallengeDomain


@dataclass(frozen=True)
class DetectionMetrics:
    """Summary metrics for blinded detection and characterization."""

    count: int
    detected_rate: float
    characterization_accuracy: float
    domain_accuracy: float
    macro_f1: float
    mean_confidence: float
    mean_hardness: float


@dataclass(frozen=True)
class PopulationMetrics:
    """Distributional metrics for a generated challenge population."""

    count: int
    mean_vector: tuple[float, ...]
    std_vector: tuple[float, ...]
    min_nearest_neighbor_distance: float
    mean_nearest_neighbor_distance: float
    mean_severity: float
    mean_difficulty: float


PHENOTYPE_VECTOR_FIELDS = (
    "stress",
    "inflammation",
    "electrical_instability",
    "contractile_impairment",
    "viability_loss",
    "oxidative_stress",
    "metabolic_disruption",
    "remodeling_signal",
    "onset",
    "persistence",
    "heterogeneity",
)


def _vector(agent: ChallengeAgent) -> tuple[float, ...]:
    phenotype = agent.phenotype.to_dict()
    return tuple(phenotype[field] for field in PHENOTYPE_VECTOR_FIELDS[:8]) + (
        agent.onset,
        agent.persistence,
        agent.heterogeneity,
    )


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def detection_metrics(
    outcomes: Iterable[DetectionOutcome],
    truth_domains: dict[str, str] | None = None,
) -> DetectionMetrics:
    """Compute detection and domain-characterization metrics.

    ``truth_domains`` maps case IDs to the hidden benchmark domain. When it is
    omitted, domain accuracy is reported as zero rather than inferred from the
    predictions.
    """
    rows = list(outcomes)
    count = len(rows)
    if count == 0:
        raise ValueError("At least one outcome is required")

    detected_rate = mean(float(row.detected) for row in rows)
    characterization_accuracy = mean(float(row.characterization_correct) for row in rows)
    mean_confidence = mean(row.confidence for row in rows)
    hardness = [
        min(
            1.0,
            0.60 * (not row.detected)
            + 0.30 * (not row.characterization_correct)
            + 0.10 * (1.0 - row.confidence),
        )
        for row in rows
    ]

    if truth_domains:
        labels = sorted(set(truth_domains.values()) | {row.predicted_domain for row in rows if row.predicted_domain})
        f1_values = []
        correct = 0
        for label in labels:
            tp = sum(1 for row in rows if truth_domains.get(row.case_id) == label and row.predicted_domain == label)
            fp = sum(1 for row in rows if truth_domains.get(row.case_id) != label and row.predicted_domain == label)
            fn = sum(1 for row in rows if truth_domains.get(row.case_id) == label and row.predicted_domain != label)
            if tp + fp + fn == 0:
                continue
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        domain_accuracy = mean(float(truth_domains.get(row.case_id) == row.predicted_domain) for row in rows)
        macro_f1 = mean(f1_values) if f1_values else 0.0
    else:
        domain_accuracy = 0.0
        macro_f1 = 0.0

    return DetectionMetrics(
        count=count,
        detected_rate=detected_rate,
        characterization_accuracy=characterization_accuracy,
        domain_accuracy=domain_accuracy,
        macro_f1=macro_f1,
        mean_confidence=mean_confidence,
        mean_hardness=mean(hardness),
    )


def population_metrics(challenges: Sequence[ChallengeAgent]) -> PopulationMetrics:
    """Measure spread and local novelty in a generated population."""
    if not challenges:
        raise ValueError("At least one challenge is required")
    vectors = [_vector(agent) for agent in challenges]
    mean_vector = tuple(mean(vector[i] for vector in vectors) for i in range(len(vectors[0])))
    std_vector = tuple(pstdev(vector[i] for vector in vectors) if len(vectors) > 1 else 0.0 for i in range(len(vectors[0])))

    nearest: list[float] = []
    for index, vector in enumerate(vectors):
        distances = [_distance(vector, other) for j, other in enumerate(vectors) if j != index]
        if distances:
            nearest.append(min(distances))

    return PopulationMetrics(
        count=len(challenges),
        mean_vector=mean_vector,
        std_vector=std_vector,
        min_nearest_neighbor_distance=min(nearest) if nearest else 0.0,
        mean_nearest_neighbor_distance=mean(nearest) if nearest else 0.0,
        mean_severity=mean(agent.severity for agent in challenges),
        mean_difficulty=mean(float(agent.metadata.get("difficulty", 0.5)) for agent in challenges),
    )


def conditional_fidelity(
    generated: Sequence[ChallengeAgent],
    *,
    domain: ChallengeDomain,
    severity: float,
    tolerance: float = 0.20,
) -> float:
    """Score whether generated cases respect requested domain/severity conditioning."""
    if not generated:
        raise ValueError("At least one generated challenge is required")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    domain_rate = mean(float(agent.domain == domain) for agent in generated)
    severity_score = mean(max(0.0, 1.0 - abs(agent.severity - severity) / max(tolerance, 1e-9)) for agent in generated)
    return 0.5 * domain_rate + 0.5 * severity_score


def reproducibility_signature(challenges: Sequence[ChallengeAgent]) -> tuple[str, ...]:
    """Return canonical signatures useful for exact-seed reproducibility tests."""
    return tuple(agent.to_json() for agent in challenges)
