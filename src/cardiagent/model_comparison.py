"""Matched deterministic-vs-ML generator comparison utilities.

The comparison is intentionally model-agnostic. It asks whether an ML generator
adds distributional fidelity, diversity, or conditional fidelity over the
existing deterministic prior under matched domain/severity/difficulty targets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from statistics import mean, pstdev
from typing import Iterable, Sequence

from .evaluation import population_metrics
from .generator import ChallengeGenerator
from .ml import AgentGeneratorModel
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
class GeneratorComparison:
    """Matched population comparison between two generators."""

    reference_count: int
    candidate_count: int
    reference_diversity: float
    candidate_diversity: float
    reference_duplicate_rate: float
    candidate_duplicate_rate: float
    phenotype_mean_distance: float
    phenotype_std_distance: float
    severity_mean_distance: float
    difficulty_mean_distance: float
    domain_coverage_difference: float
    candidate_diversity_gain: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _vector(agent: ChallengeAgent) -> tuple[float, ...]:
    values = agent.phenotype.to_dict()
    return tuple(float(values[name]) for name in PHENOTYPE_FIELDS)


def _mean_vector(items: Sequence[ChallengeAgent]) -> tuple[float, ...]:
    vectors = [_vector(item) for item in items]
    return tuple(mean(column) for column in zip(*vectors))


def _std_vector(items: Sequence[ChallengeAgent]) -> tuple[float, ...]:
    vectors = [_vector(item) for item in items]
    return tuple(pstdev(column) if len(column) > 1 else 0.0 for column in zip(*vectors))


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return sqrt(mean((x - y) ** 2 for x, y in zip(a, b)))


def compare_populations(
    reference: Iterable[ChallengeAgent],
    candidate: Iterable[ChallengeAgent],
) -> GeneratorComparison:
    """Compare two generated populations using distributional summaries."""
    ref = list(reference)
    pred = list(candidate)
    if not ref or not pred:
        raise ValueError("Both populations must contain at least one challenge")

    ref_metrics = population_metrics(ref)
    pred_metrics = population_metrics(pred)
    ref_mean = _mean_vector(ref)
    pred_mean = _mean_vector(pred)
    ref_std = _std_vector(ref)
    pred_std = _std_vector(pred)

    return GeneratorComparison(
        reference_count=len(ref),
        candidate_count=len(pred),
        reference_diversity=ref_metrics.phenotype_diversity,
        candidate_diversity=pred_metrics.phenotype_diversity,
        reference_duplicate_rate=ref_metrics.duplicate_rate,
        candidate_duplicate_rate=pred_metrics.duplicate_rate,
        phenotype_mean_distance=_euclidean(ref_mean, pred_mean),
        phenotype_std_distance=_euclidean(ref_std, pred_std),
        severity_mean_distance=abs(ref_metrics.mean_severity - pred_metrics.mean_severity),
        difficulty_mean_distance=abs(ref_metrics.mean_difficulty - pred_metrics.mean_difficulty),
        domain_coverage_difference=abs(ref_metrics.domain_coverage - pred_metrics.domain_coverage),
        candidate_diversity_gain=pred_metrics.phenotype_diversity - ref_metrics.phenotype_diversity,
    )


def build_reference_population(
    *,
    seed: int = 900,
    per_domain: int = 16,
    severities: Sequence[float] = (0.2, 0.5, 0.8),
    difficulty: float = 0.6,
) -> list[ChallengeAgent]:
    """Create a deterministic training/reference population with fixed targets."""
    if per_domain < 1:
        raise ValueError("per_domain must be positive")
    output: list[ChallengeAgent] = []
    index = 0
    for domain in ChallengeDomain:
        for item in range(per_domain):
            severity = float(severities[item % len(severities)])
            index += 1
            output.append(
                ChallengeGenerator(seed=seed + index).generate(
                    domain,
                    severity=severity,
                    difficulty=difficulty,
                    agent_id=f"reference-{seed}-{index:05d}",
                )
            )
    return output


def build_matched_ml_population(
    model: AgentGeneratorModel,
    *,
    per_domain: int = 16,
    severity: float = 0.5,
    difficulty: float = 0.6,
) -> list[ChallengeAgent]:
    """Generate an ML population under the same requested conditions."""
    if per_domain < 1:
        raise ValueError("per_domain must be positive")
    output: list[ChallengeAgent] = []
    for domain in ChallengeDomain:
        output.extend(
            model.sample(
                domain=domain,
                severity=severity,
                difficulty=difficulty,
                count=per_domain,
                agent_id_prefix=f"compare-{domain.value}",
            )
        )
    return output
