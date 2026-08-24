"""Multi-seed generator comparison experiments with uncertainty summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from statistics import mean, pstdev
from typing import Sequence

from .ml import AgentGeneratorModel
from .model_comparison import build_matched_ml_population, build_reference_population, compare_populations
from .models import ChallengeAgent


EXPERIMENT_VERSION = "0.1"


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float
    ci95_half_width: float
    values: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MultiSeedExperiment:
    version: str
    seeds: tuple[int, ...]
    per_domain: int
    training_epochs: int
    phenotype_mean_distance: MetricSummary
    phenotype_std_distance: MetricSummary
    candidate_diversity_gain: MetricSummary
    severity_mean_distance: MetricSummary
    difficulty_mean_distance: MetricSummary

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _summary(values: Sequence[float]) -> MetricSummary:
    if not values:
        raise ValueError("At least one observation is required")
    average = mean(values)
    spread = pstdev(values) if len(values) > 1 else 0.0
    half_width = 1.96 * spread / sqrt(len(values)) if len(values) > 1 else 0.0
    return MetricSummary(average, spread, half_width, tuple(float(v) for v in values))


def run_multi_seed_experiment(
    training_agents: Sequence[ChallengeAgent],
    *,
    seeds: Sequence[int] = (11, 23, 37, 41, 53),
    per_domain: int = 16,
    training_epochs: int = 25,
    severity: float = 0.5,
    difficulty: float = 0.6,
) -> MultiSeedExperiment:
    """Compare deterministic and CVAE populations across independent seeds.

    The deterministic reference is regenerated for each seed, while the CVAE
    is independently initialized and trained for each seed. This avoids
    presenting a single lucky initialization as evidence of model superiority.
    """
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("At least one seed is required")
    if per_domain < 1 or training_epochs < 1:
        raise ValueError("per_domain and training_epochs must be positive")

    phenotype_mean: list[float] = []
    phenotype_std: list[float] = []
    diversity_gain: list[float] = []
    severity_distance: list[float] = []
    difficulty_distance: list[float] = []

    for seed in seed_values:
        model = AgentGeneratorModel(seed=seed)
        model.fit(training_agents, epochs=training_epochs, batch_size=64)
        reference = build_reference_population(
            seed=seed,
            per_domain=per_domain,
            severities=(severity,),
            difficulty=difficulty,
        )
        candidate = build_matched_ml_population(
            model,
            per_domain=per_domain,
            severity=severity,
            difficulty=difficulty,
        )
        comparison = compare_populations(reference, candidate)
        phenotype_mean.append(comparison.phenotype_mean_distance)
        phenotype_std.append(comparison.phenotype_std_distance)
        diversity_gain.append(comparison.candidate_diversity_gain)
        severity_distance.append(comparison.severity_mean_distance)
        difficulty_distance.append(comparison.difficulty_mean_distance)

    return MultiSeedExperiment(
        version=EXPERIMENT_VERSION,
        seeds=seed_values,
        per_domain=per_domain,
        training_epochs=training_epochs,
        phenotype_mean_distance=_summary(phenotype_mean),
        phenotype_std_distance=_summary(phenotype_std),
        candidate_diversity_gain=_summary(diversity_gain),
        severity_mean_distance=_summary(severity_distance),
        difficulty_mean_distance=_summary(difficulty_distance),
    )
