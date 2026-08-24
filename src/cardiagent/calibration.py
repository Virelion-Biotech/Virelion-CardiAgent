"""Calibration and realized-condition diagnostics for generated populations."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Iterable

from .models import ChallengeAgent


@dataclass(frozen=True)
class CalibrationSummary:
    count: int
    requested_severity_mean: float
    realized_phenotype_mean: float
    severity_phenotype_mae: float
    difficulty_mean: float
    overlap_mean: float
    noise_mean: float
    missingness_mean: float
    difficulty_overlap_mae: float
    difficulty_noise_mae: float
    difficulty_missingness_mae: float
    severity_dispersion: float

    def to_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _phenotype_mean(agent: ChallengeAgent) -> float:
    return _mean([float(value) for value in agent.phenotype.to_dict().values()])


def summarize_calibration(challenges: Iterable[ChallengeAgent]) -> CalibrationSummary:
    """Measure whether requested controls are reflected in generated outputs."""
    items = list(challenges)
    if not items:
        raise ValueError("At least one challenge is required")

    severity = [float(item.severity) for item in items]
    phenotype = [_phenotype_mean(item) for item in items]
    difficulty = [float(item.metadata.get("difficulty", item.severity)) for item in items]
    overlap = [float(item.metadata.get("phenotype_overlap", 0.0)) for item in items]
    noise = [float(item.metadata.get("measurement_noise", 0.0)) for item in items]
    missingness = [float(item.metadata.get("partial_observation_rate", 0.0)) for item in items]

    expected_overlap = [(x - 0.15) / 0.65 for x in overlap]
    expected_noise = [(x - 0.05) / 0.35 for x in noise]
    expected_missingness = [(x - 0.02) / 0.16 for x in missingness]

    return CalibrationSummary(
        count=len(items),
        requested_severity_mean=_mean(severity),
        realized_phenotype_mean=_mean(phenotype),
        severity_phenotype_mae=_mean([abs(a - b) for a, b in zip(severity, phenotype)]),
        difficulty_mean=_mean(difficulty),
        overlap_mean=_mean(overlap),
        noise_mean=_mean(noise),
        missingness_mean=_mean(missingness),
        difficulty_overlap_mae=_mean([abs(a - b) for a, b in zip(difficulty, expected_overlap)]),
        difficulty_noise_mae=_mean([abs(a - b) for a, b in zip(difficulty, expected_noise)]),
        difficulty_missingness_mae=_mean([abs(a - b) for a, b in zip(difficulty, expected_missingness)]),
        severity_dispersion=pstdev(severity) if len(severity) > 1 else 0.0,
    )


def monotonicity_score(pairs: Iterable[tuple[float, float]]) -> float:
    """Return the fraction of ordered adjacent pairs that move together."""
    ordered = sorted(pairs)
    if len(ordered) < 2:
        return 1.0
    comparisons = 0
    correct = 0
    for (x1, y1), (x2, y2) in zip(ordered, ordered[1:]):
        if x2 == x1:
            continue
        comparisons += 1
        if y2 >= y1:
            correct += 1
    return correct / comparisons if comparisons else 1.0
