"""Reproducible benchmark reporting for CardiAgent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

from .calibration import CalibrationSummary, summarize_calibration
from .evaluation import PopulationMetrics, conditional_domain_fidelity, population_metrics
from .models import ChallengeAgent
from .quality import assess_population


REPORT_VERSION = "0.3"


@dataclass(frozen=True)
class BenchmarkReport:
    """Serializable summary of one benchmark population."""

    report_version: str
    suite: str
    suite_version: str
    seed: int
    generated_at_utc: str
    metrics: dict[str, float | int]
    calibration: dict[str, float | int]
    domain_fidelity: float
    quality_score: float
    acceptance: dict[str, bool]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_acceptance(
    metrics: PopulationMetrics,
    calibration: CalibrationSummary,
    domain_fidelity: float,
    quality_score: float,
    *,
    min_quality_score: float = 0.55,
    min_domain_fidelity: float = 0.99,
    max_duplicate_rate: float = 0.10,
) -> dict[str, bool]:
    """Apply explicit regression gates to a benchmark population."""
    return {
        "minimum_quality_score": quality_score >= min_quality_score,
        "domain_fidelity_threshold": domain_fidelity >= min_domain_fidelity,
        "duplicate_rate_threshold": metrics.duplicate_rate <= max_duplicate_rate,
        "nonempty_population": metrics.count > 0,
        "calibration_available": calibration.count == metrics.count,
    }


def build_report(
    challenges: Iterable[ChallengeAgent],
    *,
    suite: str,
    suite_version: str,
    seed: int,
    min_quality_score: float = 0.55,
    min_domain_fidelity: float = 0.99,
    max_duplicate_rate: float = 0.10,
) -> BenchmarkReport:
    """Build a deterministic report for an already materialized suite."""
    items = list(challenges)
    if not items:
        raise ValueError("At least one challenge is required")
    metrics: PopulationMetrics = population_metrics(items)
    quality = assess_population(items)
    calibration = summarize_calibration(items)
    domain_fidelity = conditional_domain_fidelity(items)
    acceptance = evaluate_acceptance(
        metrics,
        calibration,
        domain_fidelity,
        quality.quality_score,
        min_quality_score=min_quality_score,
        min_domain_fidelity=min_domain_fidelity,
        max_duplicate_rate=max_duplicate_rate,
    )
    return BenchmarkReport(
        report_version=REPORT_VERSION,
        suite=suite,
        suite_version=suite_version,
        seed=seed,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        metrics=metrics.to_dict(),
        calibration=calibration.to_dict(),
        domain_fidelity=domain_fidelity,
        quality_score=float(quality.quality_score),
        acceptance=acceptance,
    )


def assert_accepted(report: BenchmarkReport) -> None:
    """Fail loudly when any benchmark quality gate is not met."""
    failures = [name for name, passed in report.acceptance.items() if not passed]
    if failures:
        raise ValueError("Benchmark acceptance failed: " + ", ".join(sorted(failures)))


def write_report(report: BenchmarkReport, path: str | Path) -> Path:
    """Write a stable, human-readable JSON report."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
