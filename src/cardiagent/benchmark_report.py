"""Reproducible benchmark reporting for CardiAgent.

This module turns deterministic benchmark suites into machine-readable reports
without depending on downstream CardiVex internals. It deliberately separates
suite construction from evaluation so reports can be regenerated from a fixed
suite version and seed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

from .evaluation import PopulationMetrics, conditional_domain_fidelity, population_metrics
from .models import ChallengeAgent
from .quality import assess_population


REPORT_VERSION = "0.1"


@dataclass(frozen=True)
class BenchmarkReport:
    """Serializable summary of one benchmark population."""

    report_version: str
    suite: str
    suite_version: str
    seed: int
    generated_at_utc: str
    metrics: dict[str, float | int]
    domain_fidelity: float
    quality_score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_report(
    challenges: Iterable[ChallengeAgent],
    *,
    suite: str,
    suite_version: str,
    seed: int,
) -> BenchmarkReport:
    """Build a deterministic report for an already materialized suite."""
    items = list(challenges)
    if not items:
        raise ValueError("At least one challenge is required")
    metrics: PopulationMetrics = population_metrics(items)
    quality = assess_population(items)
    return BenchmarkReport(
        report_version=REPORT_VERSION,
        suite=suite,
        suite_version=suite_version,
        seed=seed,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        metrics=metrics.to_dict(),
        domain_fidelity=conditional_domain_fidelity(items),
        quality_score=float(quality.score),
    )


def write_report(report: BenchmarkReport, path: str | Path) -> Path:
    """Write a stable, human-readable JSON report."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
