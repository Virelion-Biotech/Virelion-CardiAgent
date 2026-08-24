"""Deterministic benchmark suites for generator and downstream evaluation.

Suites are deliberately phenotype-level and contain no operational biological
instructions. Each suite is generated from fixed seeds and documented intent,
which makes it suitable for regression testing and locked benchmark creation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .generator import ChallengeGenerator
from .models import ChallengeAgent, ChallengeDomain


SUITE_VERSION = "0.1"


@dataclass(frozen=True)
class BenchmarkSuite:
    """Named, reproducible collection of phenotype-level challenges."""

    name: str
    version: str
    seed: int
    challenges: tuple[ChallengeAgent, ...]
    intent: str

    @property
    def case_count(self) -> int:
        return len(self.challenges)


def _generate_grid(seed: int, severities: tuple[float, ...], difficulties: tuple[float, ...]) -> list[ChallengeAgent]:
    challenges: list[ChallengeAgent] = []
    counter = 0
    for domain in ChallengeDomain:
        for severity in severities:
            for difficulty in difficulties:
                counter += 1
                generator = ChallengeGenerator(seed=seed + counter)
                challenge = generator.generate(
                    domain,
                    severity=severity,
                    difficulty=difficulty,
                    agent_id=f"suite-{seed}-{counter:04d}",
                )
                metadata = dict(challenge.metadata)
                metadata["benchmark_suite_version"] = SUITE_VERSION
                metadata["requested_domain"] = domain.value
                challenges.append(ChallengeAgent(
                    agent_id=challenge.agent_id,
                    domain=challenge.domain,
                    version=challenge.version,
                    seed=challenge.seed,
                    severity=challenge.severity,
                    onset=challenge.onset,
                    persistence=challenge.persistence,
                    heterogeneity=challenge.heterogeneity,
                    phenotype=challenge.phenotype,
                    metadata=metadata,
                ))
    return challenges


def baseline_suite(seed: int = 100) -> BenchmarkSuite:
    """Balanced low/moderate/high severity baseline population."""
    return BenchmarkSuite(
        name="baseline",
        version=SUITE_VERSION,
        seed=seed,
        challenges=tuple(_generate_grid(seed, (0.2, 0.5, 0.8), (0.2, 0.5))),
        intent="broad regression coverage across all supported domains",
    )


def difficulty_suite(seed: int = 200) -> BenchmarkSuite:
    """Hold severity approximately fixed while increasing ambiguity."""
    return BenchmarkSuite(
        name="difficulty",
        version=SUITE_VERSION,
        seed=seed,
        challenges=tuple(_generate_grid(seed, (0.6,), (0.1, 0.4, 0.7, 0.95))),
        intent="stress-test overlap, noise, missingness and heterogeneity",
    )


def severity_suite(seed: int = 300) -> BenchmarkSuite:
    """Evaluate behavior from subtle to severe challenge presentations."""
    return BenchmarkSuite(
        name="severity",
        version=SUITE_VERSION,
        seed=seed,
        challenges=tuple(_generate_grid(seed, (0.1, 0.3, 0.5, 0.7, 0.9), (0.4,))),
        intent="evaluate severity calibration and sensitivity",
    )


def overlap_suite(seed: int = 400) -> BenchmarkSuite:
    """Concentrate on high-overlap, difficult phenotype presentations."""
    suite = difficulty_suite(seed)
    selected = tuple(
        challenge
        for challenge in suite.challenges
        if float(challenge.metadata.get("phenotype_overlap", 0.0)) >= 0.60
    )
    return BenchmarkSuite(
        name="overlap",
        version=SUITE_VERSION,
        seed=seed,
        challenges=selected,
        intent="evaluate domain ambiguity and cross-domain phenotype overlap",
    )


SUITE_BUILDERS: dict[str, Callable[[int], BenchmarkSuite]] = {
    "baseline": baseline_suite,
    "difficulty": difficulty_suite,
    "severity": severity_suite,
    "overlap": overlap_suite,
}


def build_suite(name: str, *, seed: int | None = None) -> BenchmarkSuite:
    """Build one named suite; unknown names fail loudly."""
    try:
        builder = SUITE_BUILDERS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark suite: {name}; choose from {sorted(SUITE_BUILDERS)}") from exc
    return builder(seed) if seed is not None else builder(builder.__defaults__[0])  # type: ignore[index]


def available_suites() -> tuple[str, ...]:
    return tuple(SUITE_BUILDERS)
