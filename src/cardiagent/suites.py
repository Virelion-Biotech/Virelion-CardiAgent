"""Deterministic benchmark suites for generator and downstream evaluation.

Suites are phenotype-level, reproducible, and contain no operational
biological instructions. They provide fixed regression and stress-test cases
for generator quality and downstream model evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .generator import ChallengeGenerator
from .models import ChallengeAgent, ChallengeDomain


SUITE_VERSION = "0.2"
_DEFAULT_SEEDS = {
    "baseline": 100,
    "difficulty": 200,
    "severity": 300,
    "overlap": 400,
    "temporal": 500,
    "heterogeneity": 600,
    "partial_observation": 700,
    "ood": 800,
}


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


def _with_metadata(challenge: ChallengeAgent, **updates: object) -> ChallengeAgent:
    metadata = dict(challenge.metadata)
    metadata["benchmark_suite_version"] = SUITE_VERSION
    metadata.update(updates)
    return ChallengeAgent(
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
    )


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
                challenges.append(_with_metadata(
                    challenge,
                    requested_domain=domain.value,
                    requested_severity=severity,
                    requested_difficulty=difficulty,
                ))
    return challenges


def baseline_suite(seed: int = 100) -> BenchmarkSuite:
    """Balanced low/moderate/high severity baseline population."""
    return BenchmarkSuite("baseline", SUITE_VERSION, seed, tuple(_generate_grid(seed, (0.2, 0.5, 0.8), (0.2, 0.5))), "broad regression coverage across all supported domains")


def difficulty_suite(seed: int = 200) -> BenchmarkSuite:
    """Hold severity approximately fixed while increasing ambiguity."""
    return BenchmarkSuite("difficulty", SUITE_VERSION, seed, tuple(_generate_grid(seed, (0.6,), (0.1, 0.4, 0.7, 0.95))), "stress-test overlap, noise, missingness and heterogeneity")


def severity_suite(seed: int = 300) -> BenchmarkSuite:
    """Evaluate behavior from subtle to severe challenge presentations."""
    return BenchmarkSuite("severity", SUITE_VERSION, seed, tuple(_generate_grid(seed, (0.1, 0.3, 0.5, 0.7, 0.9), (0.4,))), "evaluate severity calibration and sensitivity")


def overlap_suite(seed: int = 400) -> BenchmarkSuite:
    """Concentrate on high-overlap phenotype presentations."""
    suite = difficulty_suite(seed)
    selected = tuple(challenge for challenge in suite.challenges if float(challenge.metadata.get("phenotype_overlap", 0.0)) >= 0.60)
    return BenchmarkSuite("overlap", SUITE_VERSION, seed, selected, "evaluate domain ambiguity and cross-domain phenotype overlap")


def temporal_suite(seed: int = 500) -> BenchmarkSuite:
    """Cover early, intermediate and persistent temporal presentations."""
    challenges: list[ChallengeAgent] = []
    counter = 0
    for domain in ChallengeDomain:
        for onset, persistence in ((0.05, 0.20), (0.30, 0.50), (0.70, 0.90)):
            counter += 1
            challenge = ChallengeGenerator(seed=seed + counter).generate(
                domain, severity=0.6, difficulty=0.6, agent_id=f"suite-{seed}-{counter:04d}"
            )
            challenge = _with_metadata(challenge, requested_domain=domain.value, requested_onset=onset, requested_persistence=persistence, temporal_target=f"onset={onset:.2f};persistence={persistence:.2f}")
            challenges.append(challenge)
    return BenchmarkSuite("temporal", SUITE_VERSION, seed, tuple(challenges), "test temporal diversity and conditioning")


def heterogeneity_suite(seed: int = 600) -> BenchmarkSuite:
    """Test homogeneous through highly heterogeneous populations."""
    challenges: list[ChallengeAgent] = []
    counter = 0
    for domain in ChallengeDomain:
        for heterogeneity in (0.05, 0.35, 0.65, 0.90):
            counter += 1
            challenge = ChallengeGenerator(seed=seed + counter).generate(domain, severity=0.6, difficulty=0.7, agent_id=f"suite-{seed}-{counter:04d}")
            challenges.append(_with_metadata(challenge, requested_domain=domain.value, requested_heterogeneity=heterogeneity))
    return BenchmarkSuite("heterogeneity", SUITE_VERSION, seed, tuple(challenges), "test robustness to biological presentation heterogeneity")


def partial_observation_suite(seed: int = 700) -> BenchmarkSuite:
    """Stress-test cases with intentionally incomplete observable phenotypes."""
    challenges = _generate_grid(seed, (0.3, 0.6, 0.9), (0.7, 0.95))
    return BenchmarkSuite("partial_observation", SUITE_VERSION, seed, tuple(_with_metadata(c, observation_regime="partial") for c in challenges), "test robustness to missing or noisy observables")


def ood_suite(seed: int = 800) -> BenchmarkSuite:
    """Construct a deterministic out-of-distribution severity/difficulty regime."""
    challenges: list[ChallengeAgent] = []
    counter = 0
    for domain in ChallengeDomain:
        for severity, difficulty in ((0.15, 0.95), (0.95, 0.15), (0.95, 0.95)):
            counter += 1
            challenge = ChallengeGenerator(seed=seed + counter).generate(domain, severity=severity, difficulty=difficulty, agent_id=f"suite-{seed}-{counter:04d}")
            challenges.append(_with_metadata(challenge, requested_domain=domain.value, ood_regime=True, requested_severity=severity, requested_difficulty=difficulty))
    return BenchmarkSuite("ood", SUITE_VERSION, seed, tuple(challenges), "evaluate generalization to unusual severity-difficulty combinations")


SUITE_BUILDERS: dict[str, Callable[[int], BenchmarkSuite]] = {
    "baseline": baseline_suite,
    "difficulty": difficulty_suite,
    "severity": severity_suite,
    "overlap": overlap_suite,
    "temporal": temporal_suite,
    "heterogeneity": heterogeneity_suite,
    "partial_observation": partial_observation_suite,
    "ood": ood_suite,
}


def build_suite(name: str, *, seed: int | None = None) -> BenchmarkSuite:
    """Build one named suite; unknown names fail loudly."""
    if name not in SUITE_BUILDERS:
        raise KeyError(f"Unknown benchmark suite: {name}; choose from {sorted(SUITE_BUILDERS)}")
    return SUITE_BUILDERS[name](seed if seed is not None else _DEFAULT_SEEDS[name])


def available_suites() -> tuple[str, ...]:
    return tuple(SUITE_BUILDERS)
