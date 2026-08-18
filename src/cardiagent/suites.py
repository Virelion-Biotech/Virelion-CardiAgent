"""Standardized benchmark suite construction for CardiAgent.

Suites vary only abstract challenge properties: domain coverage, overlap,
heterogeneity, noise, temporal persistence, and partial observation.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .benchmark import BlindBenchmark, build_blind_benchmark
from .generator import ChallengeGenerator
from .models import ChallengeAgent, ChallengeDomain


SUITE_VERSIONS = {
    "baseline": "1.0",
    "cross_domain": "1.0",
    "overlap": "1.0",
    "temporal": "1.0",
    "observation": "1.0",
    "stress": "1.0",
}


def build_suite(
    suite: str,
    *,
    benchmark_id: str | None = None,
    domains: Sequence[ChallengeDomain] | None = None,
    per_domain: int = 8,
    seed: int = 0,
) -> BlindBenchmark:
    """Build one deterministic benchmark suite from the canonical generator."""
    if suite not in SUITE_VERSIONS:
        raise ValueError(f"Unknown suite {suite!r}; choose from {sorted(SUITE_VERSIONS)}")
    if per_domain < 1:
        raise ValueError("per_domain must be positive")

    selected = tuple(domains or tuple(ChallengeDomain))
    generator = ChallengeGenerator(seed=seed)
    challenges: list[ChallengeAgent] = []

    for domain_index, domain in enumerate(selected):
        for index in range(per_domain):
            severity = 0.20 + 0.12 * (index % 6)
            difficulty = 0.20 + 0.15 * (index % 5)
            if suite == "baseline":
                difficulty = min(difficulty, 0.45)
            elif suite == "cross_domain":
                difficulty = 0.60
            elif suite == "overlap":
                difficulty = 0.90
            elif suite == "temporal":
                difficulty = 0.72
            elif suite == "observation":
                difficulty = 0.82
            elif suite == "stress":
                difficulty = 0.96

            agent = generator.generate(
                domain,
                severity=severity,
                difficulty=difficulty,
                agent_id=f"{suite.upper()}-{domain_index:02d}-{index:04d}",
            )
            metadata = dict(agent.metadata)
            metadata.update({
                "benchmark_suite": suite,
                "benchmark_suite_version": SUITE_VERSIONS[suite],
            })

            if suite == "observation":
                metadata["measurement_noise"] = max(0.15, float(metadata.get("measurement_noise", 0.0)))
                metadata["partial_observation_rate"] = max(0.12, float(metadata.get("partial_observation_rate", 0.0)))
            elif suite == "temporal":
                metadata["temporal_focus"] = "trajectory_sensitive"
            elif suite == "overlap":
                metadata["phenotype_overlap"] = max(0.70, float(metadata.get("phenotype_overlap", 0.0)))
            elif suite == "stress":
                metadata["phenotype_overlap"] = max(0.82, float(metadata.get("phenotype_overlap", 0.0)))
                metadata["measurement_noise"] = max(0.20, float(metadata.get("measurement_noise", 0.0)))
                metadata["partial_observation_rate"] = max(0.24, float(metadata.get("partial_observation_rate", 0.0)))

            challenges.append(replace(agent, metadata=metadata))

    return build_blind_benchmark(
        challenges,
        benchmark_id=benchmark_id or f"cardiagent-{suite}-v{SUITE_VERSIONS[suite]}",
        seed=seed,
        shuffle=True,
    )


def build_all_suites(
    *,
    domains: Sequence[ChallengeDomain] | None = None,
    per_domain: int = 8,
    seed: int = 0,
) -> dict[str, BlindBenchmark]:
    """Build the complete canonical suite collection."""
    return {
        suite: build_suite(suite, domains=domains, per_domain=per_domain, seed=seed)
        for suite in SUITE_VERSIONS
    }
