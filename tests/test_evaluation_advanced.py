import pytest

from cardiagent import (
    ChallengeDomain,
    ChallengeGenerator,
    conditional_domain_fidelity,
    conditional_numeric_fidelity,
    pairwise_overlap_rate,
    population_metrics,
    quality_score,
)


def _population(seed=10):
    generator = ChallengeGenerator(seed=seed)
    return [
        generator.generate(domain, severity=0.2 + 0.1 * index, difficulty=0.3 + 0.05 * index)
        for index, domain in enumerate(ChallengeDomain)
    ]


def test_population_metrics_are_bounded_and_cover_domains():
    metrics = population_metrics(_population())
    assert metrics.count == len(tuple(ChallengeDomain))
    assert metrics.domain_coverage == 1.0
    assert 0.0 <= metrics.domain_balance <= 1.0
    assert 0.0 <= metrics.duplicate_rate <= 1.0
    assert metrics.phenotype_diversity >= 0.0
    assert 0.0 <= quality_score(metrics) <= 1.0


def test_conditional_domain_fidelity_is_exact_for_suite_cases():
    generator = ChallengeGenerator(seed=22)
    cases = []
    for index, domain in enumerate(ChallengeDomain):
        case = generator.generate(domain, severity=0.5, difficulty=0.5, agent_id=f"case-{index}")
        metadata = dict(case.metadata)
        metadata["requested_domain"] = domain.value
        cases.append(case.__class__(
            agent_id=case.agent_id, domain=case.domain, version=case.version, seed=case.seed,
            severity=case.severity, onset=case.onset, persistence=case.persistence,
            heterogeneity=case.heterogeneity, phenotype=case.phenotype, metadata=metadata,
        ))
    assert conditional_domain_fidelity(cases) == 1.0


def test_numeric_fidelity_requires_targets_and_respects_tolerance():
    generator = ChallengeGenerator(seed=5)
    case = generator.generate(ChallengeDomain.ISCHEMIC, severity=0.7)
    metadata = dict(case.metadata)
    metadata["requested_severity"] = case.severity
    case = case.__class__(
        agent_id=case.agent_id, domain=case.domain, version=case.version, seed=case.seed,
        severity=case.severity, onset=case.onset, persistence=case.persistence,
        heterogeneity=case.heterogeneity, phenotype=case.phenotype, metadata=metadata,
    )
    assert conditional_numeric_fidelity([case], "severity") == 1.0

    with pytest.raises(ValueError):
        conditional_numeric_fidelity([case], "unknown")


def test_pairwise_overlap_rate_handles_singletons():
    case = ChallengeGenerator(seed=1).generate(ChallengeDomain.ISCHEMIC)
    assert pairwise_overlap_rate([case]) == 0.0
