import pytest

from cardiagent import ChallengeDomain, ChallengeGenerator, conditional_domain_fidelity, phenotype_mae, population_metrics


def _population(seed=1):
    return [
        ChallengeGenerator(seed=seed + index).generate(domain, severity=0.2 + 0.1 * index, difficulty=0.3)
        for index, domain in enumerate(ChallengeDomain)
    ]


def test_population_metrics_cover_all_domains():
    metrics = population_metrics(_population())
    assert metrics.count == len(ChallengeDomain)
    assert metrics.domain_coverage == 1.0
    assert metrics.domain_balance > 0.9
    assert 0.0 <= metrics.duplicate_rate <= 1.0
    assert metrics.phenotype_diversity > 0.0


def test_population_metrics_are_reproducible():
    left = population_metrics(_population(5)).to_dict()
    right = population_metrics(_population(5)).to_dict()
    assert left == right


def test_phenotype_mae_is_zero_for_aligned_identical_population():
    population = _population()
    assert phenotype_mae(population, population) == pytest.approx(0.0)


def test_phenotype_mae_requires_alignment():
    population = _population()
    with pytest.raises(ValueError):
        phenotype_mae(population, population[:-1])


def test_conditional_domain_fidelity_uses_requested_domain_metadata():
    population = _population()
    assert conditional_domain_fidelity(population) == pytest.approx(1.0)
