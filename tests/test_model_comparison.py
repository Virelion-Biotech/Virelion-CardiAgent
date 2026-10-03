import pytest

from cardiagent.model_comparison import build_matched_ml_population, build_reference_population, compare_populations
from cardiagent.models import ChallengeDomain


def test_population_comparison_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="Both populations"):
        compare_populations([], [])


def test_reference_population_is_balanced() -> None:
    population = build_reference_population(seed=12, per_domain=2)
    assert len(population) == len(tuple(ChallengeDomain)) * 2
    assert {item.domain for item in population} == set(ChallengeDomain)


def test_ml_population_is_conditioned_and_comparable() -> None:
    torch = pytest.importorskip("torch")
    from cardiagent.ml import AgentGeneratorModel

    training = build_reference_population(seed=20, per_domain=2)
    model = AgentGeneratorModel(seed=5).fit(training, epochs=1, batch_size=16)
    candidate = build_matched_ml_population(model, per_domain=2, severity=0.5, difficulty=0.6)
    reference = build_reference_population(seed=30, per_domain=2, severities=(0.5,), difficulty=0.6)
    comparison = compare_populations(reference, candidate)
    assert comparison.reference_count == comparison.candidate_count
    assert comparison.phenotype_mean_distance >= 0.0
    assert comparison.candidate_diversity_gain == pytest.approx(
        comparison.candidate_diversity - comparison.reference_diversity
    )
    assert all(item.metadata["ml_generated"] for item in candidate)
