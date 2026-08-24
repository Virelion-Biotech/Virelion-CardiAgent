from cardiagent.experiment import run_multi_seed_experiment
from cardiagent.suites import baseline_suite


def test_multi_seed_experiment_returns_uncertainty_summary() -> None:
    training = baseline_suite(seed=100).challenges
    result = run_multi_seed_experiment(
        training,
        seeds=(11, 23),
        per_domain=2,
        training_epochs=2,
        severity=0.5,
        difficulty=0.6,
    )
    assert result.version == "0.1"
    assert result.seeds == (11, 23)
    assert result.phenotype_mean_distance.mean >= 0.0
    assert result.phenotype_mean_distance.ci95_half_width >= 0.0
    assert len(result.candidate_diversity_gain.values) == 2


def test_multi_seed_experiment_rejects_empty_seed_set() -> None:
    training = baseline_suite(seed=100).challenges
    try:
        run_multi_seed_experiment(training, seeds=(), per_domain=1, training_epochs=1)
    except ValueError as exc:
        assert "seed" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError")
