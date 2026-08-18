from cardiagent import (
    ChallengeDomain,
    ChallengeGenerator,
    DetectionOutcome,
    conditional_fidelity,
    detection_metrics,
    population_metrics,
    reproducibility_signature,
)


def test_detection_metrics_are_deterministic_and_bounded():
    outcomes = [
        DetectionOutcome("a", "ischemic", confidence=0.9, detected=True, characterization_correct=True),
        DetectionOutcome("b", "metabolic", confidence=0.4, detected=False, characterization_correct=False),
        DetectionOutcome("c", "ischemic", confidence=0.8, detected=True, characterization_correct=False),
    ]
    metrics = detection_metrics(outcomes, {"a": "ischemic", "b": "metabolic", "c": "ischemic"})

    assert metrics.count == 3
    assert 0.0 <= metrics.detected_rate <= 1.0
    assert 0.0 <= metrics.characterization_accuracy <= 1.0
    assert 0.0 <= metrics.domain_accuracy <= 1.0
    assert 0.0 <= metrics.macro_f1 <= 1.0
    assert 0.0 <= metrics.mean_hardness <= 1.0


def test_population_metrics_report_local_novelty():
    generator = ChallengeGenerator(seed=4)
    population = [
        generator.generate(ChallengeDomain.ISCHEMIC, severity=0.2, difficulty=0.2),
        generator.generate(ChallengeDomain.ISCHEMIC, severity=0.6, difficulty=0.6),
        generator.generate(ChallengeDomain.METABOLIC, severity=0.8, difficulty=0.8),
    ]
    metrics = population_metrics(population)

    assert metrics.count == 3
    assert len(metrics.mean_vector) == 11
    assert len(metrics.std_vector) == 11
    assert metrics.min_nearest_neighbor_distance >= 0.0
    assert metrics.mean_nearest_neighbor_distance >= metrics.min_nearest_neighbor_distance


def test_conditional_fidelity_rewards_matching_condition():
    generator = ChallengeGenerator(seed=2)
    population = [
        generator.generate(ChallengeDomain.INFLAMMATORY, severity=0.8, difficulty=0.7)
        for _ in range(4)
    ]
    score = conditional_fidelity(
        population,
        domain=ChallengeDomain.INFLAMMATORY,
        severity=0.8,
        tolerance=0.20,
    )
    assert score > 0.9


def test_reproducibility_signature_matches_for_same_seed():
    a = [ChallengeGenerator(seed=10).generate(ChallengeDomain.ISCHEMIC, severity=0.7, difficulty=0.5) for _ in range(2)]
    b = [ChallengeGenerator(seed=10).generate(ChallengeDomain.ISCHEMIC, severity=0.7, difficulty=0.5) for _ in range(2)]
    assert reproducibility_signature(a) == reproducibility_signature(b)
