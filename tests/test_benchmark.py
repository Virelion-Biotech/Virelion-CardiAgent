from cardiagent import (
    ChallengeDomain,
    ChallengeGenerator,
    build_blind_benchmark,
    create_blind_handoff,
)


def test_detailed_agent_contains_temporal_and_confounded_scenario():
    agent = ChallengeGenerator(seed=7).generate(
        ChallengeDomain.INFLAMMATORY,
        severity=0.6,
        difficulty=0.9,
    )
    metadata = agent.metadata
    assert metadata["scenario_version"] == "1"
    assert len(metadata["temporal_profile"]) == 5
    assert metadata["temporal_profile"][0]["phase"] == "baseline"
    assert metadata["temporal_profile"][-1]["phase"] == "recovery"
    assert "phenotype_overlap" in metadata
    assert "measurement_noise" in metadata
    assert "confounders" in metadata


def test_blind_benchmark_hides_domain():
    agent = ChallengeGenerator(seed=9).generate(ChallengeDomain.ISCHEMIC, difficulty=0.8)
    benchmark = build_blind_benchmark([agent], benchmark_id="test", seed=9)
    public = benchmark.public_dict()
    private = benchmark.evaluation_dict()

    assert "domain" not in public["cases"][0]["presentation"]
    assert private["cases"][0]["ground_truth"]["domain"] == "ischemic"


def test_blind_handoff_hides_direct_truth_label():
    agent = ChallengeGenerator(seed=11).generate(ChallengeDomain.METABOLIC)
    handoff = create_blind_handoff(agent, case_id="case-1")
    payload = handoff.to_dict()
    assert payload["case_id"] == "case-1"
    assert "domain" not in payload["presentation"]
