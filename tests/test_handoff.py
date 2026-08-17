from cardiagent import ChallengeDomain, ChallengeGenerator, create_handoff


def test_handoff_is_machine_readable():
    challenge = ChallengeGenerator(seed=7).generate(ChallengeDomain.METABOLIC)
    handoff = create_handoff(challenge)
    payload = handoff.to_dict()

    assert payload["contract_version"] == "0.1"
    assert payload["challenge"]["agent_id"] == challenge.agent_id
    assert "metabolic_disruption" in payload["expected_observables"]


def test_handoff_json_is_deterministic():
    challenge = ChallengeGenerator(seed=7).generate(ChallengeDomain.METABOLIC)
    assert create_handoff(challenge).to_json() == create_handoff(challenge).to_json()
