from cardiagent import ChallengeDomain, ChallengeGenerator


def test_repeated_generation_produces_distinct_ids():
    generator = ChallengeGenerator(seed=42)
    first = generator.generate(ChallengeDomain.ISCHEMIC)
    second = generator.generate(ChallengeDomain.ISCHEMIC)

    assert first.agent_id != second.agent_id
    assert first.to_dict() != second.to_dict()


def test_same_seed_reproduces_full_sequence():
    a = ChallengeGenerator(seed=42)
    b = ChallengeGenerator(seed=42)

    sequence_a = [a.generate(ChallengeDomain.ISCHEMIC).to_dict() for _ in range(3)]
    sequence_b = [b.generate(ChallengeDomain.ISCHEMIC).to_dict() for _ in range(3)]

    assert sequence_a == sequence_b
