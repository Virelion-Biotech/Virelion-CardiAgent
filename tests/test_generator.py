from cardiagent import ChallengeDomain, ChallengeGenerator


def test_generation_is_reproducible():
    a = ChallengeGenerator(seed=42).generate(ChallengeDomain.ISCHEMIC, severity=0.7)
    b = ChallengeGenerator(seed=42).generate(ChallengeDomain.ISCHEMIC, severity=0.7)
    assert a.to_dict() == b.to_dict()


def test_output_is_phenotype_level():
    agent = ChallengeGenerator(seed=1).generate(ChallengeDomain.VIRAL_LIKE)
    assert agent.metadata["representation"] == "phenotype-level"
    assert 0.0 <= agent.phenotype.inflammation <= 1.0
    assert 0.0 <= agent.phenotype.viability_loss <= 1.0
