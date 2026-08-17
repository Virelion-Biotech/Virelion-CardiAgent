import pytest

from cardiagent import ChallengeDomain, ChallengeGenerator, build_manifest


def test_manifest_preserves_challenge_order_and_count():
    generator = ChallengeGenerator(seed=11)
    challenges = [
        generator.generate(ChallengeDomain.ISCHEMIC, agent_id="CA-001"),
        generator.generate(ChallengeDomain.METABOLIC, agent_id="CA-002"),
    ]
    manifest = build_manifest(challenges, manifest_id="set-001", seed=11)

    assert manifest.to_dict()["challenge_count"] == 2
    assert [c["agent_id"] for c in manifest.to_dict()["challenges"]] == ["CA-001", "CA-002"]


def test_manifest_rejects_duplicate_ids():
    generator = ChallengeGenerator(seed=11)
    a = generator.generate(ChallengeDomain.ISCHEMIC, agent_id="duplicate")
    b = generator.generate(ChallengeDomain.METABOLIC, agent_id="duplicate")

    with pytest.raises(ValueError, match="unique"):
        build_manifest([a, b], manifest_id="bad", seed=11)
