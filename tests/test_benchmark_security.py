from cardiagent import ChallengeDomain, ChallengeGenerator, audit_blind_presentation, create_blind_handoff
from cardiagent.benchmark import build_blind_benchmark


def test_blind_presentation_has_no_domain_coded_identifier():
    challenge = ChallengeGenerator(seed=11).generate(ChallengeDomain.INFLAMMATORY, severity=0.7)
    handoff = create_blind_handoff(challenge)

    assert "challenge_id" not in handoff.presentation
    assert "domain" not in handoff.presentation
    assert audit_blind_presentation(handoff.presentation) == ()
    assert "inflammatory" not in handoff.case_id
    assert "inflammatory" not in handoff.to_json()


def test_blind_benchmark_keeps_truth_separate():
    challenges = [
        ChallengeGenerator(seed=20 + index).generate(domain, severity=0.5)
        for index, domain in enumerate((ChallengeDomain.ISCHEMIC, ChallengeDomain.INFLAMMATORY))
    ]
    benchmark = build_blind_benchmark(challenges, benchmark_id="security", seed=9)

    public = benchmark.public_dict()
    evaluation = benchmark.evaluation_dict()
    assert len(public["cases"]) == 2
    assert all("ground_truth" not in case for case in public["cases"])
    assert all("domain" not in case["presentation"] for case in public["cases"])
    assert all("domain" in case["ground_truth"] for case in evaluation["cases"])
