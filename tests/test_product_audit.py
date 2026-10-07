from dataclasses import replace

import pytest

from cardiagent import (
    ChallengeDomain,
    ChallengeGenerator,
    build_blind_benchmark,
    create_blind_handoff,
)
from cardiagent.adaptive import AdaptiveChallengeEngine, DetectionOutcome
from cardiagent.benchmark import opaque_case_id
from cardiagent.evaluation import pairwise_overlap_rate
from cardiagent.models import ChallengeAgent


def agent():
    return ChallengeGenerator(seed=3).generate(ChallengeDomain.ISCHEMIC)


def test_blind_handoff_audits_metadata_leakage():
    a = agent()
    a.metadata["temporal_profile"] = [{"domain": "ischemic"}]
    with pytest.raises(ValueError, match="leakage"):
        create_blind_handoff(a)


def test_duplicate_benchmark_truth_ids_rejected():
    a = agent()
    with pytest.raises(ValueError, match="unique"):
        build_blind_benchmark([a, a], benchmark_id="x")


def test_adaptive_ranking_resolves_blind_outcome_ids():
    a = agent()
    b = ChallengeGenerator(seed=4).generate(ChallengeDomain.METABOLIC)
    engine = AdaptiveChallengeEngine()
    engine.score([DetectionOutcome(opaque_case_id(b), None, detected=False)])
    assert engine.hard_cases([a, b], top_k=1) == [b]


def test_invalid_metadata_numeric_value_rejected():
    a = agent()
    with pytest.raises(ValueError):
        replace(a, metadata={**a.metadata, "difficulty": float("nan")})


def test_json_roundtrip_has_valid_domain_and_phenotype():
    a = agent()
    restored = ChallengeAgent.from_dict(a.to_dict())
    assert restored == a


def test_nonfinite_threshold_rejected():
    with pytest.raises(ValueError):
        pairwise_overlap_rate([agent()], threshold=float("nan"))
