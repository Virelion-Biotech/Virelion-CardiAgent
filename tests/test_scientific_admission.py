from dataclasses import replace
import pytest
from cardiagent.generator import ChallengeGenerator
from cardiagent.models import ChallengeDomain
from cardiagent.handoff import create_handoff, create_blind_handoff
from cardiagent.benchmark import build_blind_benchmark


def test_cvae_cannot_enter_full_or_blind_benchmark_handoff():
    item = ChallengeGenerator(seed=0).generate(ChallengeDomain.ISCHEMIC)
    item = replace(
        item,
        metadata={
            **item.metadata,
            "model_family": "conditional_variational_autoencoder",
            "quality_status": "qualified",
        },
    )
    for fn in [create_handoff, create_blind_handoff]:
        with pytest.raises(ValueError, match="CVAE"):
            fn(item)
    with pytest.raises(ValueError, match="CVAE"):
        build_blind_benchmark([item], benchmark_id="locked-test")


def test_corruptions_act_on_values_and_preserve_truth():
    np = pytest.importorskip("numpy")
    from cardiagent.corruptions import corrupt_observations

    x = np.arange(12, dtype=float).reshape(4, 3)
    result = corrupt_observations(x, kind="masking", fraction=0.5, seed=7)
    assert result["changed_mask"].sum() == 6
    assert np.isnan(result["observed"]).sum() == 6
    assert np.array_equal(result["truth"], x)
    assert not result["truth"].flags.writeable
    censored = corrupt_observations(x, kind="left_censoring", threshold=4)
    assert np.all(censored["observed"][:1] == 4)
    shifted = corrupt_observations(x, kind="batch_offset", offset=[0, 2, 0])
    assert np.array_equal(shifted["observed"][:, 1], x[:, 1] + 2)
