import json
import random

import pytest

from cardiagent.conditional import ConditionalQuantileGenerator
from cardiagent.generator import ChallengeGenerator
from cardiagent.models import ChallengeDomain


def rows():
    generator = ChallengeGenerator(seed=13)
    return [
        generator.generate(ChallengeDomain.ISCHEMIC, severity=0.5, difficulty=0.5)
        for _ in range(40)
    ]


def test_quantile_samples_preserve_support_variation_and_novel_vectors():
    training = rows()
    model = ConditionalQuantileGenerator(seed=8).fit(training)
    samples = model.sample(domain=ChallengeDomain.ISCHEMIC, severity=0.5, count=100)
    assert all(a.phenotype.inflammation == 0 for a in samples)
    assert len({a.phenotype.stress for a in samples}) > 90
    vectors = {tuple(a.phenotype.to_dict().values()) for a in training}
    assert not any(tuple(a.phenotype.to_dict().values()) in vectors for a in samples)
    assert all(a.metadata["quality_status"] == "not_qualified" for a in samples)


def test_quantile_checkpoint_preserves_private_state(tmp_path):
    random.seed(27)
    state = random.getstate()
    model = ConditionalQuantileGenerator(seed=8).fit(rows())
    first = model.sample(domain=ChallengeDomain.ISCHEMIC, count=3)
    path = tmp_path / "model.json"
    model.save(path)
    expected = model.sample(domain=ChallengeDomain.ISCHEMIC, count=3)
    restored = ConditionalQuantileGenerator.load(path)
    assert [a.to_dict() for a in restored.sample(domain=ChallengeDomain.ISCHEMIC, count=3)] == [
        a.to_dict() for a in expected
    ]
    assert random.getstate() == state
    assert not {a.agent_id for a in first} & {a.agent_id for a in expected}


@pytest.mark.parametrize("severity,difficulty", [(0.7, 0.5), (0.5, 0.7)])
def test_unseen_conditions_fail_closed(severity, difficulty):
    model = ConditionalQuantileGenerator().fit(rows())
    with pytest.raises(ValueError, match="Unseen"):
        model.sample(domain=ChallengeDomain.ISCHEMIC, severity=severity, difficulty=difficulty)


def test_small_stratum_and_duplicate_training_ids_rejected():
    with pytest.raises(ValueError, match="eight"):
        ConditionalQuantileGenerator().fit(rows()[:7])
    with pytest.raises(ValueError, match="unique"):
        ConditionalQuantileGenerator().fit(rows() + rows())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_invalid_sampling_numbers_rejected(value):
    with pytest.raises(ValueError):
        ConditionalQuantileGenerator().fit(rows()).sample(
            domain=ChallengeDomain.ISCHEMIC, severity=value
        )


def test_corrupt_model_support_rejected(tmp_path):
    path = tmp_path / "model.json"
    ConditionalQuantileGenerator().fit(rows()).save(path)
    payload = json.loads(path.read_text())
    payload["groups"][0]["knots"][0][0] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        ConditionalQuantileGenerator.load(path)


def test_quantile_cli_without_neural_training(tmp_path, monkeypatch, capsys):
    from cardiagent.cli import main
    from cardiagent.manifest import build_manifest
    from cardiagent.serialization import write_json
    import sys

    source = tmp_path / "source.json"
    write_json(build_manifest(rows(), manifest_id="test", seed=13).to_dict(), source)
    path = tmp_path / "model.json"
    output = tmp_path / "sample.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cardiagent",
            "train",
            "--model-family",
            "quantile",
            "--input",
            str(source),
            "--output",
            str(path),
        ],
    )
    assert main() == 0
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cardiagent",
            "sample",
            "--model-family",
            "quantile",
            "--model",
            str(path),
            "--domain",
            "ischemic",
            "--count",
            "3",
            "--output",
            str(output),
        ],
    )
    assert main() == 0
    assert output.is_file()
