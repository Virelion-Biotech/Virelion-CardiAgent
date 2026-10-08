import json
import random

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
from cardiagent.joint import ConditionalCopulaGenerator  # noqa: E402
from cardiagent.models import ChallengeAgent, ChallengeDomain, PhenotypeProfile  # noqa: E402
from cardiagent.ml import PHENOTYPE_FIELDS  # noqa: E402


def rows(conditions=((0.2, 0.2), (0.2, 0.8), (0.8, 0.2), (0.8, 0.8)), count=128):
    rng = np.random.default_rng(72)
    result = []
    for severity, difficulty in conditions:
        for i in range(count):
            u = rng.random()
            v = 0.9 * u + 0.1 * rng.random()
            vector = [
                0.1 + 0.2 * severity + 0.1 * difficulty + 0.2 * u,
                0.1 + 0.2 * severity + 0.1 * difficulty + 0.2 * v,
                0,
                0.2 + 0.1 * u,
                0.3 - 0.1 * u,
                0.2,
                0.1,
                0.2,
            ]
            result.append(
                ChallengeAgent(
                    agent_id=f"{severity}-{difficulty}-{i}",
                    domain=ChallengeDomain.ISCHEMIC,
                    version="fixture",
                    seed=72,
                    severity=severity,
                    onset=u,
                    persistence=v,
                    heterogeneity=0.5,
                    phenotype=PhenotypeProfile(**dict(zip(PHENOTYPE_FIELDS, vector))),
                    metadata={"requested_difficulty": difficulty},
                )
            )
    return result


def test_dependence_constants_interpolation_and_private_rng():
    from scipy.stats import spearmanr

    random.seed(5)
    state = random.getstate()
    training = rows()
    model = ConditionalCopulaGenerator(seed=9).fit(training)
    samples = model.sample(domain="ischemic", severity=0.5, difficulty=0.5, count=1000)
    assert (
        spearmanr(
            [a.phenotype.stress for a in samples], [a.phenotype.inflammation for a in samples]
        ).statistic
        > 0.9
    )
    assert all(a.phenotype.electrical_instability == 0 and a.heterogeneity == 0.5 for a in samples)
    assert all(
        a.metadata["condition_interpolated"] and a.metadata["quality_status"] == "not_qualified"
        for a in samples
    )
    assert not any(a.metadata["patient_validated"] for a in samples)
    assert random.getstate() == state
    weights = samples[0].metadata["support_weights"]
    assert sum(v["weight"] for v in weights) == pytest.approx(1)
    assert 0.33 < np.mean([a.phenotype.stress for a in samples]) < 0.37


def test_checkpoint_continuation(tmp_path):
    model = ConditionalCopulaGenerator(seed=19).fit(rows())
    model.sample(domain="ischemic", severity=0.5, count=3)
    path = tmp_path / "model.json"
    model.save(path)
    expected = model.sample(domain="ischemic", severity=0.5, count=5)
    restored = ConditionalCopulaGenerator.load(path)
    actual = restored.sample(domain="ischemic", severity=0.5, count=5)
    assert [a.to_dict() for a in expected] == [a.to_dict() for a in actual]


@pytest.mark.parametrize(
    "conditions,query,allowed",
    [
        (((0.2, 0.2),), (0.2, 0.2), True),
        (((0.2, 0.2),), (0.3, 0.3), False),
        (((0.2, 0.2), (0.8, 0.8)), (0.5, 0.5), True),
        (((0.2, 0.2), (0.8, 0.8)), (0.5, 0.4), False),
        (((0.2, 0.2), (0.8, 0.8)), (0.9, 0.9), False),
        (((0.2, 0.2), (0.2, 0.8), (0.8, 0.2)), (0.7, 0.7), False),
        (((0.2, 0.2), (0.2, 0.8), (0.8, 0.2)), (0.3, 0.4), True),
    ],
)
def test_supported_condition_hulls(conditions, query, allowed):
    model = ConditionalCopulaGenerator().fit(rows(conditions))
    if allowed:
        assert model.sample(domain="ischemic", severity=query[0], difficulty=query[1])
    else:
        with pytest.raises(ValueError):
            model.sample(domain="ischemic", severity=query[0], difficulty=query[1])


@pytest.mark.parametrize(
    "changes",
    [
        {"domain": "metabolic"},
        {"severity": True},
        {"difficulty": float("nan")},
        {"count": 0},
        {"count": 100001},
        {"agent_id_prefix": ""},
        {"severity": 0.9},
    ],
)
def test_invalid_queries(changes):
    args = {"domain": "ischemic", "severity": 0.5}
    args.update(changes)
    with pytest.raises(ValueError):
        ConditionalCopulaGenerator().fit(rows()).sample(**args)


def test_fit_rejects_bad_training():
    with pytest.raises(ValueError):
        ConditionalCopulaGenerator().fit(rows(count=31))
    with pytest.raises(ValueError):
        ConditionalCopulaGenerator().fit(rows() + rows())
    with pytest.raises(RuntimeError):
        ConditionalCopulaGenerator().save("/tmp/untrained-copula.json")
    with pytest.raises(ValueError):
        ConditionalCopulaGenerator(seed=True)
    with pytest.raises(ValueError):
        ConditionalCopulaGenerator(shrinkage=0)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(version="wrong"),
        lambda p: p.update(extra=1),
        lambda p: p.update(groups=[]),
        lambda p: p["groups"][0]["knots"][0].__setitem__(0, 2),
        lambda p: p["groups"][0]["correlation"][0].__setitem__(1, 1),
        lambda p: p["groups"][0]["correlation"][0].__setitem__(0, 0),
        lambda p: p["groups"][0].update(key=["ischemic", True, 0.2]),
        lambda p: p["groups"].append(p["groups"][0]),
        lambda p: p["training_summary"].update(patient_validated=True),
        lambda p: p.update(counter=-1),
        lambda p: p["groups"][0].update(correlation=[]),
    ],
)
def test_invalid_checkpoints(tmp_path, change):
    path = tmp_path / "model.json"
    ConditionalCopulaGenerator().fit(rows()).save(path)
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        ConditionalCopulaGenerator.load(path)


def test_copula_cli(tmp_path):
    from cardiagent.cli import main
    from cardiagent.manifest import build_manifest
    from cardiagent.serialization import write_json

    source = tmp_path / "source.json"
    model = tmp_path / "model.json"
    sample = tmp_path / "sample.json"
    write_json(build_manifest(rows(), manifest_id="joint-fixture", seed=72).to_dict(), source)
    assert (
        main(["train", "--model-family", "copula", "--input", str(source), "--output", str(model)])
        == 0
    )
    assert (
        main(
            [
                "sample",
                "--model-family",
                "copula",
                "--model",
                str(model),
                "--domain",
                "ischemic",
                "--severity",
                ".5",
                "--count",
                "5",
                "--output",
                str(sample),
            ]
        )
        == 0
    )
    assert sample.exists()


def test_checkpoint_duplicate_json_fields_fail(tmp_path):
    from cardiagent.serialization import read_json

    path = tmp_path / "corrupt.json"
    path.write_text('{"seed":1,"seed":2}')
    with pytest.raises(ValueError, match="Duplicate"):
        read_json(path)


def test_optional_joint_dependency_error(monkeypatch):
    import cardiagent.cli as cli

    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="joint"):
        cli._joint_model_class()
