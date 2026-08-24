import json
from pathlib import Path

from cardiagent.experiment_artifact import run_and_write_experiment
from cardiagent.generator import ChallengeGenerator
from cardiagent.models import ChallengeDomain


def _training_agents():
    output = []
    for seed in range(8):
        generator = ChallengeGenerator(seed=seed)
        for domain in ChallengeDomain:
            output.append(
                generator.generate(
                    domain,
                    severity=0.2 + 0.1 * (seed % 6),
                    difficulty=0.3 + 0.1 * (seed % 5),
                )
            )
    return output


def test_artifact_is_reproducible(tmp_path: Path) -> None:
    training = _training_agents()
    path1 = tmp_path / "a.json"
    path2 = tmp_path / "b.json"
    run_and_write_experiment(training, output_path=path1, seeds=(11, 23), per_domain=2, training_epochs=2)
    run_and_write_experiment(training, output_path=path2, seeds=(11, 23), per_domain=2, training_epochs=2)
    assert path1.read_text(encoding="utf-8") == path2.read_text(encoding="utf-8")


def test_artifact_contains_valid_fingerprint(tmp_path: Path) -> None:
    destination = tmp_path / "experiment.json"
    run_and_write_experiment(_training_agents(), output_path=destination, seeds=(11,), per_domain=1, training_epochs=1)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["artifact_version"] == "0.1"
    assert len(payload["fingerprint"]) == 64
    assert len(payload["experiment"]["seeds"]) == 1
