from pathlib import Path

from cardiagent.experiment_artifact import experiment_fingerprint, run_and_write_experiment
from cardiagent.generator import ChallengeGenerator
from cardiagent.models import ChallengeDomain


def _training_agents():
    output = []
    for seed in range(8):
        generator = ChallengeGenerator(seed=seed)
        for domain in ChallengeDomain:
            output.append(generator.generate(domain, severity=0.2 + 0.1 * (seed % 6), difficulty=0.3 + 0.1 * (seed % 5)))
    return output


def test_fingerprint_is_stable() -> None:
    training = _training_agents()
    path1 = Path('/tmp/cardiagent-experiment-a.json')
    path2 = Path('/tmp/cardiagent-experiment-b.json')
    run_and_write_experiment(training, output_path=path1, seeds=(11, 23), per_domain=2, training_epochs=2)
    run_and_write_experiment(training, output_path=path2, seeds=(11, 23), per_domain=2, training_epochs=2)
    assert path1.read_text() == path2.read_text()


def test_artifact_contains_fingerprint() -> None:
    training = _training_agents()
    destination = Path('/tmp/cardiagent-experiment.json')
    run_and_write_experiment(training, output_path=destination, seeds=(11,), per_domain=1, training_epochs=1)
    payload = destination.read_text(encoding='utf-8')
    assert '"fingerprint"' in payload
    assert len(experiment_fingerprint(__import__('cardiagent').run_multi_seed_experiment(training, seeds=(11,), per_domain=1, training_epochs=1))) == 64
