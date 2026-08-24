"""End-to-end smoke test for the complete CardiAgent pipeline."""

import json
from pathlib import Path

import pytest

from cardiagent import (
    AgentGeneratorModel,
    ChallengeDomain,
    ChallengeGenerator,
    audit_blind_presentation,
    build_blind_benchmark,
    build_report,
    create_blind_handoff,
    population_metrics,
    run_and_write_experiment,
    summarize_calibration,
)
from cardiagent.experiment_artifact import experiment_fingerprint
from cardiagent.experiment import run_multi_seed_experiment


def _training_population() -> list:
    cases = []
    for seed in range(8):
        generator = ChallengeGenerator(seed=seed)
        for domain in ChallengeDomain:
            cases.append(
                generator.generate(
                    domain,
                    severity=0.2 + 0.1 * (seed % 6),
                    difficulty=0.3 + 0.1 * (seed % 5),
                )
            )
    return cases


def test_full_pipeline_end_to_end(tmp_path: Path) -> None:
    training = _training_population()
    assert len(training) == 8 * len(ChallengeDomain)

    calibration = summarize_calibration(training)
    assert calibration.count == len(training)

    reference_generator = ChallengeGenerator(seed=101)
    reference = [
        reference_generator.generate(
            ChallengeDomain.ISCHEMIC,
            severity=0.6,
            difficulty=0.6,
            agent_id=f"e2e-{index}",
        )
        for index in range(8)
    ]
    report = build_report(reference, suite="e2e", suite_version="0.2", seed=101)
    assert report.metrics["count"] == 8
    assert report.acceptance["nonempty_population"]
    assert report.domain_fidelity == pytest.approx(1.0)

    blind = build_blind_benchmark(reference, benchmark_id="e2e", seed=17)
    assert blind.cases
    assert all(not audit_blind_presentation(case.presentation) for case in blind.cases)
    assert all("domain" not in case.public_dict() for case in blind.cases)

    handoff = create_blind_handoff(reference[0])
    assert handoff.case_id != reference[0].agent_id
    assert not audit_blind_presentation(handoff.presentation)

    torch = pytest.importorskip("torch")
    del torch
    model = AgentGeneratorModel(seed=5).fit(training, epochs=2, batch_size=32)
    generated = model.sample(
        domain=ChallengeDomain.INFLAMMATORY,
        severity=0.6,
        difficulty=0.7,
        count=8,
    )
    assert len(generated) == 8
    assert all(agent.metadata["ml_generated"] for agent in generated)
    assert population_metrics(generated).count == 8

    experiment = run_multi_seed_experiment(
        training,
        seeds=(11, 23),
        per_domain=2,
        training_epochs=2,
        severity=0.6,
        difficulty=0.6,
    )
    assert experiment.seeds == (11, 23)
    assert len(experiment.candidate_diversity_gain.values) == 2
    assert len(experiment_fingerprint(experiment)) == 64

    artifact = tmp_path / "generator-experiment.json"
    run_and_write_experiment(
        training,
        output_path=artifact,
        seeds=(11,),
        per_domain=1,
        training_epochs=1,
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["artifact_version"] == "0.1"
    assert len(payload["fingerprint"]) == 64
