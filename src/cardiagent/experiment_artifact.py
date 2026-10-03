"""Persist multi-seed generator experiments as auditable artifacts."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from .experiment import MultiSeedExperiment, run_multi_seed_experiment
from .models import ChallengeAgent


ARTIFACT_VERSION = "0.1"


def experiment_fingerprint(experiment: MultiSeedExperiment) -> str:
    """Return a stable fingerprint for experiment configuration and results."""
    payload = json.dumps(asdict(experiment), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_and_write_experiment(
    training_agents: list[ChallengeAgent],
    *,
    output_path: str | Path,
    seeds: tuple[int, ...] = (11, 23, 37, 41, 53),
    per_domain: int = 16,
    training_epochs: int = 25,
    severity: float = 0.5,
    difficulty: float = 0.6,
) -> Path:
    """Run a multi-seed comparison and write one versioned JSON artifact."""
    experiment = run_multi_seed_experiment(
        training_agents,
        seeds=seeds,
        per_domain=per_domain,
        training_epochs=training_epochs,
        severity=severity,
        difficulty=difficulty,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "experiment": asdict(experiment),
        "fingerprint": experiment_fingerprint(experiment),
    }
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
