"""Blinded benchmark utilities for CardiAgent -> CardiVex evaluation.

The public case deliberately omits the challenge domain and other direct truth
signals. Ground truth remains in a separate evaluation record so CardiVex can
make an independent detection/characterization call.
"""

from dataclasses import dataclass
import json
import random
from typing import Any, Iterable

from .models import ChallengeAgent


BENCHMARK_VERSION = "0.1"


@dataclass(frozen=True)
class BlindCase:
    """One challenge presentation plus a separate hidden evaluation label."""

    case_id: str
    presentation: dict[str, Any]
    ground_truth: dict[str, Any]

    def public_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "presentation": self.presentation}

    def evaluation_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "ground_truth": self.ground_truth}


@dataclass(frozen=True)
class BlindBenchmark:
    """A reproducible collection of blinded cases."""

    benchmark_id: str
    seed: int
    version: str
    cases: tuple[BlindCase, ...]

    def public_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "seed": self.seed,
            "version": self.version,
            "case_count": len(self.cases),
            "cases": [case.public_dict() for case in self.cases],
        }

    def evaluation_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "seed": self.seed,
            "version": self.version,
            "case_count": len(self.cases),
            "cases": [case.evaluation_dict() for case in self.cases],
        }

    def public_json(self) -> str:
        return json.dumps(self.public_dict(), indent=2, sort_keys=True)

    def evaluation_json(self) -> str:
        return json.dumps(self.evaluation_dict(), indent=2, sort_keys=True)


def _presentation(challenge: ChallengeAgent) -> dict[str, Any]:
    metadata = challenge.metadata
    return {
        "challenge_id": challenge.agent_id,
        "representation": "phenotype-level",
        "version": challenge.version,
        "severity_band": "low" if challenge.severity < 0.34 else "moderate" if challenge.severity < 0.67 else "high",
        "temporal": {
            "onset": challenge.onset,
            "persistence": challenge.persistence,
            "trajectory": metadata.get("temporal_profile", []),
        },
        "cell_context": metadata.get("cell_context", []),
        "phenotype": challenge.phenotype.to_dict() if hasattr(challenge.phenotype, "to_dict") else challenge.phenotype.__dict__,
        "observation_characteristics": {
            "measurement_noise": metadata.get("measurement_noise", 0.0),
            "partial_observation_rate": metadata.get("partial_observation_rate", 0.0),
            "heterogeneity": challenge.heterogeneity,
        },
        "confounders": metadata.get("confounders", []),
    }


def build_blind_benchmark(
    challenges: Iterable[ChallengeAgent],
    *,
    benchmark_id: str,
    seed: int = 0,
    shuffle: bool = True,
) -> BlindBenchmark:
    """Build a public challenge set while keeping truth in a separate record.

    The function never changes the underlying challenge objects. Shuffling is
    deterministic and only affects case order, making benchmark runs auditable.
    """
    items = list(challenges)
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(items)

    cases: list[BlindCase] = []
    for index, challenge in enumerate(items, start=1):
        case_id = f"{benchmark_id}-{index:04d}"
        truth = {
            "challenge_id": challenge.agent_id,
            "domain": challenge.domain.value,
            "severity": challenge.severity,
            "scenario_family": challenge.metadata.get("scenario_family"),
            "difficulty": challenge.metadata.get("difficulty"),
            "overlap_reference": challenge.metadata.get("overlap_reference"),
        }
        cases.append(BlindCase(case_id, _presentation(challenge), truth))

    return BlindBenchmark(
        benchmark_id=benchmark_id,
        seed=seed,
        version=BENCHMARK_VERSION,
        cases=tuple(cases),
    )
