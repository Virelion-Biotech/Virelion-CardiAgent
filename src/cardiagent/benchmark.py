"""Blinded benchmark utilities for CardiAgent -> CardiVex evaluation.

The public case deliberately omits challenge identity and other direct truth
signals. Ground truth remains in a separate evaluation record so CardiVex can
make an independent detection/characterization call.
"""

from dataclasses import dataclass
import hashlib
import json
import random
import re
from typing import Any, Iterable

from .models import ChallengeAgent


BENCHMARK_VERSION = "0.2"
_OPAQUE_ID_PATTERN = re.compile(r"^CA-(?:ischemic|inflammatory|electrophysiologic|toxic_injury|viral_like|metabolic|genetic_susceptibility)-")


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


def _opaque_case_id(challenge: ChallengeAgent) -> str:
    """Create a stable identifier that does not encode the challenge domain."""
    digest = hashlib.sha256(challenge.to_json().encode("utf-8")).hexdigest()[:16]
    return f"case-{digest}"


def _presentation(challenge: ChallengeAgent) -> dict[str, Any]:
    metadata = challenge.metadata
    # Deliberately exclude challenge.agent_id. Domain-coded generator IDs are
    # a trivial label-leakage channel and must never enter a blind presentation.
    return {
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


def audit_blind_presentation(presentation: dict[str, Any]) -> tuple[str, ...]:
    """Detect common direct label-leakage channels in a public presentation.

    This is intentionally conservative: it flags keys or values that directly
    expose a generator/domain identity. A clean result does not prove absence
    of all statistical leakage, but it catches accidental schema leakage.
    """
    violations: list[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                lowered = str(key).lower()
                if lowered in {"domain", "challenge_id", "agent_id", "ground_truth", "scenario_family", "overlap_reference"}:
                    violations.append(f"forbidden field: {path}.{key}")
                walk(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, str) and _OPAQUE_ID_PATTERN.search(value):
            violations.append(f"domain-coded identifier: {path}")

    walk(presentation, "presentation")
    return tuple(violations)


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
        presentation = _presentation(challenge)
        violations = audit_blind_presentation(presentation)
        if violations:
            raise ValueError(f"Blind presentation failed leakage audit: {violations}")
        truth = {
            "challenge_id": challenge.agent_id,
            "domain": challenge.domain.value,
            "severity": challenge.severity,
            "scenario_family": challenge.metadata.get("scenario_family"),
            "difficulty": challenge.metadata.get("difficulty"),
            "overlap_reference": challenge.metadata.get("overlap_reference"),
        }
        cases.append(BlindCase(case_id, presentation, truth))

    return BlindBenchmark(
        benchmark_id=benchmark_id,
        seed=seed,
        version=BENCHMARK_VERSION,
        cases=tuple(cases),
    )


def opaque_case_id(challenge: ChallengeAgent) -> str:
    """Return the stable opaque ID used by blind handoffs."""
    return _opaque_case_id(challenge)
