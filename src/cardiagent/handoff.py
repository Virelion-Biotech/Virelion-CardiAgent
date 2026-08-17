"""Machine-readable CardiAgent -> CardiVex handoff contracts.

The normal handoff preserves full challenge provenance for trusted pipelines.
The blind handoff is intended for detection benchmarks and deliberately omits
challenge domain and other direct ground-truth labels.
"""

from dataclasses import dataclass
from typing import Any
import json

from .benchmark import _presentation
from .models import ChallengeAgent


HANDOFF_VERSION = "0.2"


@dataclass(frozen=True)
class CardiVexHandoff:
    """Stable envelope used to transfer one trusted challenge to CardiVex."""

    contract_version: str
    challenge: ChallengeAgent
    expected_observables: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "challenge": self.challenge.to_dict(),
            "expected_observables": list(self.expected_observables),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass(frozen=True)
class BlindCardiVexHandoff:
    """Detection-only envelope with the challenge identity hidden."""

    contract_version: str
    case_id: str
    presentation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "case_id": self.case_id,
            "presentation": self.presentation,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def create_handoff(challenge: ChallengeAgent) -> CardiVexHandoff:
    """Create the full downstream envelope for trusted evaluation."""
    observables = (
        "stress",
        "inflammation",
        "electrical_instability",
        "contractile_impairment",
        "viability_loss",
        "oxidative_stress",
        "metabolic_disruption",
        "remodeling_signal",
    )
    return CardiVexHandoff(HANDOFF_VERSION, challenge, observables)


def create_blind_handoff(
    challenge: ChallengeAgent,
    *,
    case_id: str | None = None,
) -> BlindCardiVexHandoff:
    """Create a detection benchmark handoff with truth fields withheld."""
    return BlindCardiVexHandoff(
        contract_version=f"{HANDOFF_VERSION}-blind",
        case_id=case_id or challenge.agent_id,
        presentation=_presentation(challenge),
    )
