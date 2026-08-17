"""Machine-readable CardiAgent -> CardiVex handoff contract.

The contract carries only abstract challenge metadata and host-observable
phenotype expectations. It deliberately excludes operational biological
construction parameters.
"""

from dataclasses import dataclass
from typing import Any

from .models import ChallengeAgent


HANDOFF_VERSION = "0.1"


@dataclass(frozen=True)
class CardiVexHandoff:
    """Stable envelope used to transfer one challenge to CardiVex."""

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
        import json

        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def create_handoff(challenge: ChallengeAgent) -> CardiVexHandoff:
    """Create the minimal downstream envelope for CardiVex."""
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
