"""Core data models.

The models intentionally represent host-observable challenge properties rather
than operational biological construction instructions.
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any
import json


class ChallengeDomain(str, Enum):
    ISCHEMIC = "ischemic"
    INFLAMMATORY = "inflammatory"
    ELECTROPHYSIOLOGIC = "electrophysiologic"
    TOXIC_INJURY = "toxic_injury"
    VIRAL_LIKE = "viral_like"
    METABOLIC = "metabolic"
    GENETIC_SUSCEPTIBILITY = "genetic_susceptibility"


@dataclass(frozen=True)
class PhenotypeProfile:
    """Normalized phenotype-level challenge features.

    Values are abstract intensities in [0, 1]. They are not concentrations,
    doses, sequences, growth conditions, or other operational parameters.
    """

    stress: float = 0.0
    inflammation: float = 0.0
    electrical_instability: float = 0.0
    contractile_impairment: float = 0.0
    viability_loss: float = 0.0
    oxidative_stress: float = 0.0
    metabolic_disruption: float = 0.0
    remodeling_signal: float = 0.0

    def __post_init__(self) -> None:
        values = asdict(self)
        invalid = {k: v for k, v in values.items() if not 0.0 <= v <= 1.0}
        if invalid:
            raise ValueError(f"Phenotype intensities must be within [0, 1]: {invalid}")


@dataclass(frozen=True)
class ChallengeAgent:
    """Serializable challenge instance passed downstream to CardiVex."""

    agent_id: str
    domain: ChallengeDomain
    version: str
    seed: int
    severity: float
    onset: float
    persistence: float
    heterogeneity: float
    phenotype: PhenotypeProfile
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        for name in ("severity", "onset", "persistence", "heterogeneity"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value}")
        if not self.agent_id:
            raise ValueError("agent_id cannot be empty")
        if not self.version:
            raise ValueError("version cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
