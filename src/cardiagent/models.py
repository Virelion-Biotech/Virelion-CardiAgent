"""Core data models.

The models intentionally represent host-observable challenge properties rather
than operational biological construction instructions.
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any
import json
from copy import deepcopy
from .serialization import finite_number, strict_json


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
        for name, value in values.items():
            finite_number(value, name, low=0, high=1)
        invalid = {k: v for k, v in values.items() if not 0.0 <= v <= 1.0}
        if invalid:
            raise ValueError(f"Phenotype intensities must be within [0, 1]: {invalid}")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


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
        if not isinstance(self.domain, ChallengeDomain) or not isinstance(
            self.phenotype, PhenotypeProfile
        ):
            raise ValueError("domain and phenotype must be typed models")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
        strict_json(self.metadata)
        for key in (
            "difficulty",
            "requested_severity",
            "requested_difficulty",
            "measurement_noise",
            "partial_observation_rate",
            "phenotype_overlap",
        ):
            if key in self.metadata:
                finite_number(self.metadata[key], key, low=0, high=1)
        object.__setattr__(self, "metadata", deepcopy(self.metadata))
        for name in ("severity", "onset", "persistence", "heterogeneity"):
            finite_number(getattr(self, name), name, low=0, high=1)
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value}")
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise ValueError("agent_id cannot be empty")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return strict_json(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError("Challenge must be a JSON object")
        values = dict(payload)
        try:
            values["domain"] = ChallengeDomain(values["domain"])
            values["phenotype"] = PhenotypeProfile(**values["phenotype"])
            return cls(**values)
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid challenge schema") from exc

    @classmethod
    def from_json(cls, text):
        def invalid(value):
            raise ValueError(f"Nonstandard JSON number: {value}")

        return cls.from_dict(json.loads(text, parse_constant=invalid))
