"""Challenge-set manifests for reproducible CardiAgent batches."""

from dataclasses import dataclass
from typing import Any, Iterable

from .models import ChallengeAgent


@dataclass(frozen=True)
class ChallengeManifest:
    """A deterministic collection of challenge instances."""

    manifest_id: str
    generator_version: str
    seed: int
    challenges: tuple[ChallengeAgent, ...]

    def __post_init__(self) -> None:
        if not self.manifest_id:
            raise ValueError("manifest_id cannot be empty")
        if len({c.agent_id for c in self.challenges}) != len(self.challenges):
            raise ValueError("challenge agent_id values must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "challenge_count": len(self.challenges),
            "challenges": [challenge.to_dict() for challenge in self.challenges],
        }

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def build_manifest(
    challenges: Iterable[ChallengeAgent],
    *,
    manifest_id: str,
    seed: int,
    generator_version: str = "0.1",
) -> ChallengeManifest:
    """Freeze an iterable into a validated reproducible manifest."""
    return ChallengeManifest(
        manifest_id=manifest_id,
        generator_version=generator_version,
        seed=seed,
        challenges=tuple(challenges),
    )
