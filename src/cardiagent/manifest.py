"""Challenge-set manifests for reproducible CardiAgent batches."""

from dataclasses import dataclass
from typing import Any, Iterable

from .models import ChallengeAgent
from .serialization import strict_json


@dataclass(frozen=True)
class ChallengeManifest:
    """A deterministic collection of challenge instances."""

    manifest_id: str
    generator_version: str
    seed: int
    challenges: tuple[ChallengeAgent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest_id, str) or not self.manifest_id.strip():
            raise ValueError("manifest_id cannot be empty")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(self.generator_version, str) or not self.generator_version.strip():
            raise ValueError("generator_version cannot be empty")
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

        return strict_json(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError("Manifest must be an object")
        expected = {"manifest_id", "generator_version", "seed", "challenge_count", "challenges"}
        if set(payload) != expected:
            raise ValueError("Invalid manifest fields")
        challenges = tuple(ChallengeAgent.from_dict(item) for item in payload["challenges"])
        if payload["challenge_count"] != len(challenges):
            raise ValueError("Manifest challenge_count mismatch")
        return cls(
            payload["manifest_id"], payload["generator_version"], payload["seed"], challenges
        )


def build_manifest(
    challenges: Iterable[ChallengeAgent],
    *,
    manifest_id: str,
    seed: int,
    generator_version: str | None = None,
) -> ChallengeManifest:
    """Freeze an iterable into a validated reproducible manifest."""
    items = tuple(challenges)
    versions = {item.version for item in items}
    return ChallengeManifest(
        manifest_id=manifest_id,
        generator_version=generator_version
        or (next(iter(versions)) if len(versions) == 1 else "mixed"),
        seed=seed,
        challenges=items,
    )
