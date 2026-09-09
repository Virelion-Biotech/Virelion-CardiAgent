"""HeartTwin local-command adapter for CardiAgent challenge generation."""
from __future__ import annotations

import json
import os
import sys

from .generator import ChallengeGenerator
from .models import ChallengeDomain


def _params(payload: dict) -> dict:
    for obs in payload.get("observations", []):
        if obs.get("modality") == "structural" and isinstance(obs.get("values"), dict):
            return obs["values"]
    return payload.get("context", {}).get("cardiagent", {}) or {}


def main() -> int:
    raw = os.environ.get("HEARTTWIN_PAYLOAD")
    if not raw:
        print("HEARTTWIN_PAYLOAD environment variable not set", file=sys.stderr)
        return 1
    try:
        payload = json.loads(raw)
        params = _params(payload)
        domain = ChallengeDomain(str(params.get("domain", "ischemic")))
        seed = int(params.get("seed", 0))
        count = int(params.get("count", 1))
        if count < 1:
            raise ValueError("count must be >= 1")
        generator = ChallengeGenerator(seed=seed)
        challenges = [
            generator.generate(
                domain,
                severity=float(params.get("severity", 0.5)),
                difficulty=(float(params["difficulty"]) if params.get("difficulty") is not None else None),
            ).to_dict()
            for _ in range(count)
        ]
        print(json.dumps({"entity_id": payload.get("entity_id"), "challenges": challenges}, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - adapter boundary
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
