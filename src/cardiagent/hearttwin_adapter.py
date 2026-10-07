"""HeartTwin local-command adapter for CardiAgent challenge generation."""

from __future__ import annotations

import json
import os
import sys

from .generator import ChallengeGenerator
from .models import ChallengeDomain
from .serialization import finite_number, positive_integer, strict_json


def _params(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("HeartTwin payload must be an object")
    context = payload.get("context", {})
    if not isinstance(context, dict):
        raise ValueError("context must be an object")
    if "cardiagent" in context:
        params = context["cardiagent"]
    else:
        params = {}
        for obs in payload.get("observations", []):
            values = obs.get("values")
            if (
                obs.get("modality") == "structural"
                and isinstance(values, dict)
                and "domain" in values
            ):
                params = values
                break
    if not isinstance(params, dict):
        raise ValueError("CardiAgent parameters must be an object")
    if unknown := set(params) - {"domain", "seed", "count", "severity", "difficulty"}:
        raise ValueError(f"Unsupported CardiAgent parameters: {sorted(unknown)}")
    return params


def main() -> int:
    raw = (
        sys.stdin.read()
        if os.environ.get("HEARTTWIN_PAYLOAD_STDIN") == "1"
        else os.environ.get("HEARTTWIN_PAYLOAD")
    )
    if not raw:
        print("HEARTTWIN_PAYLOAD environment variable not set", file=sys.stderr)
        return 1
    try:
        payload = json.loads(raw)
        params = _params(payload)
        domain = ChallengeDomain(str(params.get("domain", "ischemic")))
        seed = params.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        count = positive_integer(params.get("count", 1), "count")
        if count < 1:
            raise ValueError("count must be >= 1")
        generator = ChallengeGenerator(seed=seed)
        challenges = [
            generator.generate(
                domain,
                severity=finite_number(params.get("severity", 0.5), "severity", low=0, high=1),
                difficulty=(
                    finite_number(params["difficulty"], "difficulty", low=0, high=1)
                    if params.get("difficulty") is not None
                    else None
                ),
            ).to_dict()
            for _ in range(count)
        ]
        print(
            strict_json(
                {"entity_id": payload.get("entity_id"), "challenges": challenges}, sort_keys=True
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - adapter boundary
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
