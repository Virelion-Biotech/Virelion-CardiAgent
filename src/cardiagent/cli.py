"""Command-line entry points for reproducible CardiAgent challenge generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generator import ChallengeGenerator
from .manifest import build_manifest
from .models import ChallengeDomain


def _domain(value: str) -> ChallengeDomain:
    try:
        return ChallengeDomain(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"unknown domain {value!r}; choose from {[d.value for d in ChallengeDomain]}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate phenotype-level CardiAgent challenges")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="generate one challenge")
    generate.add_argument("--domain", type=_domain, required=True)
    generate.add_argument("--severity", type=float, default=0.5)
    generate.add_argument("--difficulty", type=float)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--agent-id")

    manifest = sub.add_parser("manifest", help="generate and freeze a challenge manifest")
    manifest.add_argument("--domain", type=_domain, required=True)
    manifest.add_argument("--count", type=int, default=1)
    manifest.add_argument("--severity", type=float, default=0.5)
    manifest.add_argument("--difficulty", type=float)
    manifest.add_argument("--seed", type=int, default=0)
    manifest.add_argument("--manifest-id", default="cardiagent-manifest")
    manifest.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generator = ChallengeGenerator(seed=args.seed)
    if args.command == "generate":
        challenge = generator.generate(
            args.domain, severity=args.severity, difficulty=args.difficulty, agent_id=args.agent_id
        )
        print(challenge.to_json())
        return 0

    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    challenges = [
        generator.generate(args.domain, severity=args.severity, difficulty=args.difficulty)
        for _ in range(args.count)
    ]
    result = build_manifest(
        challenges,
        manifest_id=args.manifest_id,
        seed=args.seed,
        generator_version=generator.VERSION,
    )
    text = result.to_json()
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
