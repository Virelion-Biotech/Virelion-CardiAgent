"""Command-line workflows for reproducible phenotype challenge evaluation."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from .generator import ChallengeGenerator
from .manifest import ChallengeManifest, build_manifest
from .models import ChallengeAgent, ChallengeDomain
from .serialization import positive_integer, read_json, strict_json, write_json


def _domain(value):
    try:
        return ChallengeDomain(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Unknown domain: {value}") from exc


def _population(path):
    payload = read_json(path)
    if "challenges" in payload:
        return list(ChallengeManifest.from_dict(payload).challenges)
    return [ChallengeAgent.from_dict(payload)]


def build_parser():
    parser = argparse.ArgumentParser(prog="cardiagent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Report version and optional ML availability")
    for name in ("generate", "manifest", "sample"):
        item = sub.add_parser(name)
        item.add_argument("--domain", type=_domain, required=True)
        item.add_argument("--severity", type=float, default=0.5)
        item.add_argument("--difficulty", type=float)
        if name != "sample":
            item.add_argument("--seed", type=int, default=0)
        item.add_argument("--count", type=int, default=1)
        item.add_argument("--output", type=Path)
        item.add_argument("--manifest-id", default="cardiagent-manifest")
        if name == "generate":
            item.add_argument("--agent-id")
        if name == "sample":
            item.add_argument("--model", type=Path, required=True)
    item = sub.add_parser("benchmark")
    item.add_argument("--suite", default="all")
    item.add_argument("--seed", type=int)
    item.add_argument("--output-dir", type=Path)
    item = sub.add_parser("blind")
    item.add_argument("--input", type=Path, required=True)
    item.add_argument("--public", type=Path, required=True)
    item.add_argument("--truth", type=Path, required=True)
    item.add_argument("--benchmark-id", default="cardiagent-blind")
    item.add_argument("--seed", type=int, default=0)
    item = sub.add_parser("train")
    item.add_argument("--input", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)
    item.add_argument("--epochs", type=int, default=25)
    item.add_argument("--batch-size", type=int, default=64)
    item.add_argument("--seed", type=int, default=0)
    item = sub.add_parser("handoff")
    item.add_argument("--input", type=Path, required=True)
    item.add_argument("--blind", action="store_true")
    item.add_argument("--output", type=Path)
    item = sub.add_parser("adapt")
    item.add_argument("--input", type=Path, required=True)
    item.add_argument("--outcomes", type=Path, required=True)
    item.add_argument("--count", type=int, default=10)
    item.add_argument("--seed", type=int, default=0)
    item.add_argument("--output", type=Path)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            from . import __version__

            result = {
                "service": "CardiAgent",
                "version": __version__,
                "status": "ok",
                "ml_available": importlib.util.find_spec("torch") is not None,
            }
        elif args.command in {"generate", "manifest", "sample"}:
            positive_integer(args.count, "count")
            if args.command == "sample":
                from .ml import AgentGeneratorModel

                items = AgentGeneratorModel.load(args.model).sample(
                    domain=args.domain,
                    severity=args.severity,
                    difficulty=0.75 if args.difficulty is None else args.difficulty,
                    count=args.count,
                )
            else:
                generator = ChallengeGenerator(seed=args.seed)
                if args.command == "generate" and args.count != 1:
                    raise ValueError("generate produces one case; use manifest for a batch")
                items = [
                    generator.generate(
                        args.domain,
                        severity=args.severity,
                        difficulty=args.difficulty,
                        agent_id=getattr(args, "agent_id", None),
                    )
                    for _ in range(args.count)
                ]
            result = (
                items[0].to_dict()
                if args.command == "generate"
                else build_manifest(
                    items, manifest_id=args.manifest_id, seed=items[0].seed
                ).to_dict()
            )
        elif args.command == "benchmark":
            from .benchmark_runner import run_all_benchmarks, run_benchmark

            if args.suite == "all" and args.seed is not None:
                raise ValueError("Select a suite to override its seed")
            result = (
                run_all_benchmarks(output_dir=args.output_dir)
                if args.suite == "all"
                else run_benchmark(
                    suite_name=args.suite, seed=args.seed, output_dir=args.output_dir
                )
            )
        elif args.command == "blind":
            from .benchmark import build_blind_benchmark

            if args.public.resolve() == args.truth.resolve():
                raise ValueError("Public and truth files must be different")
            benchmark = build_blind_benchmark(
                _population(args.input), benchmark_id=args.benchmark_id, seed=args.seed
            )
            # Validate both payloads before writing either destination.
            strict_json(benchmark.public_dict())
            strict_json(benchmark.evaluation_dict())
            write_json(benchmark.public_dict(), args.public)
            write_json(benchmark.evaluation_dict(), args.truth)
            result = {
                "public": str(args.public),
                "truth": str(args.truth),
                "case_count": len(benchmark.cases),
            }
        elif args.command == "train":
            from .ml import AgentGeneratorModel

            model = AgentGeneratorModel(seed=args.seed).fit(
                _population(args.input), epochs=args.epochs, batch_size=args.batch_size
            )
            model.save(args.output)
            result = {"model": str(args.output), "training_summary": model.training_summary}
            print(strict_json(result, indent=2, sort_keys=True))
            return 0
        elif args.command == "handoff":
            from .handoff import create_handoff, create_blind_handoff

            items = _population(args.input)
            if len(items) != 1:
                raise ValueError("handoff requires exactly one challenge")
            result = (
                create_blind_handoff(items[0]) if args.blind else create_handoff(items[0])
            ).to_dict()
        else:
            from .adaptive import AdaptiveChallengeEngine, DetectionOutcome

            engine = AdaptiveChallengeEngine(seed=args.seed)
            outcomes = read_json(args.outcomes)
            if not isinstance(outcomes, list) or not outcomes:
                raise ValueError("Outcomes must be a nonempty array")
            scores = engine.score([DetectionOutcome(**value) for value in outcomes])
            items = _population(args.input)
            known = {item.agent_id for item in items}
            from .benchmark import opaque_case_id

            known.update(opaque_case_id(item) for item in items)
            if any(score.case_id not in known for score in scores):
                raise ValueError("Outcome case ID is not in the input population")
            stage = engine.next_stage(sum(score.hardness for score in scores) / len(scores))
            parents = engine.hard_cases(items, top_k=min(10, len(items)))
            items = engine.evolve(parents, count=args.count, stage=stage)
            result = build_manifest(
                items, manifest_id="adaptive-manifest", seed=args.seed
            ).to_dict()
        output = getattr(args, "output", None)
        if output is not None:
            write_json(result, output)
        else:
            print(strict_json(result, indent=2, sort_keys=True))
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
