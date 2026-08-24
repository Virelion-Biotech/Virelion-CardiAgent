"""Command-oriented execution of the deterministic CardiAgent benchmark suite."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

from .benchmark_report import assert_accepted, build_report
from .suites import available_suites, build_suite


RUNNER_VERSION = "0.2"


def run_benchmark(
    *,
    suite_name: str = "baseline",
    seed: int | None = None,
    output_dir: str | Path | None = None,
    enforce_acceptance: bool = False,
) -> dict[str, object]:
    """Generate one suite, evaluate it, and optionally persist its artifacts."""
    suite = build_suite(suite_name, seed=seed)
    report = build_report(suite.challenges, suite=suite.name, suite_version=suite.version, seed=suite.seed)
    if enforce_acceptance:
        assert_accepted(report)
    result: dict[str, object] = {
        "runner_version": RUNNER_VERSION,
        "suite": suite.name,
        "suite_version": suite.version,
        "seed": suite.seed,
        "case_count": suite.case_count,
        "intent": suite.intent,
        "report": report.to_dict(),
    }
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{suite.name}.report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (root / f"{suite.name}.cases.jsonl").write_text(
            "".join(json.dumps(asdict(challenge), sort_keys=True) + "\n" for challenge in suite.challenges),
            encoding="utf-8",
        )
    return result


def run_all_benchmarks(*, output_dir: str | Path | None = None, enforce_acceptance: bool = False) -> list[dict[str, object]]:
    """Run every registered suite using its canonical seed."""
    return [
        run_benchmark(suite_name=name, output_dir=output_dir, enforce_acceptance=enforce_acceptance)
        for name in available_suites()
    ]
