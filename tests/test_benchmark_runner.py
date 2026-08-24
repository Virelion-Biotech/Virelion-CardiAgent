from pathlib import Path

from cardiagent.benchmark_runner import run_all_benchmarks, run_benchmark
from cardiagent.suites import available_suites


def test_runner_is_reproducible_except_timestamp() -> None:
    first = run_benchmark(suite_name="baseline", seed=42)
    second = run_benchmark(suite_name="baseline", seed=42)
    assert first["suite"] == second["suite"]
    assert first["seed"] == second["seed"]
    assert first["case_count"] == second["case_count"]
    assert first["report"]["metrics"] == second["report"]["metrics"]
    assert first["report"]["calibration"] == second["report"]["calibration"]
    assert first["report"]["quality_score"] == second["report"]["quality_score"]
    assert first["report"]["acceptance"] == second["report"]["acceptance"]


def test_runner_persists_report_and_cases(tmp_path: Path) -> None:
    run_benchmark(suite_name="severity", seed=42, output_dir=tmp_path, enforce_acceptance=True)
    assert (tmp_path / "severity.report.json").exists()
    assert (tmp_path / "severity.cases.jsonl").exists()
    assert (tmp_path / "severity.cases.jsonl").read_text(encoding="utf-8").strip()


def test_all_registered_suites_run_with_acceptance() -> None:
    results = run_all_benchmarks(enforce_acceptance=True)
    assert {result["suite"] for result in results} == set(available_suites())
    assert all(int(result["case_count"]) > 0 for result in results)
    assert all(all(result["report"]["acceptance"].values()) for result in results)
