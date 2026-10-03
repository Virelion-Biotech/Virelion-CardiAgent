from pathlib import Path

import pytest

from cardiagent.benchmark_report import assert_accepted, build_report, evaluate_acceptance, write_report
from cardiagent.suites import baseline_suite


def test_build_report_is_structured_and_complete() -> None:
    suite = baseline_suite(seed=123)
    report = build_report(
        suite.challenges,
        suite=suite.name,
        suite_version=suite.version,
        seed=suite.seed,
    )
    assert report.report_version == "0.3"
    assert report.suite == "baseline"
    assert report.seed == 123
    assert report.metrics["count"] == suite.case_count
    assert report.calibration["count"] == suite.case_count
    assert 0.0 <= report.domain_fidelity <= 1.0
    assert 0.0 <= report.quality_score <= 1.0
    assert all(report.acceptance.values())


def test_acceptance_rejects_bad_quality() -> None:
    suite = baseline_suite(seed=123)
    report = build_report(suite.challenges, suite=suite.name, suite_version=suite.version, seed=suite.seed, min_quality_score=2.0)
    assert not report.acceptance["minimum_quality_score"]
    with pytest.raises(ValueError, match="minimum_quality_score"):
        assert_accepted(report)


def test_evaluate_acceptance_rejects_duplicate_population() -> None:
    suite = baseline_suite(seed=1)
    metrics = report = build_report(suite.challenges, suite=suite.name, suite_version=suite.version, seed=suite.seed)
    del metrics
    from cardiagent.calibration import summarize_calibration
    from cardiagent.evaluation import population_metrics
    population = list(suite.challenges)
    population.extend(population[:20])
    duplicate_metrics = population_metrics(population)
    calibration = summarize_calibration(population)
    checks = evaluate_acceptance(duplicate_metrics, calibration, 1.0, 0.9, max_duplicate_rate=0.01)
    assert not checks["duplicate_rate_threshold"]


def test_write_report_creates_parent_directory(tmp_path: Path) -> None:
    suite = baseline_suite(seed=321)
    report = build_report(suite.challenges, suite=suite.name, suite_version=suite.version, seed=suite.seed)
    output = write_report(report, tmp_path / "reports" / "baseline.json")
    assert output.exists()
    assert '"suite": "baseline"' in output.read_text(encoding="utf-8")


def test_empty_population_rejected() -> None:
    with pytest.raises(ValueError, match="At least one challenge"):
        build_report([], suite="empty", suite_version="0.1", seed=1)
