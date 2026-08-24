from pathlib import Path

import pytest

from cardiagent.benchmark_report import build_report, write_report
from cardiagent.suites import baseline_suite


def test_build_report_is_structured_and_complete() -> None:
    suite = baseline_suite(seed=123)
    report = build_report(
        suite.challenges,
        suite=suite.name,
        suite_version=suite.version,
        seed=suite.seed,
    )
    assert report.report_version == "0.1"
    assert report.suite == "baseline"
    assert report.seed == 123
    assert report.metrics["count"] == suite.case_count
    assert 0.0 <= report.domain_fidelity <= 1.0
    assert 0.0 <= report.quality_score <= 1.0


def test_write_report_creates_parent_directory(tmp_path: Path) -> None:
    suite = baseline_suite(seed=321)
    report = build_report(suite.challenges, suite=suite.name, suite_version=suite.version, seed=suite.seed)
    output = write_report(report, tmp_path / "reports" / "baseline.json")
    assert output.exists()
    assert '"suite": "baseline"' in output.read_text(encoding="utf-8")


def test_empty_population_rejected() -> None:
    with pytest.raises(ValueError, match="At least one challenge"):
        build_report([], suite="empty", suite_version="0.1", seed=1)
