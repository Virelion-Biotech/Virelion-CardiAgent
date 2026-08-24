import pytest

from cardiagent.calibration import monotonicity_score, summarize_calibration
from cardiagent.generator import ChallengeGenerator
from cardiagent.models import ChallengeDomain


def test_calibration_reports_expected_control_means() -> None:
    generator = ChallengeGenerator(seed=7)
    cases = [
        generator.generate(ChallengeDomain.ISCHEMIC, severity=severity, difficulty=severity)
        for severity in (0.1, 0.5, 0.9)
    ]
    summary = summarize_calibration(cases)
    assert summary.count == 3
    assert summary.difficulty_mean == pytest.approx(0.5)
    assert 0.0 <= summary.severity_phenotype_mae <= 1.0
    assert 0.0 <= summary.difficulty_overlap_mae <= 1.0


def test_empty_calibration_rejected() -> None:
    with pytest.raises(ValueError, match="At least one challenge"):
        summarize_calibration([])


def test_monotonicity_score() -> None:
    assert monotonicity_score([(0.1, 0.2), (0.5, 0.5), (0.9, 0.8)]) == 1.0
    assert monotonicity_score([(0.1, 0.8), (0.5, 0.5), (0.9, 0.2)]) == 0.0
