import pytest

from cardiagent.statistics import paired_effect


def test_paired_effect_has_expected_summary() -> None:
    result = paired_effect([0.1, 0.2, 0.3, 0.4, 0.5])
    assert result.count == 5
    assert result.mean_delta == pytest.approx(0.3)
    assert result.std_delta > 0
    assert result.ci95_half_width > 0
    assert result.cohens_dz > 0


def test_constant_positive_effect_has_infinite_dz() -> None:
    result = paired_effect([0.2, 0.2, 0.2])
    assert result.cohens_dz == float("inf")


def test_empty_effect_rejected() -> None:
    with pytest.raises(ValueError, match="At least one paired observation"):
        paired_effect([])
