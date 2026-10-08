import pytest
from cardiagent.screen_calibration import ks_null_gate


def test_nine_by_nine_gate_exact_null_failure_rate():
    result = ks_null_gate(9, 9, 0.2)
    assert result["accepted_label_orderings"] == 2**9
    assert result["total_label_orderings"] == 48620
    assert result["null_rejection_probability"] == pytest.approx(0.9894693541752365)


def test_ks_null_boundary_cases_and_symmetry():
    assert ks_null_gate(1, 1, 0)["null_acceptance_probability"] == 0
    assert ks_null_gate(1, 1, 1)["null_acceptance_probability"] == 1
    assert (
        ks_null_gate(3, 5, 0.4)["null_acceptance_probability"]
        == ks_null_gate(5, 3, 0.4)["null_acceptance_probability"]
    )


@pytest.mark.parametrize("n,m,d", [(True, 9, 0.2), (9, 0, 0.2), (9, 9, float("nan"))])
def test_invalid_gate_inputs(n, m, d):
    with pytest.raises(ValueError):
        ks_null_gate(n, m, d)
