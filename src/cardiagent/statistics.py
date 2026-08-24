"""Paired multi-seed statistics for CardiAgent generator comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from statistics import mean, pstdev
from typing import Sequence


@dataclass(frozen=True)
class PairedEffect:
    """Summary of paired candidate-minus-reference deltas."""

    count: int
    mean_delta: float
    std_delta: float
    ci95_half_width: float
    cohens_dz: float
    wilcoxon_signed_rank_p: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def paired_effect(values: Sequence[float]) -> PairedEffect:
    """Compute paired effect size and an exact Wilcoxon p-value when possible.

    SciPy is intentionally optional. Without it, the effect-size fields remain
    available and the p-value is ``None`` rather than silently using a weaker
    approximation.
    """
    if not values:
        raise ValueError("At least one paired observation is required")
    deltas = [float(value) for value in values]
    average = mean(deltas)
    spread = pstdev(deltas) if len(deltas) > 1 else 0.0
    ci = 1.96 * spread / sqrt(len(deltas)) if len(deltas) > 1 else 0.0
    dz = average / spread if spread > 0 else (float("inf") if average > 0 else 0.0)

    p_value: float | None = None
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        pass
    else:
        if len(deltas) >= 2 and any(delta != 0 for delta in deltas):
            try:
                p_value = float(wilcoxon(deltas, alternative="two-sided", method="auto").pvalue)
            except ValueError:
                p_value = None

    return PairedEffect(
        count=len(deltas),
        mean_delta=average,
        std_delta=spread,
        ci95_half_width=ci,
        cohens_dz=dz,
        wilcoxon_signed_rank_p=p_value,
    )
