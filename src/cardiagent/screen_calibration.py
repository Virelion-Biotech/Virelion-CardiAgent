"""Exact null calibration of an empirical two-sample KS distance gate.

For independent IID draws from the same continuous distribution, every label
ordering has equal probability. Dynamic programming counts label paths that
stay within the specified empirical-CDF distance. Ties/atoms require a separate
reference-control calculation; this formula must not be applied to them.
"""

from fractions import Fraction
from math import comb

from .serialization import finite_number, positive_integer


def ks_null_gate(n, m, maximum_distance):
    positive_integer(n, "n")
    positive_integer(m, "m")
    finite_number(maximum_distance, "maximum_distance", low=0, high=1)
    threshold = Fraction(str(maximum_distance))
    paths = [[0] * (m + 1) for _ in range(n + 1)]
    paths[0][0] = 1
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            if abs(Fraction(i, n) - Fraction(j, m)) <= threshold:
                paths[i][j] = (paths[i - 1][j] if i else 0) + (paths[i][j - 1] if j else 0)
    accepted = paths[n][m]
    total = comb(n + m, n)
    return {
        "n_reference": n,
        "n_generated": m,
        "maximum_distance": maximum_distance,
        "accepted_label_orderings": accepted,
        "total_label_orderings": total,
        "null_acceptance_probability": accepted / total,
        "null_rejection_probability": 1 - accepted / total,
        "assumptions": "independent IID samples; identical continuous distribution; no ties",
    }
