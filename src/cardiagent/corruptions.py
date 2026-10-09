"""Literal observation corruptions with preserved ground truth and changed-entry masks.

These alter an observation array. They are not mechanistic simulations of injury.
"""

from __future__ import annotations
import numpy as np


def corrupt_observations(values, *, kind, seed=0, fraction=None, threshold=None, offset=None):
    truth = np.asarray(values, dtype=float)
    if truth.ndim != 2 or not truth.size or not np.isfinite(truth).all():
        raise ValueError("Observations must be a nonempty finite row-by-feature matrix")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    observed = truth.copy()
    mask = np.zeros(truth.shape, dtype=bool)
    if kind == "masking":
        if (
            isinstance(fraction, bool)
            or fraction is None
            or not np.isfinite(fraction)
            or not 0 <= fraction <= 1
        ):
            raise ValueError("masking requires fraction in [0,1]")
        n = int(round(fraction * truth.size))
        selected = np.random.default_rng(seed).choice(truth.size, n, replace=False)
        mask.flat[selected] = True
        observed[mask] = np.nan
    elif kind == "left_censoring":
        if isinstance(threshold, bool) or threshold is None or not np.isfinite(threshold):
            raise ValueError("left_censoring requires a finite threshold")
        mask = truth < threshold
        observed[mask] = threshold
    elif kind == "batch_offset":
        shift = np.asarray(offset, dtype=float)
        if shift.shape != (truth.shape[1],) or not np.isfinite(shift).all():
            raise ValueError("batch_offset requires one finite offset per feature")
        observed = truth + shift
        mask = observed != truth
    else:
        raise ValueError("Unsupported literal corruption")
    truth = truth.copy()
    truth.setflags(write=False)
    return {"observed": observed, "truth": truth, "changed_mask": mask, "kind": kind, "seed": seed}
