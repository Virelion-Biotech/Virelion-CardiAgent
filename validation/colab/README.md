# Colab Scientific Validation Results

These artifacts come from a Colab-based scientific validation run against commit `4348d454459d6c459937a2804af70a72cb2952a2`.

## Summary

See `scientific_validation_summary.txt`:

- reproducibility: **PASS**
- bounds: **PASS**
- uniqueness: **PASS**
- trajectories: **PASS**
- ML status: **PASS_WITH_DIAGNOSTICS**

- 700 training examples
- 100 ML epochs
- Final CVAE loss: 0.005856
- Runtime: 18.61 s

## Files present in repo

| File | Description |
|------|-------------|
| `scientific_validation_summary.txt` | Human-readable pass/fail summary + ML training metrics |
| `validation_colab.log` | Full console log from the validation run (includes NumPy correlation warnings) |
| `package_inventory.txt` | Inventory of the original validation package |

## Full detailed report

The complete `scientific_validation_report.json` (~95 KB) contains the full numerical results (per-domain population statistics, ML novelty/distribution diagnostics, environment, etc.).

It was not committed here because of size limits when pushing large single-file content through the available GitHub integration. The file is available as a downloadable artifact from the conversation that produced these results, and can also be regenerated from the same commit + Colab validation script.

**Note on the NumPy warnings**: the log shows repeated `invalid value encountered in divide` from `np.corrcoef` caused by zero-variance features. The validation script currently masks these with `nan_to_num`; the multivariate-fidelity / correlation results should be treated as diagnostic until that is hardened.

Original archives (`.zip` / `.tar.gz`) were deliberately not committed to avoid redundancy.
