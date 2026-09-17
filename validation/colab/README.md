# Colab Scientific Validation Results

These artifacts come from a Colab-based scientific validation run against commit `4348d454459d6c459937a2804af70a72cb2952a2`.

## Summary

See `scientific_validation_summary.txt`:

- reproducibility: **PASS**
- bounds: **PASS**
- uniqueness: **PASS**
- trajectories: **PASS**
- ML status: **PASS_WITH_DIAGNOSTICS**

Elapsed: ~18.6 s

## Files

| File | Description |
|------|-------------|
| `scientific_validation_summary.txt` | Human-readable pass/fail summary + ML training metrics |
| `validation_colab.log` | Full console log from the validation run |
| `package_inventory.txt` | Inventory of the original validation package |

The full detailed JSON report (`scientific_validation_report.json`, ~95 KB) was omitted from this commit to keep the repository lightweight; it can be regenerated from the same commit and configuration or supplied separately if needed.

Original archives (`CardiAgent_validation_results.zip` / `.tar.gz`) were not committed to avoid redundancy.
