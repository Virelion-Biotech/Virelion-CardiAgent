# Scientific audit changes — 2026-10-09

## Behavior

Hard-block CVAE-derived challenges from benchmark reports, blind benchmarks and full/blind handoffs. Add literal observation masking, left censoring and feature batch offsets with immutable ground truth and changed-entry masks.

## Scope and remaining evidence

The CVAE block is unconditional pending independently reviewed locked quality gates. Exploratory model research remains possible. Literal overlap mixtures and wiring corruptions into physiological challenge generators remain unresolved. Self-declared quality_status is not independent qualification of other learned generators.

## Implementation

- `src/cardiagent/corruptions.py`
- `src/cardiagent/scientific_gate.py`
- `tests/test_scientific_admission.py`
- `src/cardiagent/benchmark.py`
- `src/cardiagent/benchmark_report.py`
- `src/cardiagent/handoff.py`

## Verification

Regression tests accompany the changes. Repository test results are recorded in the audit completion report and draft pull request. Software regression checks do not establish numerical, biological, transport or clinical validity.
