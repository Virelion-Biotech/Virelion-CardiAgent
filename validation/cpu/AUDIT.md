# CardiAgent 0.4.0 CPU audit — 2026-10-07

## Software result

65 tests passed, with 90.68% statement coverage. Ruff passed. Source and wheel
builds succeeded. The wheel was installed into a clean virtual environment with
no optional dependencies: doctor and all eight deterministic benchmark suites
completed. The CPU ML training/save/load/sample and blinded-outcome/adaptation
CLI paths are exercised by integration tests. The canonical five-seed,
25-epoch comparison completed on 336 training cases; its fingerprinted artifact
is `generator-experiment.json`. A comparison artifact is not a quality approval.

Repairs include strict finite numeric inputs and portable JSON, unique ML IDs,
private random state persisted across checkpoints, restricted checkpoint loading,
blinding audits, stable opaque-ID joins for outcome ranking, rejection of duplicate
truth IDs, removal of stale adaptive trajectories, atomic artifact writes,
packaged schemas, complete CLI entry points, and sample-SD/Student-t summaries.
Optional unavailable intervals and undefined effect sizes are represented as null.

## Scientific result: FAIL

The existing held-out protocol retained its thresholds and configuration:
2,835 synthetic source cases, 1,701 training / 567 validation / 567 held-out,
seven domains, nine severity levels, three model seeds, 100 training epochs.
Reproducibility passed. All 21 model/domain screens and 189 condition screens
failed; 21 discriminator checks failed, with AUC 1.0. The CVAE does not reproduce
the held-out distributions well enough to qualify as a benchmark generator.
These results are internal computational checks against the deterministic source,
not external patient or biological validation.

`heldout/summary.json`, condition metrics and the Markdown report are readable
summaries. Full raw JSON and split membership are gzip-compressed without loss;
`heldout/checksums.json` records compressed and uncompressed SHA-256 values.
The environment records each source module's SHA-256 and the harness hash.
The recorded git commit is the checkout baseline before this audit was committed;
use those file hashes to identify the actual audited source. No failure thresholds
were weakened. The manually invoked held-out workflow enforces the quality gate
and is expected to fail until the model improves under a newly declared protocol.

## Remaining product limits

Overlap and partial observation remain descriptive controls rather than actual
mixing/masking. CVAE profiles are static. Adaptive offspring discard inherited
trajectories. Schemas and handoff producers are tested, but this repo does not
establish a live CardiVex consumer integration. There is no patient reference
cohort or independent clinical validation. Software checks passing must not be
interpreted as removal of these gaps or a claim that the product is perfect.

Publication: software repairs and preserved evidence committed to main as `9763a6efc88c4ca078628bc97a0389288b5fb0be`. GitHub CI verifies the published software separately from the failed scientific screen.

## Follow-up in 0.5.0

This file preserves the original failed CVAE audit. The later support repair, independent reference-control diagnosis, and separately declared expanded CPU quantile confirmation are documented in [GENERATOR_RECOVERY.md](../../docs/GENERATOR_RECOVERY.md). The new quantile result does not retroactively qualify this original CVAE run.
