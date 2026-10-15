# CardiAgent Colab controls review — 2026-10-15

The uploaded notebook completed in 181.7 seconds. All 26 artifact hashes and all 102 files in the pinned source snapshot match. The original ZIP is retained byte for byte; `inspection/` exposes its protocol, provenance and primary reports without rewriting the original model checkpoints.

The expanded quantile screen passes across three seeds and seven domains. A local replay reproduces the summary excluding environment/runtime and reproduces all three model JSON checkpoints byte for byte. The joint model passes 9/9 screens, independent-source controls pass 9/9, and all nine deliberately broken dependence controls fail as intended. Joint AUC is 0.5044–0.5340. Local replay preserves all decisions and identical KS/Spearman errors; the maximum AUC change is 0.0000168 with newer numerical/scikit-learn dependencies. Joint population hashes differ across environments; the replay is not claimed byte-identical.

No scientific algorithm change is justified by this run. The failed historical CVAE remains failed; absence of optional torch is expected in this CPU-only control protocol. Synthetic reference draws are independent draws from a specified simulator, not independent biological evidence. Quantile KS rejection probability at the declared loose threshold is only 1.78e-9 under its null, so passing this gate is not a tight calibration guarantee. AUC near chance is not proof of indistinguishability. Exact-copy fraction zero is not a privacy guarantee.

Biological validation remains **blocked** pending a matched independent empirical cohort, frozen model qualification, cross-domain joint dependence/utility evaluation and blinded challenge evaluation. No available artifact supplies those data. No biological or clinical performance claim is added.

Local verification: 102 tests passed, six optional torch tests skipped; both full CPU protocols rerun. Hosted CI including CPU-ML is required before merging.
