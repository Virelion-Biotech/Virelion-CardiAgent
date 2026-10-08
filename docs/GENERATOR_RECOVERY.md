# CardiAgent 0.5.0: CPU diagnosis and generator recovery

## Conclusion

No GPU is required. The original CVAE had genuine distributional defects, and the original per-condition screen also had an extreme false rejection rate. This release repairs exact constant-feature support, adds an explicitly separate CPU conditional marginal quantile generator, and preserves every old failure. It does not relabel the CVAE as qualified.

The expanded, independently declared quantile protocol passes all 21 domain and 189 condition screens against a fresh synthetic reference, with the original screening thresholds unchanged. This is internal statistical fidelity for the tested exact conditions, not external biological validation or evidence of realistic patient responses.

## What caused the massive failures

1. The original sigmoid decoder returned strictly positive values for features that the training source made exactly zero. A discriminator could exploit these support violations.
2. MSE reconstruction with latent sampling produced insufficient phenotype variation. Repairing exact support does not solve all distributional defects. We measured output mismatch; we did not establish a specific latent posterior-collapse mechanism.
3. Each old condition compared nine reference observations with nine generated observations and required empirical KS distance at most 0.2. With equal sample sizes of nine, the smallest nonzero KS increment is 1/9. Under independent samples from an identical continuous distribution, only 512 of the 48,620 possible sample-label orderings satisfy this gate. Its single-feature false rejection probability is **98.9469%**. The exact calculation is implemented and tested in `screen_calibration.py`. It assumes no ties; clipped boundary atoms require the explicit reference-control experiment.
4. Undefined correlation entries were serialized as nonstandard JSON NaN. They now serialize as null, with the existing lists of undefined feature pairs retained. No missing correlation is replaced with zero, and no numerical screening threshold changes.

SciPy documents that its KS comparison concerns independent samples from continuous distributions and provides exact/asymptotic p-value methods: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html. Its occasional fallback warning changes the p-value calculation, not the empirical KS distance used by this screen. Our finite-sample acceptance calculation is an independent label-path count, not a new choice of p-value cutoff.

VAE posterior collapse is a known training phenomenon (He et al., ICLR 2019, https://arxiv.org/abs/1901.05534), but increasing compute does not by itself establish that this implementation models a distribution correctly. The actual root-cause evidence here is observed support mismatch, variance mismatch, and independent control failure.

## Unchanged nine-sample screen

| Population / implementation | Failed domain screens / 21 | Failed condition screens / 189 | Discriminator failures / 21 |
|---|---:|---:|---:|
| Original CVAE, preserved 0.4.0 audit | 21 | 189 | 21 |
| Support-aware CVAE, original development split | 16 | 189 | 2 |
| Independent reference control, original development split | 0 | 189 | 0 |
| Conditional quantile, original development split | 0 | 189 | 0 |
| Support-aware CVAE, fresh confirmation split | 17 | 189 | 4 |
| Independent reference control, fresh confirmation split | 0 | 189 | 0 |
| Conditional quantile, fresh confirmation split | 0 | 189 | 0 |

The reference control generates independent draws from the exact source law and never trains on or reads held-out observations. Its 189 failed condition screens demonstrate that the small-sample gate cannot qualify a generator reliably. This does not excuse the CVAE's remaining domain-level failures. Development and fresh confirmation reports use fixed model seeds 17/29/41, 100 CVAE epochs, and the unchanged source/scoring implementation apart from portable null serialization.

## Separate expanded confirmation

`validation/cpu/recovery/expanded_protocol.json` was written before execution. It declares a fresh reference seed 20261010, 63 conditions, 512 training and 512 held-out observations per condition, and 512 generated observations per condition for each of three model seeds. No validation-driven hyperparameter search or threshold tuning is performed. The model is the fixed conditional quantile algorithm, not a substituted CVAE. Training and held-out IDs are disjoint.

| Expanded quantile result | Value |
|---|---:|
| Training examples | 32,256 |
| Held-out examples | 32,256 |
| Generated examples across three seeds | 96,768 |
| Failed domain screens | 0 / 21 |
| Failed condition screens | 0 / 189 |
| Discriminator failures | 0 / 21 |
| Domain discriminator AUC range | 0.483884–0.517459 |
| Evaluation time across three model seeds | 35.59 seconds, excluding source generation |

All old threshold values remain unchanged. Increasing the sample size is a separately declared protocol change, not a retroactive pass for the original protocol. The screen evaluates the eight phenotype fields. Onset, persistence and heterogeneity are sampled, but their distributional fidelity is not qualified by this screen. The raw expanded results, split membership, protocol, source fingerprints, and checksums are committed under `validation/cpu/recovery/expanded`.

## Implementation and scope

`AgentGeneratorModel` learns exact domain-level constants from training rows only and restores them after decoding. New checkpoints persist this support information and private random state. Legacy checkpoints remain explicitly labeled `0.3-ml-cvae` and `legacy_unconstrained`; loading them does not manufacture support information. New checkpoints missing support data fail closed.

`ConditionalQuantileGenerator` learns sorted marginal values for each exact domain/severity/difficulty stratum. It samples piecewise linear midpoint quantiles, preserving constants and boundary atoms. It assumes conditional feature independence; it does not learn joint copulas, interpolate to unseen conditions, infer mechanisms, or generate temporal trajectories. Unseen strata, undersized strata, duplicate training IDs and malformed checkpoints fail closed. Checkpoint reload resumes the private random sequence exactly.

A passing family-level audit does not automatically qualify arbitrary newly fitted checkpoints. Generated objects retain `quality_status: not_qualified`; qualify a specific dataset/model against its intended benchmark independently. This implementation's passing result applies to the declared synthetic source and conditions only.

## Run everything yourself on CPU

```bash
python -m pip install -e '.[test,validation]'
# Optional, only for CVAE diagnosis:
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/diagnose_generator.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/validate_quantile_cpu.py
```

The expanded script requires NumPy/SciPy/scikit-learn, but not PyTorch. It reconstructs all source data, writes numerical reports and checkpoints, verifies checkpoint continuation, and exits nonzero on a failed screen. Its source guard rejects a run if implementation or protocol files change during execution. The default-push CPU quality job repeats this expanded check and uploads complete reports and checkpoints as `quantile-cpu-validation`. No Colab or Drive workflow is needed.

Checkpoints are about 8.6 MB each and reproducible from the pinned code/protocol. They are uploaded as CI artifacts rather than committed as redundant model files. `checkpoint_manifest.json` records their exact hashes; report/split gzip artifacts remain committed and checksum-verifiable. For long-term model retention, regenerate from the protocol or retain the CI artifact before GitHub's artifact expiry.

Use the CPU generator explicitly:

```bash
cardiagent train --model-family quantile --input population.json --output quantile.json
cardiagent sample --model-family quantile --model quantile.json --domain ischemic --severity 0.5 --difficulty 0.5 --count 40 --output samples.json
```

The input must contain at least eight unique training examples for every requested exact condition. Difficulty defaults to severity for quantile sampling. The existing default `cvae` remains explicit and unqualified; there is no silent switch of model families.

## Software verification

The original checkout passed 65 tests on CPU. The repaired implementation passed 84 tests, including support preservation, legacy checkpoint behavior, private-RNG continuation, strict unseen-condition rejection, exact KS calibration, and the quantile CLI. Core-only installation and installed-wheel checks are also run; the expanded scientific job runs without installing PyTorch.

Local reports record the checkout head before publication. Their complete source-file and harness hashes identify the audited working code; the prepublication Git head alone does not. Hosted CI reruns the same source from the published commit.
