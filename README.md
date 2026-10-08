# Virelion-CardiAgent

CardiAgent is a Python framework for generating reproducible, phenotype-level cardiac challenge cases for computational evaluation.

## What it contains

A `ChallengeAgent` can represent:

- challenge domain;
- severity and difficulty;
- onset, persistence, recovery, and temporal state;
- observable phenotype signals;
- cell-context categories;
- response heterogeneity;
- phenotype overlap;
- measurement noise and partial observation;
- confounder tags and provenance.

The repository also contains a conditional variational autoencoder for phenotype-level challenge generation and an adaptive challenge engine that uses outcome summaries to select harder or diagnostically useful cases.

CardiAgent operates on abstract host-response/phenotype representations. It does not generate pathogen sequences, wet-lab protocols, culture conditions, doses, or other operational biological parameters.

## Installation

```bash
pip install -e '.[test]'
```

## Usage

The main Python objects are `ChallengeDomain`, `PhenotypeProfile`, `ChallengeAgent`, `ChallengeGenerator`, `AgentGeneratorModel`, `DetectionOutcome`, `AdaptiveChallengeEngine`, `PopulationReport`, `ChallengeManifest`, and `BlindBenchmark`.

A typical workflow is:

```text
challenge definitions / empirical profiles
        ↓
deterministic or ML generation
        ↓
population quality checks
        ↓
challenge manifest / blinded representation
```

## Inputs and outputs

**Inputs:** abstract phenotype profiles, challenge-domain definitions, severity/temporal parameters, heterogeneity/noise settings, empirical distributions where available, and generation configuration.

**Outputs:** phenotype-level challenge scenarios, population reports, manifests, blinded benchmark representations, ground-truth records, and provenance metadata.

## Validation

Validation includes population quality checks and deterministic/reproducibility checks. ML-generated cases require distribution, novelty, and plausibility checks before benchmark use. Generated cases are computational representations unless explicitly linked to empirical observations.

## Limitations

Generated cases are not automatically measured biological states. Synthetic and ML-generated scenarios can reproduce statistical patterns without reproducing biological mechanisms. Ground truth quality depends on the source profiles and assumptions. Challenge difficulty is conditional on the representation and detector used.

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See `LICENSE`.

## Complete CPU workflows (0.4.0)

The core package needs Python 3.10+ and no GPU or PyTorch. Install optional ML
on CPU explicitly:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test,validation]'
cardiagent doctor
cardiagent generate --domain ischemic --seed 7 --output case.json
cardiagent manifest --domain ischemic --count 40 --seed 7 --output population.json
cardiagent benchmark --output-dir artifacts/benchmarks
cardiagent blind --input population.json --public public.json --truth truth.json
cardiagent handoff --input case.json --blind --output handoff.json
cardiagent train --input population.json --epochs 25 --output model.pt
cardiagent sample --model model.pt --domain ischemic --count 10 --output samples.json
cardiagent adapt --input population.json --outcomes outcomes.json --count 10 --output adapted.json
```

`outcomes.json` is an array, for example
`[{"case_id":"<case_id from public.json>","predicted_domain":null,"confidence":0.2,"detected":false,"characterization_correct":false}]`.
Blinded IDs remain stable across shuffle order and can be joined to adaptive
scores. Keep truth files separate from downstream benchmark inputs. Opaque IDs
are reproducible hashes, not cryptographic anonymization against known inputs.

Models persist their private random state and sample counter. Reloading resumes
that sequence; `sample` intentionally has no seed override. Legacy checkpoints
without random-state metadata cannot resume an earlier sequence exactly.
Checkpoint loading uses PyTorch's restricted `weights_only=True` loader.
JSON and checkpoint outputs use atomic replacement per file; a public/truth
pair is not a two-file transaction.

## Historical CVAE validation limits (0.4.0)

The CPU held-out CVAE protocol **failed** its predeclared quality checks.
Reproducibility passed, but all 21 model/domain and 189 condition screens failed;
all domain discriminator AUCs were 1.0. The ML generator is not benchmark-qualified.
See [CPU audit](validation/cpu/AUDIT.md) and preserved raw results.

Deterministic tests establish software behavior on synthetic abstract phenotypes.
They do not establish patient realism, clinical efficacy, or external biological
validity. Overlap and partial-observation controls currently describe scenarios;
they do not implement feature mixing or observation masking. CVAE samples are
static profiles, and adaptive offspring omit inherited trajectories to avoid
presenting stale time series as generated data. Handoff schemas are packaged and
tested; a live CardiVex consumer integration is not established in this repository.

Statistical summaries use sample standard deviations. Student-t confidence
intervals require SciPy and at least two observations; unavailable intervals and
undefined standardized effects are JSON null, never infinity. Repeated seeds
are rejected as independent experiment replicates.

## Generator recovery in 0.5.0

The massive CVAE failures were reproduced and diagnosed on CPU. The old
nine-observation condition gate also rejects independent reference-source draws:
its `KS <= 0.2` threshold has a 98.95% single-feature false rejection probability
under an identical continuous-distribution null. Original failed evidence and
thresholds are retained.

A **separate conditional marginal quantile model** passes all 21 domain and
189 condition screens in a predeclared fresh confirmation with 512 held-out
observations per condition. It needs no PyTorch or GPU. The support-aware CVAE
improves but remains unqualified.

```bash
cardiagent train --model-family quantile --input population.json --output quantile.json
cardiagent sample --model-family quantile --model quantile.json --domain ischemic --severity 0.5 --difficulty 0.5 --count 40 --output samples.json
python scripts/validate_quantile_cpu.py
```

The quantile model assumes conditional feature independence, supports only
observed exact domain/severity/difficulty strata, and produces static profiles.
It does not establish patient realism or biological mechanisms. Arbitrary newly
trained models do not inherit the audit's qualification.

[Detailed diagnosis and limits](docs/GENERATOR_RECOVERY.md) ·
[Expanded confirmation](validation/cpu/recovery/expanded/summary.json) ·
[Unchanged-screen controls](validation/cpu/recovery/summary.json)
