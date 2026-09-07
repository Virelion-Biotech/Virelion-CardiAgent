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
