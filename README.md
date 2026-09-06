# Virelion-CardiAgent

CardiAgent is a Python framework for generating reproducible, phenotype-level cardiac challenge cases for evaluation by downstream systems such as CardiVex.

## Scope

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

The repository also contains a conditional variational autoencoder for phenotype-level challenge generation and an adaptive challenge engine that uses downstream outcome summaries to select harder or diagnostically useful cases.

## Workflow

```text
challenge definitions / empirical profiles
                 ↓
        deterministic generator
                 ↓
          optional ML generator
                 ↓
       population quality checks
                 ↓
          blinded handoff
                 ↓
             CardiVex
                 ↓
        outcome-level feedback
                 ↓
       adaptive challenge engine
```

CardiAgent operates on abstract host-response/phenotype representations. It does not generate pathogen sequences, wet-lab protocols, culture conditions, doses, or other operational biological parameters.

## Installation

```bash
pip install -e '.[test]'
```

## Core objects

- `ChallengeDomain`
- `PhenotypeProfile`
- `ChallengeAgent`
- `ChallengeGenerator`
- `AgentGeneratorModel`
- `DetectionOutcome`
- `AdaptiveChallengeEngine`
- `PopulationReport`
- `CardiVexHandoff`
- `BlindCardiVexHandoff`
- `ChallengeManifest`
- `BlindBenchmark`

## Integration

CardiAgent creates challenge definitions and preserves ground truth. CardiVex performs downstream detection/characterization. CardiBench can provide benchmark context, CardiEval can score submissions, CardiBridge carries typed messages, and CardiTrace can record provenance.

## Validation and limitations

Generated cases are computational representations. They should not be treated as measured biological states unless explicitly derived from and linked to empirical observations. ML-generated cases require distribution and novelty checks before benchmark use.

## Testing

```bash
pytest
```

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See `LICENSE`.

## Citation

Cite the repository release and the empirical sources used to construct challenge profiles.
