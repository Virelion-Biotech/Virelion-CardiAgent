# Virelion-CardiAgent

**A reproducible challenge-agent framework for Virelion's cardiac detection platform.**

CardiAgent is the upstream sibling of **Virelion-CardiVex**. Its job is to create standardized, reproducible challenge representations that CardiVex can receive and evaluate against cardiac-cell response measurements.

## Architecture

```text
CardiAgent
   │
   ├── challenge domain
   ├── severity
   ├── temporal behavior
   ├── heterogeneity
   └── phenotype-level response profile
          │
          ▼
   Challenge Instance
          │
          ▼
       CardiVex
          │
          ├── detected
          ├── partially detected
          └── not detected
```

The boundary is intentional: **CardiAgent specifies the challenge; CardiVex independently evaluates the observable cardiac response.**

## Initial challenge domains

- ischemic
- inflammatory
- electrophysiologic
- toxic injury
- viral-like
- metabolic
- genetic susceptibility

These are phenotype-level categories. The repository does not implement pathogen construction, sequence design, wet-lab protocols, culture conditions, dosing instructions, or other operational biological procedures.

## Core design principles

1. **Reproducibility** — seeded generation produces identical challenge instances.
2. **Separation of concerns** — generation and detection remain independent systems.
3. **Observable phenotypes** — outputs are expressed as abstract host-response features.
4. **Machine-readable handoff** — every challenge is serializable to JSON.
5. **Benchmarkability** — challenge instances can eventually support blinded detection and characterization benchmarks in CardiVex.

## Package

The initial Python package defines:

- `ChallengeDomain` — controlled challenge categories.
- `PhenotypeProfile` — normalized host-observable feature vector.
- `ChallengeAgent` — complete serializable challenge instance.
- `ChallengeGenerator` — deterministic instance generator.

## Example

```python
from cardiagent import ChallengeDomain, ChallengeGenerator

generator = ChallengeGenerator(seed=42)
challenge = generator.generate(
    ChallengeDomain.ISCHEMIC,
    severity=0.7,
)

print(challenge.to_json())
```

## Roadmap

- Formal challenge schema and versioning
- Scenario templates and controlled parameter distributions
- Challenge-set generation with reproducible manifests
- CardiVex handoff contract
- Detection/characterization benchmark protocol
- Provenance and audit metadata
- Automated validation and test coverage
- AI-assisted challenge sampling and benchmark analysis
