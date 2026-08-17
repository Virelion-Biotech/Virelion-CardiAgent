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
   CardiVex Handoff Contract
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
4. **Machine-readable handoff** — every challenge can be wrapped in a versioned CardiVex handoff contract.
5. **Manifested challenge sets** — batches retain order, identity, seed, and generator version.
6. **Benchmarkability** — challenge instances can support blinded detection and characterization benchmarks in CardiVex.

## Package

The Python package currently defines:

- `ChallengeDomain` — controlled challenge categories.
- `PhenotypeProfile` — normalized host-observable feature vector.
- `ChallengeAgent` — complete serializable challenge instance.
- `ChallengeGenerator` — deterministic instance generator.
- `CardiVexHandoff` / `create_handoff` — versioned downstream handoff envelope.
- `ChallengeManifest` / `build_manifest` — reproducible batch container.

## Example

```python
from cardiagent import (
    ChallengeDomain,
    ChallengeGenerator,
    build_manifest,
    create_handoff,
)

generator = ChallengeGenerator(seed=42)
challenge = generator.generate(
    ChallengeDomain.ISCHEMIC,
    severity=0.7,
)

handoff = create_handoff(challenge)
print(handoff.to_json())

manifest = build_manifest(
    [challenge],
    manifest_id="demo-001",
    seed=42,
)
print(manifest.to_json())
```

## Handoff contract

The handoff is deliberately narrow. CardiAgent supplies:

- challenge identity and domain
- generator/version provenance
- abstract severity and temporal descriptors
- phenotype-level expected observables
- reproducibility metadata

CardiVex supplies the **observed** response and detection result. CardiAgent does not decide whether a challenge is detectable.

## Roadmap

- Formal contract versioning and compatibility rules
- Scenario templates and controlled parameter distributions
- Challenge-set generation with reproducible manifests
- Independent CardiVex observation/result schema
- Blinded detection/characterization benchmark protocol
- Provenance and audit metadata
- Automated schema validation and CI
- AI-assisted challenge sampling and benchmark analysis
