# Virelion-CardiAgent

**A reproducible challenge-agent framework for Virelion's cardiac detection platform.**

CardiAgent is the upstream sibling of **Virelion-CardiVex**. Its job is to create standardized, reproducible, increasingly realistic challenge representations that CardiVex can receive and evaluate against cardiac-cell response measurements.

## Architecture

```text
CardiAgent
   │
   ├── challenge domain
   ├── severity / difficulty
   ├── temporal trajectory
   ├── heterogeneity
   ├── phenotype overlap
   ├── observation noise / partial observation
   └── phenotype-level response profile
          │
          ▼
   Detailed Challenge Agent
          │
          ├────────────── trusted handoff ──────────────► CardiVex
          │
          └────────────── blind handoff ─────────────────► CardiVex
                                                           │
                                                   independent call
                                                           │
                                             detected / partially detected
                                                   / not detected
```

The boundary is intentional: **CardiAgent specifies the challenge; CardiVex independently evaluates the observable cardiac response.**

## Challenge-agent design

CardiAgent does more than sample one static phenotype vector. Each generated agent can contain:

- domain and provenance
- severity and independent difficulty
- onset and persistence
- a deterministic baseline → early → peak → persistent → recovery trajectory
- dominant and secondary observable signals
- cell-context categories
- response heterogeneity
- phenotype overlap with a neighboring challenge domain
- measurement noise and partial-observation characteristics
- explicit confounder tags
- scenario-family and generator version metadata

The generator is deterministic for a given seed and call sequence, so difficult cases can be reproduced exactly for debugging and benchmarking.

## Initial challenge domains

- ischemic
- inflammatory
- electrophysiologic
- toxic injury
- viral-like
- metabolic
- genetic susceptibility

These are phenotype-level categories. The repository does not implement pathogen construction, sequence design, wet-lab protocols, culture conditions, dosing instructions, or other operational biological procedures.

## Blind benchmarking

For detection benchmarking, use `build_blind_benchmark()` or `create_blind_handoff()`.

The public presentation contains the observable challenge representation but omits the challenge domain and other direct ground-truth labels. The evaluator record retains the true domain, severity, scenario family, difficulty, and overlap reference separately.

This enables a clean evaluation loop:

```text
challenge truth
     │
     ├──► blinded presentation ──► CardiVex ──► prediction
     │                                      │
     └──────────── private truth ───────────┴──► scoring
```

CardiAgent therefore does **not** decide whether a challenge is detectable. It creates the challenge and preserves the ground truth so CardiVex can be tested independently.

## Example

```python
from cardiagent import (
    ChallengeDomain,
    ChallengeGenerator,
    build_blind_benchmark,
    create_blind_handoff,
)

generator = ChallengeGenerator(seed=42)
challenges = [
    generator.generate(
        ChallengeDomain.ISCHEMIC,
        severity=0.7,
        difficulty=0.85,
    ),
    generator.generate(
        ChallengeDomain.INFLAMMATORY,
        severity=0.45,
        difficulty=0.75,
    ),
]

# One-case blind handoff for CardiVex.
blind = create_blind_handoff(challenges[0], case_id="bench-0001")
print(blind.to_json())

# Reproducible multi-case benchmark with private ground truth.
benchmark = build_blind_benchmark(
    challenges,
    benchmark_id="demo-001",
    seed=42,
)
print(benchmark.public_json())
print(benchmark.evaluation_json())
```

## Handoff contract

The trusted handoff supplies challenge identity, domain, generator/version provenance, abstract severity and temporal descriptors, phenotype-level expected observables, and reproducibility metadata.

The **blind handoff** supplies only the challenge presentation required for independent detection. Ground truth remains in the benchmark evaluation record.

## Core design principles

1. **Reproducibility** — seeded generation produces identical challenge instances.
2. **Separation of concerns** — generation and detection remain independent systems.
3. **Observable phenotypes** — outputs are expressed as abstract host-response features.
4. **Machine-readable handoff** — challenges can be wrapped in versioned CardiVex contracts.
5. **Manifested challenge sets** — batches retain order, identity, seed, and generator version.
6. **Blinded evaluation** — CardiVex can be evaluated without receiving the domain label.
7. **Controlled difficulty** — overlap, heterogeneity, noise, and temporal behavior can make cases progressively harder.
8. **Benchmarkability** — challenge instances support detection and characterization benchmarks.
9. **Safe abstraction boundary** — no operational biological construction parameters are generated.

## Package

The Python package defines:

- `ChallengeDomain` — controlled challenge categories.
- `PhenotypeProfile` — normalized host-observable feature vector.
- `ChallengeAgent` — complete serializable challenge instance.
- `ChallengeGenerator` — deterministic detailed scenario generator.
- `CardiVexHandoff` / `create_handoff` — trusted downstream envelope.
- `BlindCardiVexHandoff` / `create_blind_handoff` — detection-only envelope.
- `ChallengeManifest` / `build_manifest` — reproducible batch container.
- `BlindBenchmark` / `build_blind_benchmark` — reproducible blinded benchmark set.

## Roadmap

- Formal contract compatibility and schema validation
- richer controlled scenario families
- independent CardiVex observation/result schema
- blinded detection/characterization benchmark protocol
- provenance and audit metadata
- automated schema validation and CI
- AI-assisted challenge sampling and benchmark analysis
