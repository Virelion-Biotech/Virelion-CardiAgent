# Virelion-CardiAgent

**A reproducible challenge-agent framework for Virelion's cardiac detection platform.**

CardiAgent is the upstream sibling of **Virelion-CardiVex**. Its job is to create standardized, reproducible, increasingly realistic challenge representations that CardiVex can receive and evaluate against cardiac-cell response measurements.

## Architecture

```text
                         CardiAgent
                            │
                 ┌──────────┴──────────┐
                 │                     │
        deterministic prior      ML agent model
                 │                     │
                 └──────────┬──────────┘
                            ▼
                   Detailed Challenge Agent
                            │
                 ┌──────────┴──────────┐
                 │                     │
          trusted handoff        blind handoff
                 │                     │
                 ▼                     ▼
             CardiVex              CardiVex
                                      │
                               independent call
                                      │
                         detected / partial / missed
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

The deterministic generator is the safe prior and benchmark baseline. The ML layer learns distributions from these phenotype-level agents and produces **new sampled agents**, rather than simply replaying the training examples.

## ML agent generator

The package includes `AgentGeneratorModel`, a conditional variational autoencoder (CVAE) implemented with optional PyTorch. It learns the distribution of phenotype-level `ChallengeAgent` examples conditioned on challenge domain and severity.

The model can:

1. train on a corpus of existing `ChallengeAgent` instances;
2. learn latent variation across phenotype, onset, persistence, and heterogeneity;
3. sample new agents from the learned distribution;
4. increase latent sampling temperature as benchmark difficulty increases;
5. mark generated cases as `ml_generated` with model provenance;
6. directly package ML-generated cases into a blinded CardiVex benchmark.

Install the optional ML dependency with:

```bash
pip install -e '.[ml]'
```

Example:

```python
from cardiagent import (
    ChallengeDomain,
    ChallengeGenerator,
    AgentGeneratorModel,
    generate_ml_benchmark,
)

# Build a safe phenotype-level training corpus.
generator = ChallengeGenerator(seed=7)
training_agents = [
    generator.generate(domain, severity=severity, difficulty=difficulty)
    for domain in ChallengeDomain
    for severity in (0.2, 0.4, 0.6, 0.8)
    for difficulty in (0.3, 0.6, 0.9)
]

# Learn the challenge distribution.
model = AgentGeneratorModel(seed=11).fit(
    training_agents,
    epochs=250,
)

# Produce novel ML-generated cases.
new_cases = model.sample(
    domain=ChallengeDomain.INFLAMMATORY,
    severity=0.75,
    difficulty=0.95,
    count=32,
)

# Or generate all domains and package them directly for blind CardiVex evaluation.
benchmark = generate_ml_benchmark(
    model,
    benchmark_id="ml-stress-test-001",
    difficulty=0.95,
    per_domain=32,
)

print(benchmark.public_json())
```

The resulting objects are ordinary `ChallengeAgent` instances, so they use the same CardiVex handoff and blinded benchmark contracts as deterministic cases.

The ML model operates only over abstract phenotype-level features. It does not generate pathogen sequences, biological construction instructions, wet-lab protocols, culture conditions, doses, or other operational parameters.

## Initial challenge domains

- ischemic
- inflammatory
- electrophysiologic
- toxic injury
- viral-like
- metabolic
- genetic susceptibility

These are phenotype-level categories.

## Blind benchmarking

For detection benchmarking, use `build_blind_benchmark()` or `create_blind_handoff()`.

The public presentation contains the observable challenge representation but omits the challenge domain and other direct ground-truth labels. The evaluator record retains the true domain, severity, scenario family, difficulty, and overlap reference separately.

This enables a clean evaluation loop:

```text
challenge truth
     │
     ├──► blinded presentation ──► CardiVex ──► prediction
     │                                      │
     └──────── private truth ───────────────┴──► scoring
```

CardiAgent therefore does **not** decide whether a challenge is detectable. It creates the challenge and preserves the ground truth so CardiVex can be tested independently.

## Deterministic example

```python
from cardiagent import ChallengeDomain, ChallengeGenerator, build_blind_benchmark

generator = ChallengeGenerator(seed=42)
challenges = [
    generator.generate(ChallengeDomain.ISCHEMIC, severity=0.7, difficulty=0.85),
    generator.generate(ChallengeDomain.INFLAMMATORY, severity=0.45, difficulty=0.75),
]

benchmark = build_blind_benchmark(challenges, benchmark_id="demo-001", seed=42)
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
8. **ML sampling** — a learned latent distribution can produce novel challenge agents for stress testing.
9. **Benchmarkability** — challenge instances support detection and characterization benchmarks.
10. **Safe abstraction boundary** — no operational biological construction parameters are generated.

## Package

The Python package defines:

- `ChallengeDomain` — controlled challenge categories.
- `PhenotypeProfile` — normalized host-observable feature vector.
- `ChallengeAgent` — complete serializable challenge instance.
- `ChallengeGenerator` — deterministic detailed scenario generator.
- `AgentGeneratorModel` — conditional VAE for learned challenge-agent generation.
- `train_agent_model` / `generate_ml_agents` — ML training and sampling helpers.
- `generate_ml_benchmark` — ML generation directly into a blind benchmark.
- `CardiVexHandoff` / `create_handoff` — trusted downstream envelope.
- `BlindCardiVexHandoff` / `create_blind_handoff` — detection-only envelope.
- `ChallengeManifest` / `build_manifest` — reproducible batch container.
- `BlindBenchmark` / `build_blind_benchmark` — reproducible blinded benchmark set.

## Roadmap

- richer controlled scenario families
- independent CardiVex observation/result schema
- blinded detection/characterization benchmark protocol
- provenance and audit metadata
- automated schema validation and CI
- stronger learned challenge distributions as real phenotype-level training data become available
