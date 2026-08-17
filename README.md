# Virelion-CardiAgent

**A reproducible challenge-agent framework for Virelion's cardiac detection platform.**

CardiAgent is the upstream sibling of **Virelion-CardiVex**. Its job is to create standardized, reproducible, increasingly realistic challenge representations that CardiVex can receive and evaluate against cardiac-cell response measurements.

## Architecture

```text
                         CardiAgent
                            │
              ┌─────────────┴─────────────┐
              │                           │
      deterministic prior            ML generator
              │                           │
              └─────────────┬─────────────┘
                            ▼
                  Challenge population
                            │
                    quality / diversity
                            │
                            ▼
                    blind CardiVex
                            │
                      observations
                            │
                            ▼
                   detection outcomes
                            │
                            ▼
                  AdaptiveChallengeEngine
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
          harder cases             curriculum
                │                       │
                └───────────┬───────────┘
                            ▼
                     next benchmark
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

`AgentGeneratorModel` is a conditional variational autoencoder (CVAE) implemented with optional PyTorch. It learns the distribution of phenotype-level `ChallengeAgent` examples conditioned on challenge domain and severity.

It can train on an existing challenge corpus, sample novel latent variants, control difficulty through latent sampling temperature, and package the results directly into a blinded CardiVex benchmark.

```python
model = AgentGeneratorModel(seed=11).fit(training_agents, epochs=250)
new_cases = model.sample(
    domain=ChallengeDomain.INFLAMMATORY,
    severity=0.75,
    difficulty=0.95,
    count=32,
)
```

## Adaptive generation

The ML generator is only the first layer. `AdaptiveChallengeEngine` creates a closed evaluation loop **without requiring access to CardiVex internals**.

CardiVex returns only outcome-level information such as:

```python
DetectionOutcome(
    case_id="...",
    predicted_domain="...",
    confidence=0.61,
    detected=False,
    characterization_correct=False,
)
```

CardiAgent converts those outcomes into hardness signals and can evolve the next generation of phenotype-level challenges.

```python
engine = AdaptiveChallengeEngine(seed=42)
engine.score(outcomes)
stage = engine.next_stage(mean_hardness=0.72)
next_generation = engine.evolve(
    parents=hard_cases,
    count=64,
    stage=stage,
)
```

Evolution uses phenotype-space recombination and bounded mutation. It does **not** optimize operational biological parameters.

This gives CardiAgent a genuine feedback loop:

```text
ML generation
     ↓
CardiVex benchmark
     ↓
CardiVex outcome
     ↓
hardness analysis
     ↓
phenotype-space evolution
     ↓
new challenge population
     ↓
CardiVex
```

## Curriculum learning

Challenges are organized into controlled stages:

1. `baseline`
2. `moderate`
3. `hard`
4. `stress`
5. `edge`

Difficulty increases through abstract challenge properties such as phenotype overlap, response heterogeneity, measurement noise, and partial observation. The engine deliberately avoids jumping straight to maximum difficulty when downstream performance is poor; the benchmark remains diagnostically useful rather than becoming an arbitrary failure generator.

## Population quality control

`assess_population()` provides a quality gate before a generated population is handed to CardiVex. It reports:

- domain balance
- mean severity
- mean difficulty
- phenotype-space diversity
- duplicate rate
- overall quality score
- warnings for collapsed or imbalanced populations

This prevents an ML generator from producing thousands of superficially different but effectively identical cases.

## Blind benchmarking

For detection benchmarking, use `build_blind_benchmark()` or `create_blind_handoff()`.

The public presentation contains the observable challenge representation but omits the challenge domain and other direct ground-truth labels. The evaluator record retains the true domain, severity, scenario family, difficulty, and overlap reference separately.

The evaluator therefore remains independent:

```text
truth ────────────────┐
                      │
blinded presentation → CardiVex → prediction
                      │
                      └──────────→ scoring
```

CardiAgent does **not** decide whether a challenge is detectable. It creates the challenge and preserves the ground truth so CardiVex can be tested independently.

## Package

The Python package defines:

- `ChallengeDomain` — controlled challenge categories.
- `PhenotypeProfile` — normalized host-observable feature vector.
- `ChallengeAgent` — complete serializable challenge instance.
- `ChallengeGenerator` — deterministic detailed scenario generator.
- `AgentGeneratorModel` — conditional VAE for learned challenge-agent generation.
- `train_agent_model` / `generate_ml_agents` — ML training and sampling helpers.
- `generate_ml_benchmark` — ML generation directly into a blind benchmark.
- `DetectionOutcome` — CardiVex outcome contract.
- `AdaptiveChallengeEngine` — outcome-driven challenge evolution and curriculum.
- `CurriculumStage` — controlled difficulty regime.
- `PopulationReport` / `assess_population` — diversity and quality gates.
- `CardiVexHandoff` / `create_handoff` — trusted downstream envelope.
- `BlindCardiVexHandoff` / `create_blind_handoff` — detection-only envelope.
- `ChallengeManifest` / `build_manifest` — reproducible batch container.
- `BlindBenchmark` / `build_blind_benchmark` — reproducible blinded benchmark set.

## Initial challenge domains

- ischemic
- inflammatory
- electrophysiologic
- toxic injury
- viral-like
- metabolic
- genetic susceptibility

These are phenotype-level categories.

## Safety boundary

All generated challenge content stays at the abstract host-response / phenotype level. CardiAgent does not generate pathogen sequences, biological construction instructions, wet-lab protocols, culture conditions, doses, or other operational biological parameters.
