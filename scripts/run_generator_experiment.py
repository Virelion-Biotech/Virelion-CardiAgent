"""Generate the canonical multi-seed deterministic-vs-CVAE experiment artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from cardiagent.experiment_artifact import run_and_write_experiment
from cardiagent.generator import ChallengeGenerator
from cardiagent.models import ChallengeDomain


def build_training_population() -> list:
    """Create the versioned phenotype-only training population."""
    agents = []
    for seed in range(12):
        generator = ChallengeGenerator(seed=10_000 + seed)
        for domain in ChallengeDomain:
            for severity in (0.2, 0.4, 0.6, 0.8):
                agents.append(
                    generator.generate(
                        domain,
                        severity=severity,
                        difficulty=0.2 + 0.1 * (seed % 6),
                        agent_id=f"train-{seed}-{domain.value}-{severity:.1f}",
                    )
                )
    return agents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/generator-experiment.json"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--per-domain", type=int, default=16)
    args = parser.parse_args()

    training = build_training_population()
    output = run_and_write_experiment(
        training,
        output_path=args.output,
        training_epochs=args.epochs,
        per_domain=args.per_domain,
    )
    print(f"wrote {output} from {len(training)} training agents")


if __name__ == "__main__":
    main()
