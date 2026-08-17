"""Deterministic generation of abstract cardiac challenge instances."""

from dataclasses import replace
import random

from .models import ChallengeAgent, ChallengeDomain, PhenotypeProfile


_BASE_PROFILES: dict[ChallengeDomain, PhenotypeProfile] = {
    ChallengeDomain.ISCHEMIC: PhenotypeProfile(stress=.8, contractile_impairment=.7, viability_loss=.45, metabolic_disruption=.75),
    ChallengeDomain.INFLAMMATORY: PhenotypeProfile(inflammation=.85, oxidative_stress=.55, remodeling_signal=.65, stress=.45),
    ChallengeDomain.ELECTROPHYSIOLOGIC: PhenotypeProfile(electrical_instability=.9, contractile_impairment=.35, stress=.4),
    ChallengeDomain.TOXIC_INJURY: PhenotypeProfile(stress=.75, viability_loss=.7, oxidative_stress=.8, contractile_impairment=.55),
    ChallengeDomain.VIRAL_LIKE: PhenotypeProfile(stress=.55, inflammation=.7, viability_loss=.35, metabolic_disruption=.4),
    ChallengeDomain.METABOLIC: PhenotypeProfile(metabolic_disruption=.85, oxidative_stress=.6, contractile_impairment=.45, stress=.5),
    ChallengeDomain.GENETIC_SUSCEPTIBILITY: PhenotypeProfile(stress=.3, electrical_instability=.35, metabolic_disruption=.35, remodeling_signal=.3),
}


class ChallengeGenerator:
    """Create reproducible phenotype-level challenge agents.

    This generator deliberately contains no pathogen design, sequence design,
    culture instructions, dosing instructions, or other wet-lab operational
    procedures. Its output is a standardized challenge representation for
    downstream detection and benchmarking.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def generate(
        self,
        domain: ChallengeDomain,
        *,
        severity: float = 0.5,
        agent_id: str | None = None,
    ) -> ChallengeAgent:
        if not 0.0 <= severity <= 1.0:
            raise ValueError("severity must be within [0, 1]")

        rng = random.Random(self.seed)
        base = _BASE_PROFILES[domain]

        def vary(value: float) -> float:
            # Small deterministic variation prevents every instance in a
            # domain from being identical while remaining bounded.
            return max(0.0, min(1.0, value * (0.9 + 0.2 * rng.random())))

        phenotype = replace(
            base,
            stress=vary(base.stress * (0.5 + severity)),
            inflammation=vary(base.inflammation * (0.5 + severity)),
            electrical_instability=vary(base.electrical_instability * (0.5 + severity)),
            contractile_impairment=vary(base.contractile_impairment * (0.5 + severity)),
            viability_loss=vary(base.viability_loss * (0.5 + severity)),
            oxidative_stress=vary(base.oxidative_stress * (0.5 + severity)),
            metabolic_disruption=vary(base.metabolic_disruption * (0.5 + severity)),
            remodeling_signal=vary(base.remodeling_signal * (0.5 + severity)),
        )

        return ChallengeAgent(
            agent_id=agent_id or f"CA-{domain.value}-{self.seed:06d}",
            domain=domain,
            version="0.1",
            seed=self.seed,
            severity=severity,
            onset=rng.random(),
            persistence=rng.random(),
            heterogeneity=rng.random(),
            phenotype=phenotype,
            metadata={"generator": "virelion-cardiagent", "representation": "phenotype-level"},
        )
