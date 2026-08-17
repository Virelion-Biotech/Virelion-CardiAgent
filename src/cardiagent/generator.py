"""Deterministic generation of detailed, phenotype-level challenge agents."""

from dataclasses import replace
import random

from .models import ChallengeAgent, ChallengeDomain, PhenotypeProfile


_BASE_PROFILES: dict[ChallengeDomain, PhenotypeProfile] = {
    ChallengeDomain.ISCHEMIC: PhenotypeProfile(stress=.80, contractile_impairment=.70, viability_loss=.45, metabolic_disruption=.75),
    ChallengeDomain.INFLAMMATORY: PhenotypeProfile(inflammation=.85, oxidative_stress=.55, remodeling_signal=.65, stress=.45),
    ChallengeDomain.ELECTROPHYSIOLOGIC: PhenotypeProfile(electrical_instability=.90, contractile_impairment=.35, stress=.40),
    ChallengeDomain.TOXIC_INJURY: PhenotypeProfile(stress=.75, viability_loss=.70, oxidative_stress=.80, contractile_impairment=.55),
    ChallengeDomain.VIRAL_LIKE: PhenotypeProfile(stress=.55, inflammation=.70, viability_loss=.35, metabolic_disruption=.40),
    ChallengeDomain.METABOLIC: PhenotypeProfile(metabolic_disruption=.85, oxidative_stress=.60, contractile_impairment=.45, stress=.50),
    ChallengeDomain.GENETIC_SUSCEPTIBILITY: PhenotypeProfile(stress=.30, electrical_instability=.35, metabolic_disruption=.35, remodeling_signal=.30),
}

_DOMAIN_NEIGHBORS: dict[ChallengeDomain, tuple[ChallengeDomain, ...]] = {
    ChallengeDomain.ISCHEMIC: (ChallengeDomain.METABOLIC, ChallengeDomain.TOXIC_INJURY),
    ChallengeDomain.INFLAMMATORY: (ChallengeDomain.VIRAL_LIKE, ChallengeDomain.TOXIC_INJURY),
    ChallengeDomain.ELECTROPHYSIOLOGIC: (ChallengeDomain.GENETIC_SUSCEPTIBILITY, ChallengeDomain.METABOLIC),
    ChallengeDomain.TOXIC_INJURY: (ChallengeDomain.ISCHEMIC, ChallengeDomain.INFLAMMATORY),
    ChallengeDomain.VIRAL_LIKE: (ChallengeDomain.INFLAMMATORY, ChallengeDomain.METABOLIC),
    ChallengeDomain.METABOLIC: (ChallengeDomain.ISCHEMIC, ChallengeDomain.ELECTROPHYSIOLOGIC),
    ChallengeDomain.GENETIC_SUSCEPTIBILITY: (ChallengeDomain.ELECTROPHYSIOLOGIC, ChallengeDomain.METABOLIC),
}


class ChallengeGenerator:
    """Create reproducible, increasingly difficult phenotype-level scenarios.

    The generator produces abstract host-response trajectories, heterogeneity,
    ambiguity and measurement characteristics. It never generates sequences,
    protocols, concentrations, culture conditions, or other operational
    biological instructions.
    """

    VERSION = "0.2"
    SCENARIO_VERSION = "1"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._rng = random.Random(seed)
        self._counter = 0

    def generate(
        self,
        domain: ChallengeDomain,
        *,
        severity: float = 0.5,
        agent_id: str | None = None,
        difficulty: float | None = None,
    ) -> ChallengeAgent:
        if not 0.0 <= severity <= 1.0:
            raise ValueError("severity must be within [0, 1]")
        if difficulty is not None and not 0.0 <= difficulty <= 1.0:
            raise ValueError("difficulty must be within [0, 1]")

        self._counter += 1
        rng = self._rng
        difficulty = severity if difficulty is None else difficulty
        base = _BASE_PROFILES[domain]

        def clip(value: float) -> float:
            return max(0.0, min(1.0, value))

        def vary(value: float, spread: float = 0.10) -> float:
            return clip(value * (1.0 - spread + 2.0 * spread * rng.random()))

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

        onset = rng.uniform(0.05, 0.45)
        persistence = rng.uniform(0.25, 0.95)
        heterogeneity = clip(rng.uniform(0.05, 0.85) + 0.20 * difficulty)
        overlap = clip(0.15 + 0.65 * difficulty + rng.uniform(-0.10, 0.10))
        noise = clip(0.05 + 0.35 * difficulty + rng.uniform(-0.04, 0.04))
        missingness = clip(0.02 + 0.16 * difficulty + rng.uniform(-0.02, 0.02))

        # A deterministic temporal trajectory gives CardiVex something richer
        # to reason over than one static phenotype vector.
        phase_weights = (
            max(0.15, onset),
            clip(0.55 + 0.45 * severity),
            clip(0.80 + 0.20 * persistence),
            clip(0.60 + 0.30 * persistence),
            clip(0.35 + 0.40 * persistence),
        )
        phases = ("baseline", "early", "peak", "persistent", "recovery")
        trajectory: list[dict[str, object]] = []
        phenotype_dict = phenotype.__dict__
        for phase, weight in zip(phases, phase_weights):
            values = {
                key: round(clip(value * weight + rng.uniform(-noise, noise) * 0.20), 6)
                for key, value in phenotype_dict.items()
            }
            trajectory.append({"phase": phase, "relative_time": phases.index(phase) / 4.0, "phenotype": values})

        # Difficult cases deliberately borrow a subset of signals associated
        # with a neighboring domain. This creates realistic phenotype overlap
        # without encoding any operational biological mechanism.
        neighbor = rng.choice(_DOMAIN_NEIGHBORS[domain]).value
        dominant = sorted(phenotype_dict, key=phenotype_dict.get, reverse=True)[:3]
        secondary = sorted(phenotype_dict, key=phenotype_dict.get, reverse=True)[3:6]

        cell_context = rng.sample(
            ["cardiomyocyte", "cardiac_supporting_cell", "multicellular_cardiac_context"],
            k=2,
        )
        confounders = [
            "phenotype_overlap" if overlap >= 0.40 else "limited_overlap",
            "temporal_sampling_sensitivity" if onset < 0.25 else "broad_temporal_signal",
            "cell_population_heterogeneity" if heterogeneity >= 0.45 else "relatively_uniform_response",
        ]
        if noise >= 0.25:
            confounders.append("measurement_noise")
        if missingness >= 0.10:
            confounders.append("partial_observation")

        metadata = {
            "generator": "virelion-cardiagent",
            "generator_version": self.VERSION,
            "scenario_version": self.SCENARIO_VERSION,
            "representation": "phenotype-level",
            "sequence_index": self._counter,
            "difficulty": round(difficulty, 6),
            "scenario_family": f"{domain.value}-v{self.SCENARIO_VERSION}",
            "temporal_profile": trajectory,
            "cell_context": cell_context,
            "dominant_signals": dominant,
            "secondary_signals": secondary,
            "phenotype_overlap": round(overlap, 6),
            "overlap_reference": neighbor,
            "measurement_noise": round(noise, 6),
            "partial_observation_rate": round(missingness, 6),
            "confounders": confounders,
            "expected_observables": dominant + secondary,
            "benchmark_intent": "independent_detection_and_characterization",
        }

        return ChallengeAgent(
            agent_id=agent_id or f"CA-{domain.value}-{self.seed:06d}-{self._counter:04d}",
            domain=domain,
            version=self.VERSION,
            seed=self.seed,
            severity=severity,
            onset=onset,
            persistence=persistence,
            heterogeneity=heterogeneity,
            phenotype=phenotype,
            metadata=metadata,
        )
