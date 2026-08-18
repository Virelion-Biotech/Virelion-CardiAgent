"""Virelion CardiAgent: reproducible cardiac challenge specifications."""

from .adaptive import AdaptiveChallengeEngine, AdaptiveScore, CurriculumStage, DetectionOutcome
from .benchmark import BlindBenchmark, BlindCase, build_blind_benchmark
from .generator import ChallengeGenerator
from .handoff import BlindCardiVexHandoff, CardiVexHandoff, create_blind_handoff, create_handoff
from .literature import LITERATURE_ARCHETYPES, LiteratureArchetype, archetypes_by_domain, get_archetype
from .manifest import ChallengeManifest, build_manifest
from .metrics import DetectionMetrics, PopulationMetrics, conditional_fidelity, detection_metrics, population_metrics, reproducibility_signature
from .ml import AgentGeneratorModel, generate_ml_agents, generate_ml_benchmark, train_agent_model
from .models import ChallengeAgent, ChallengeDomain, PhenotypeProfile
from .pathogens import PATHOGEN_PHENOTYPES, PathogenClass, PathogenPhenotype, get_pathogen_phenotype, pathogen_phenotypes
from .quality import PopulationReport, assess_population
from .suites import SUITE_VERSIONS, build_all_suites, build_suite

__all__ = [
    "ChallengeAgent", "ChallengeDomain", "PhenotypeProfile", "ChallengeGenerator",
    "CardiVexHandoff", "BlindCardiVexHandoff", "create_handoff", "create_blind_handoff",
    "ChallengeManifest", "build_manifest", "BlindCase", "BlindBenchmark", "build_blind_benchmark",
    "AgentGeneratorModel", "train_agent_model", "generate_ml_agents", "generate_ml_benchmark",
    "DetectionOutcome", "AdaptiveScore", "CurriculumStage", "AdaptiveChallengeEngine",
    "PopulationReport", "assess_population",
    "DetectionMetrics", "PopulationMetrics", "detection_metrics", "population_metrics",
    "conditional_fidelity", "reproducibility_signature",
    "SUITE_VERSIONS", "build_suite", "build_all_suites",
    "LiteratureArchetype", "LITERATURE_ARCHETYPES", "archetypes_by_domain", "get_archetype",
    "PathogenClass", "PathogenPhenotype", "PATHOGEN_PHENOTYPES", "pathogen_phenotypes", "get_pathogen_phenotype",
]
