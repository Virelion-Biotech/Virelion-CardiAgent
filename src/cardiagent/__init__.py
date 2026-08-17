"""Virelion CardiAgent: reproducible cardiac challenge specifications."""

from .adaptive import AdaptiveChallengeEngine, AdaptiveScore, CurriculumStage, DetectionOutcome
from .benchmark import BlindBenchmark, BlindCase, build_blind_benchmark
from .generator import ChallengeGenerator
from .handoff import BlindCardiVexHandoff, CardiVexHandoff, create_blind_handoff, create_handoff
from .literature import LITERATURE_ARCHETYPES, LiteratureArchetype, archetypes_by_domain, get_archetype
from .manifest import ChallengeManifest, build_manifest
from .ml import AgentGeneratorModel, generate_ml_agents, generate_ml_benchmark, train_agent_model
from .models import ChallengeAgent, ChallengeDomain, PhenotypeProfile
from .pathogens import PATHOGEN_PHENOTYPES, PathogenClass, PathogenPhenotype, get_pathogen_phenotype, pathogen_phenotypes
from .quality import PopulationReport, assess_population

__all__ = [
    "ChallengeAgent", "ChallengeDomain", "PhenotypeProfile", "ChallengeGenerator",
    "CardiVexHandoff", "BlindCardiVexHandoff", "create_handoff", "create_blind_handoff",
    "ChallengeManifest", "build_manifest", "BlindCase", "BlindBenchmark", "build_blind_benchmark",
    "AgentGeneratorModel", "train_agent_model", "generate_ml_agents", "generate_ml_benchmark",
    "DetectionOutcome", "AdaptiveScore", "CurriculumStage", "AdaptiveChallengeEngine",
    "PopulationReport", "assess_population",
    "LiteratureArchetype", "LITERATURE_ARCHETYPES", "archetypes_by_domain", "get_archetype",
    "PathogenClass", "PathogenPhenotype", "PATHOGEN_PHENOTYPES", "pathogen_phenotypes", "get_pathogen_phenotype",
]
