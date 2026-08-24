"""Virelion CardiAgent: reproducible cardiac challenge specifications."""

from .adaptive import AdaptiveChallengeEngine, AdaptiveScore, CurriculumStage, DetectionOutcome
from .benchmark import BlindBenchmark, BlindCase, audit_blind_presentation, build_blind_benchmark
from .benchmark_report import BenchmarkReport, build_report, write_report
from .benchmark_runner import run_all_benchmarks, run_benchmark
from .calibration import CalibrationSummary, monotonicity_score, summarize_calibration
from .evaluation import (
    PopulationMetrics,
    conditional_domain_fidelity,
    conditional_numeric_fidelity,
    pairwise_overlap_rate,
    phenotype_mae,
    population_metrics,
    quality_score,
)
from .experiment import MetricSummary, MultiSeedExperiment, run_multi_seed_experiment
from .experiment_artifact import ARTIFACT_VERSION, experiment_fingerprint, run_and_write_experiment
from .generator import ChallengeGenerator
from .handoff import BlindCardiVexHandoff, CardiVexHandoff, create_blind_handoff, create_handoff
from .literature import LITERATURE_ARCHETYPES, LiteratureArchetype, archetypes_by_domain, get_archetype
from .ml import AgentGeneratorModel, generate_ml_agents, generate_ml_benchmark, train_agent_model
from .model_comparison import GeneratorComparison, build_matched_ml_population, build_reference_population, compare_populations
from .models import ChallengeAgent, ChallengeDomain, PhenotypeProfile
from .pathogens import PATHOGEN_PHENOTYPES, PathogenClass, PathogenPhenotype, get_pathogen_phenotype, pathogen_phenotypes
from .quality import PopulationReport, assess_population
from .suites import (
    BenchmarkSuite,
    available_suites,
    baseline_suite,
    build_suite,
    difficulty_suite,
    heterogeneity_suite,
    ood_suite,
    overlap_suite,
    partial_observation_suite,
    severity_suite,
    temporal_suite,
)

__all__ = [
    "ChallengeAgent", "ChallengeDomain", "PhenotypeProfile", "ChallengeGenerator",
    "CardiVexHandoff", "BlindCardiVexHandoff", "create_handoff", "create_blind_handoff",
    "ChallengeManifest", "build_manifest", "BlindCase", "BlindBenchmark", "build_blind_benchmark", "audit_blind_presentation",
    "AgentGeneratorModel", "train_agent_model", "generate_ml_agents", "generate_ml_benchmark",
    "GeneratorComparison", "compare_populations", "build_reference_population", "build_matched_ml_population",
    "MetricSummary", "MultiSeedExperiment", "run_multi_seed_experiment",
    "ARTIFACT_VERSION", "experiment_fingerprint", "run_and_write_experiment",
    "DetectionOutcome", "AdaptiveScore", "CurriculumStage", "AdaptiveChallengeEngine",
    "PopulationReport", "assess_population",
    "PopulationMetrics", "population_metrics", "phenotype_mae", "conditional_domain_fidelity",
    "conditional_numeric_fidelity", "pairwise_overlap_rate", "quality_score",
    "BenchmarkReport", "build_report", "write_report", "run_benchmark", "run_all_benchmarks",
    "CalibrationSummary", "summarize_calibration", "monotonicity_score",
    "BenchmarkSuite", "baseline_suite", "difficulty_suite", "severity_suite", "overlap_suite",
    "temporal_suite", "heterogeneity_suite", "partial_observation_suite", "ood_suite",
    "build_suite", "available_suites",
    "LiteratureArchetype", "LITERATURE_ARCHETYPES", "archetypes_by_domain", "get_archetype",
    "PathogenClass", "PathogenPhenotype", "PATHOGEN_PHENOTYPES", "pathogen_phenotypes", "get_pathogen_phenotype",
]
