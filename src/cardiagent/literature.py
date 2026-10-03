"""Literature-grounded phenotype archetypes for CardiAgent.

These archetypes encode published cardiac *response phenotypes*, not operational
ways to create biological injury. They are intended as priors for ML generation
and benchmark construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .models import ChallengeDomain


@dataclass(frozen=True)
class LiteratureArchetype:
    name: str
    domain: ChallengeDomain
    biological_context: str
    mechanisms: tuple[str, ...]
    observable_axes: tuple[str, ...]
    temporal_pattern: str
    cell_contexts: tuple[str, ...]
    confounders: tuple[str, ...]
    literature: tuple[str, ...]


LITERATURE_ARCHETYPES: tuple[LiteratureArchetype, ...] = (
    LiteratureArchetype(
        name="ischemia_reperfusion",
        domain=ChallengeDomain.ISCHEMIC,
        biological_context="acute ischemic/reoxygenation cardiac injury",
        mechanisms=("ATP depletion", "ionic imbalance", "calcium overload", "oxidative stress", "mitochondrial dysfunction", "inflammatory amplification"),
        observable_axes=("viability_loss", "oxidative_stress", "electrical_instability", "contractile_impairment", "metabolic_disruption", "inflammation"),
        temporal_pattern="biphasic injury with acute stress followed by inflammatory/remodeling response",
        cell_contexts=("cardiomyocyte", "endothelial", "fibroblast", "immune"),
        confounders=("inflammation", "oxidative stress", "metabolic disruption"),
        literature=("PMID:39577392", "PMID:41677628", "PMID:41828337"),
    ),
    LiteratureArchetype(
        name="post_infarction_remodeling",
        domain=ChallengeDomain.ISCHEMIC,
        biological_context="post-myocardial-injury remodeling",
        mechanisms=("cell loss", "inflammatory recruitment", "fibroblast activation", "extracellular-matrix remodeling", "electrical uncoupling"),
        observable_axes=("inflammation", "remodeling_signal", "contractile_impairment", "electrical_instability", "viability_loss"),
        temporal_pattern="early inflammatory phase followed by repair and persistent remodeling",
        cell_contexts=("cardiomyocyte", "fibroblast", "endothelial", "hematopoietic"),
        confounders=("acute ischemic stress", "fibrosis", "inflammation"),
        literature=("PMID:34820429", "PMID:42035955", "s44161-025-00617-1"),
    ),
    LiteratureArchetype(
        name="inflammatory_electrical_instability",
        domain=ChallengeDomain.INFLAMMATORY,
        biological_context="inflammation-associated cardiac electrical dysfunction",
        mechanisms=("cytokine signaling", "immune-cell cardiomyocyte interactions", "inflammasome-associated signaling", "electrical remodeling"),
        observable_axes=("inflammation", "electrical_instability", "stress", "contractile_impairment"),
        temporal_pattern="inflammatory escalation with potentially persistent electrical phenotype",
        cell_contexts=("cardiomyocyte", "immune", "endothelial"),
        confounders=("primary electrophysiologic dysfunction", "ischemic injury"),
        literature=("PMID:41198069",),
    ),
    LiteratureArchetype(
        name="fibroblast_driven_fibrosis",
        domain=ChallengeDomain.METABOLIC,
        biological_context="fibroblast-dominant extracellular-matrix remodeling",
        mechanisms=("fibroblast activation", "matrix synthesis", "cell-cell signaling", "scar expansion"),
        observable_axes=("remodeling_signal", "inflammation", "contractile_impairment", "stress"),
        temporal_pattern="delayed accumulation with persistent remodeling phenotype",
        cell_contexts=("fibroblast", "immune", "cardiomyocyte"),
        confounders=("post-infarction inflammation", "pressure-associated remodeling"),
        literature=("s44161-025-00617-1", "s41467-024-52068-0"),
    ),
    LiteratureArchetype(
        name="pressure_hypertrophy",
        domain=ChallengeDomain.METABOLIC,
        biological_context="pressure-associated cardiac hypertrophic remodeling",
        mechanisms=("hypertrophic growth", "cell-state remodeling", "fibroblast activation", "vascular adaptation"),
        observable_axes=("remodeling_signal", "contractile_impairment", "metabolic_disruption", "stress"),
        temporal_pattern="progressive adaptation with chronic remodeling",
        cell_contexts=("cardiomyocyte", "fibroblast", "endothelial", "pericyte", "smooth-muscle", "immune"),
        confounders=("metabolic disease", "fibrosis", "electrical remodeling"),
        literature=("s44161-022-00019-7",),
    ),
    LiteratureArchetype(
        name="oxidative_mitochondrial_stress",
        domain=ChallengeDomain.TOXIC_INJURY,
        biological_context="cardiomyocyte oxidative and mitochondrial stress phenotype",
        mechanisms=("reactive oxygen species", "mitochondrial dysfunction", "oxidative damage", "calcium dysregulation"),
        observable_axes=("oxidative_stress", "metabolic_disruption", "viability_loss", "contractile_impairment", "stress"),
        temporal_pattern="rapid stress response with severity-dependent recovery or persistent dysfunction",
        cell_contexts=("cardiomyocyte", "endothelial"),
        confounders=("ischemia-reperfusion", "inflammation", "metabolic dysfunction"),
        literature=("PMID:40227421", "PMID:34396745"),
    ),
    LiteratureArchetype(
        name="metabolic_remodeling",
        domain=ChallengeDomain.METABOLIC,
        biological_context="cardiac metabolic stress with functional remodeling",
        mechanisms=("energy imbalance", "metabolic reprogramming", "mitochondrial dysfunction", "contractile adaptation"),
        observable_axes=("metabolic_disruption", "oxidative_stress", "contractile_impairment", "stress"),
        temporal_pattern="gradual adaptation with severity-dependent persistence",
        cell_contexts=("cardiomyocyte", "endothelial"),
        confounders=("ischemia", "hypertrophy", "mitochondrial stress"),
        literature=("PMID:34396745", "PMID:39577392"),
    ),
    LiteratureArchetype(
        name="cardiac_cell_state_shift",
        domain=ChallengeDomain.GENETIC_SUSCEPTIBILITY,
        biological_context="altered cardiomyocyte state with dedifferentiation/senescence-like features",
        mechanisms=("cell-state transition", "senescence-associated signaling", "altered electrical coupling", "stress response"),
        observable_axes=("stress", "inflammation", "electrical_instability", "contractile_impairment", "remodeling_signal"),
        temporal_pattern="heterogeneous cell-state transition with persistence in a subset",
        cell_contexts=("cardiomyocyte", "fibroblast", "immune"),
        confounders=("post-infarction remodeling", "inflammation", "hypertrophy"),
        literature=("PMID:34820429",),
    ),
)


def archetypes_by_domain() -> Mapping[ChallengeDomain, tuple[LiteratureArchetype, ...]]:
    result: dict[ChallengeDomain, list[LiteratureArchetype]] = {}
    for archetype in LITERATURE_ARCHETYPES:
        result.setdefault(archetype.domain, []).append(archetype)
    return {domain: tuple(items) for domain, items in result.items()}


def get_archetype(name: str) -> LiteratureArchetype:
    for archetype in LITERATURE_ARCHETYPES:
        if archetype.name == name:
            return archetype
    raise KeyError(f"Unknown literature archetype: {name}")
