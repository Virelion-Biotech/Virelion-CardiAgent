"""Pathogen-associated challenge archetypes.

This module deliberately models only clinically observed pathogen-associated
cardiac response phenotypes. It contains no pathogen sequences, engineering
parameters, propagation methods, inoculation procedures, doses, or other
operational instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .models import ChallengeDomain


class PathogenClass(str, Enum):
    VIRAL = "viral"
    BACTERIAL = "bacterial"
    PARASITIC = "parasitic"
    FUNGAL = "fungal"
    POST_INFECTIOUS = "post_infectious"


@dataclass(frozen=True)
class PathogenPhenotype:
    name: str
    pathogen_class: PathogenClass
    domain: ChallengeDomain
    clinical_context: str
    cardiac_phenotypes: tuple[str, ...]
    mechanisms: tuple[str, ...]
    temporal_pattern: str
    relevant_cell_contexts: tuple[str, ...]
    confounders: tuple[str, ...]
    literature: tuple[str, ...]


PATHOGEN_PHENOTYPES = (
    PathogenPhenotype(
        "viral_myocarditis_like",
        PathogenClass.VIRAL,
        ChallengeDomain.VIRAL_LIKE,
        "infection-associated myocardial inflammation and dysfunction",
        ("inflammation", "contractile impairment", "electrical instability", "cell stress", "viability loss"),
        ("innate immune activation", "cytokine signaling", "immune-mediated injury", "myocyte stress"),
        "acute inflammatory phase with variable recovery or persistent dysfunction",
        ("cardiomyocyte", "immune", "endothelial", "fibroblast"),
        ("ischemia", "autoimmune inflammation", "toxic injury"),
        ("PMID:41198069", "PMID:34820429"),
    ),
    PathogenPhenotype(
        "systemic_inflammation_cardiac",
        PathogenClass.BACTERIAL,
        ChallengeDomain.INFLAMMATORY,
        "systemic infection-associated cardiac stress phenotype",
        ("inflammation", "endothelial stress", "contractile impairment", "metabolic disruption", "electrical instability"),
        ("systemic inflammatory signaling", "endothelial dysfunction", "microvascular stress", "metabolic demand mismatch"),
        "rapid systemic inflammatory escalation followed by recovery or remodeling",
        ("endothelial", "cardiomyocyte", "immune", "fibroblast"),
        ("primary myocardial inflammation", "ischemia", "metabolic stress"),
        ("PMID:41198069",),
    ),
    PathogenPhenotype(
        "post_infectious_inflammatory",
        PathogenClass.POST_INFECTIOUS,
        ChallengeDomain.INFLAMMATORY,
        "delayed cardiac inflammatory phenotype following an infectious trigger",
        ("persistent inflammation", "electrical instability", "contractile impairment", "remodeling signal"),
        ("immune dysregulation", "persistent inflammatory signaling", "vascular dysfunction"),
        "delayed onset relative to the initiating infectious episode with heterogeneous persistence",
        ("immune", "endothelial", "cardiomyocyte", "fibroblast"),
        ("active infection", "autoimmune disease", "ischemic remodeling"),
        ("PMID:41198069",),
    ),
    PathogenPhenotype(
        "endothelial_inflammatory_stress",
        PathogenClass.VIRAL,
        ChallengeDomain.INFLAMMATORY,
        "infection-associated endothelial and microvascular dysfunction phenotype",
        ("endothelial stress", "inflammation", "oxidative stress", "microvascular dysfunction", "contractile impairment"),
        ("endothelial activation", "immune signaling", "oxidative stress", "microvascular dysfunction"),
        "early vascular stress with severity-dependent downstream myocardial dysfunction",
        ("endothelial", "pericyte", "immune", "cardiomyocyte"),
        ("primary cardiomyocyte injury", "ischemia-reperfusion", "systemic inflammation"),
        ("PMID:41198069",),
    ),
    PathogenPhenotype(
        "parasitic_cardiomyopathy_like",
        PathogenClass.PARASITIC,
        ChallengeDomain.VIRAL_LIKE,
        "chronic infection-associated inflammatory and remodeling phenotype",
        ("chronic inflammation", "fibrosis", "electrical instability", "contractile impairment", "remodeling signal"),
        ("persistent immune activation", "fibrotic remodeling", "electrical remodeling"),
        "prolonged heterogeneous course with delayed remodeling",
        ("cardiomyocyte", "fibroblast", "immune", "endothelial"),
        ("idiopathic fibrosis", "post-infarction remodeling", "primary arrhythmogenic disease"),
        ("PMID:34820429",),
    ),
)


def pathogen_phenotypes() -> tuple[PathogenPhenotype, ...]:
    return PATHOGEN_PHENOTYPES


def get_pathogen_phenotype(name: str) -> PathogenPhenotype:
    for phenotype in PATHOGEN_PHENOTYPES:
        if phenotype.name == name:
            return phenotype
    raise KeyError(f"Unknown pathogen phenotype: {name}")
