"""Scientific admission gate. Failed CVAE models remain available for research diagnostics only."""

from __future__ import annotations


def require_admissible_generator(challenges):
    for challenge in challenges:
        metadata = challenge.metadata
        family = str(metadata.get("model_family", "")).lower()
        version = str(challenge.version).lower()
        if (
            family in {"cvae", "conditional_variational_autoencoder"}
            or "cvae" in version
            or "ml-cvae" in version
        ):
            raise ValueError(
                "CVAE output is disabled for benchmark/validation handoff; "
                "locked fidelity, real-test utility and inferential gates have not passed"
            )
        if metadata.get("ml_generated") and metadata.get("quality_status") != "qualified":
            raise ValueError("Unqualified learned generator cannot feed scientific benchmarks")
