from __future__ import annotations

from .schemas import MarkerSupport


def project_decision(
    original_decision: str,
    original_labels: str,
    marker_support: list[MarkerSupport],
    lineage_label: str = "T-cell lineage supported; subtype unresolved",
) -> tuple[str, str, str]:
    """Project what the decision would become under HPA-aware support.

    This does not overwrite Agent 3 output. It only supports an audit report.
    """
    if original_decision.upper() != "UNRESOLVED":
        return original_decision, original_labels, "No projection needed; original decision was not unresolved."

    if any(marker.supported_anywhere for marker in marker_support):
        return (
            "AMBIGUOUS",
            lineage_label,
            "At least one high-ranked marker was supported by supplemental/structured/literature evidence.",
        )

    return (
        "UNRESOLVED",
        original_labels,
        "Markers remained unsupported across structured sources, HPA, and literature evidence.",
    )
