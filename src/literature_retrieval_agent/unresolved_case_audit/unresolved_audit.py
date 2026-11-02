from __future__ import annotations

from pathlib import Path

from .decision_projection import project_decision
from .hpa_support import marker_has_hpa_support
from .literature_support import marker_has_literature_support
from .marker_support_sources import marker_has_structured_support
from .schemas import ClusterAuditResult, MarkerSupport


def audit_unresolved_cluster(
    dataset: str,
    cluster: str,
    original_decision: str,
    original_labels: str,
    agent2_json: Path,
    rank_coverage_json: Path,
    genes_to_check: list[str],
    candidate_terms: list[str],
) -> ClusterAuditResult:
    marker_support = []

    for gene in genes_to_check:
        marker_support.append(
            MarkerSupport(
                gene=gene,
                supported_by_cellmarker_panglaodb=marker_has_structured_support(
                    agent2_json, gene, candidate_terms
                ),
                supported_by_hpa=marker_has_hpa_support(agent2_json, gene, candidate_terms),
                supported_by_literature=marker_has_literature_support(rank_coverage_json, gene),
            )
        )

    projected_decision, projected_labels, reason = project_decision(
        original_decision=original_decision,
        original_labels=original_labels,
        marker_support=marker_support,
    )

    return ClusterAuditResult(
        dataset=dataset,
        cluster=cluster,
        original_decision=original_decision,
        original_labels=original_labels,
        projected_decision=projected_decision,
        projected_labels=projected_labels,
        rescue_trigger_changed=(projected_decision != original_decision),
        final_decision_changed=(projected_decision != original_decision),
        reason=reason,
    )
