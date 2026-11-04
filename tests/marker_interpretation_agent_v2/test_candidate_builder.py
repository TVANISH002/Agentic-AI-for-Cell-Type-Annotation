from __future__ import annotations

from typing import Any

import pytest

from marker_interpretation_agent_v2.interpretation.candidate_builder import (
    CandidateBuilder,
)
from marker_interpretation_agent_v2.schemas import (
    GeneInterpretation,
    GeneStatistics,
    HPAEvidenceSummary,
)


def _hpa(
    *,
    top_cell_type: str = "breast glandular cells",
    breast_expression_found: bool = True,
) -> HPAEvidenceSummary:
    """Create reusable HPA evidence for candidate tests."""

    return HPAEvidenceSummary(
        queried=True,
        gene_present_in_hpa=True,
        breast_expression_found=breast_expression_found,
        top_cell_type=(
            top_cell_type
            if breast_expression_found
            else None
        ),
        top_ncpm=(
            10.0
            if breast_expression_found
            else None
        ),
        detected_cell_type_count=2,
        highest_read_count=100.0,
    )


def _direct_marker(
    *,
    gene: str,
    cell_types: list[str],
    sources: list[str],
    cluster_rank: int,
    log_fold_change: float,
    context_evidence: list[dict[str, Any]],
) -> GeneInterpretation:
    """Create one direct-marker-supported interpretation."""

    return GeneInterpretation(
        gene=gene,
        direct_marker_supported=True,
        evidence_category="direct_marker_supported",
        supported_cell_types=cell_types,
        context_evidence=context_evidence,
        evidence_sources=sources,
        statistics=GeneStatistics(
            source_agreement_count=len(
                set(sources)
            ),
            marker_specificity_score=(
                1.0 / len(set(cell_types))
            ),
            context_evidence_count=len(
                context_evidence
            ),
            cluster_gene_rank=cluster_rank,
            log_fold_change=log_fold_change,
        ),
        hpa=_hpa(),
    )


def _expression_only(
    gene: str,
    *,
    cluster_rank: int,
) -> GeneInterpretation:
    """Create one HPA-only interpretation."""

    return GeneInterpretation(
        gene=gene,
        direct_marker_supported=False,
        evidence_category="expression_only",
        supported_cell_types=[],
        context_evidence=[],
        evidence_sources=[],
        statistics=GeneStatistics(
            source_agreement_count=0,
            marker_specificity_score=0.0,
            context_evidence_count=0,
            cluster_gene_rank=cluster_rank,
            log_fold_change=4.0,
        ),
        hpa=_hpa(
            top_cell_type="fibroblasts"
        ),
    )


def test_aggregates_multiple_genes_for_same_candidate() -> None:
    """Combine direct-marker genes supporting one candidate."""

    epcam = _direct_marker(
        gene="EPCAM",
        cell_types=[
            "Luminal epithelial cell"
        ],
        sources=[
            "CellMarker",
            "PanglaoDB",
        ],
        cluster_rank=4,
        log_fold_change=2.8,
        context_evidence=[
            {
                "source": "CellMarker",
                "cell_type": (
                    "Luminal epithelial cell"
                ),
                "tissue": "Breast",
                "disease_context": (
                    "Breast Cancer"
                ),
                "cell_ontology_id": (
                    "CL:0002326"
                ),
            },
            {
                "source": "PanglaoDB",
                "cell_type": (
                    "Luminal epithelial cell"
                ),
                "tissue": "Mammary gland",
            },
        ],
    )

    gata3 = _direct_marker(
        gene="GATA3",
        cell_types=[
            "Luminal epithelial cell"
        ],
        sources=["CellMarker"],
        cluster_rank=2,
        log_fold_change=3.0,
        context_evidence=[
            {
                "source": "CellMarker",
                "cell_type": (
                    "Luminal epithelial cell"
                ),
                "tissue": "Breast",
            }
        ],
    )

    candidates = CandidateBuilder().build(
        [
            epcam,
            gata3,
        ]
    )

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.candidate_rank == 1

    assert (
        candidate.cell_type
        == "Luminal epithelial cell"
    )

    assert (
        candidate.unique_supporting_gene_count
        == 2
    )

    assert candidate.supporting_genes == [
        "EPCAM",
        "GATA3",
    ]

    assert (
        candidate
        .both_source_supporting_gene_count
        == 1
    )

    assert (
        candidate
        .both_source_supporting_genes
        == ["EPCAM"]
    )

    assert (
        candidate.specificity_weighted_support
        == pytest.approx(2.0)
    )

    assert (
        candidate.mean_supporting_gene_rank
        == pytest.approx(3.0)
    )

    assert (
        candidate
        .mean_supporting_log_fold_change
        == pytest.approx(2.9)
    )

    assert candidate.evidence_count == 3

    assert candidate.evidence_sources == [
        "CellMarker",
        "PanglaoDB",
    ]

    assert candidate.tissue_contexts == [
        "Breast",
        "Mammary gland",
    ]

    assert candidate.disease_contexts == [
        "Breast Cancer"
    ]

    assert candidate.cell_ontology_ids == [
        "CL:0002326"
    ]

    assert (
        candidate.hpa_queried_gene_count
        == 2
    )

    assert (
        candidate
        .hpa_breast_expression_gene_count
        == 2
    )


def test_hpa_only_gene_does_not_create_candidate() -> None:
    """Exclude HPA-only genes from candidate marker voting."""

    hpa_only = _expression_only(
        "GENE2",
        cluster_rank=1,
    )

    candidates = CandidateBuilder().build(
        [hpa_only]
    )

    assert candidates == []


def test_candidate_ranking_prioritizes_more_supporting_genes() -> None:
    """Rank the candidate with more direct-marker genes first."""

    epcam = _direct_marker(
        gene="EPCAM",
        cell_types=[
            "Luminal epithelial cell"
        ],
        sources=["CellMarker"],
        cluster_rank=1,
        log_fold_change=3.0,
        context_evidence=[
            {
                "source": "CellMarker",
                "cell_type": (
                    "Luminal epithelial cell"
                ),
            }
        ],
    )

    gata3 = _direct_marker(
        gene="GATA3",
        cell_types=[
            "Luminal epithelial cell"
        ],
        sources=["CellMarker"],
        cluster_rank=2,
        log_fold_change=2.8,
        context_evidence=[
            {
                "source": "CellMarker",
                "cell_type": (
                    "Luminal epithelial cell"
                ),
            }
        ],
    )

    col1a1 = _direct_marker(
        gene="COL1A1",
        cell_types=["Fibroblast"],
        sources=[
            "CellMarker",
            "PanglaoDB",
        ],
        cluster_rank=3,
        log_fold_change=2.5,
        context_evidence=[
            {
                "source": "CellMarker",
                "cell_type": "Fibroblast",
            },
            {
                "source": "PanglaoDB",
                "cell_type": "Fibroblast",
            },
        ],
    )

    candidates = CandidateBuilder().build(
        [
            col1a1,
            gata3,
            epcam,
        ]
    )

    assert [
        candidate.cell_type
        for candidate in candidates
    ] == [
        "Luminal epithelial cell",
        "Fibroblast",
    ]

    assert [
        candidate.candidate_rank
        for candidate in candidates
    ] == [
        1,
        2,
    ]

    assert (
        candidates[0]
        .unique_supporting_gene_count
        == 2
    )

    assert (
        candidates[1]
        .unique_supporting_gene_count
        == 1
    )


def test_merges_cell_type_labels_case_insensitively() -> None:
    """Treat capitalization differences as one candidate."""

    gene1 = _direct_marker(
        gene="GENE1",
        cell_types=["Fibroblast"],
        sources=["CellMarker"],
        cluster_rank=1,
        log_fold_change=2.5,
        context_evidence=[
            {
                "source": "CellMarker",
                "cell_type": "Fibroblast",
            }
        ],
    )

    gene2 = _direct_marker(
        gene="GENE2",
        cell_types=["fibroblast"],
        sources=["PanglaoDB"],
        cluster_rank=2,
        log_fold_change=2.0,
        context_evidence=[
            {
                "source": "PanglaoDB",
                "cell_type": "fibroblast",
            }
        ],
    )

    candidates = CandidateBuilder().build(
        [
            gene1,
            gene2,
        ]
    )

    assert len(candidates) == 1

    assert (
        candidates[0]
        .unique_supporting_gene_count
        == 2
    )

    assert candidates[0].supporting_genes == [
        "GENE1",
        "GENE2",
    ]


def test_rejects_duplicate_gene_interpretations() -> None:
    """Reject duplicate interpretations for the same gene."""

    interpretation = _direct_marker(
        gene="EPCAM",
        cell_types=[
            "Luminal epithelial cell"
        ],
        sources=["CellMarker"],
        cluster_rank=1,
        log_fold_change=3.0,
        context_evidence=[
            {
                "source": "CellMarker",
                "cell_type": (
                    "Luminal epithelial cell"
                ),
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="(?i)duplicate",
    ):
        CandidateBuilder().build(
            [
                interpretation,
                interpretation,
            ]
        )