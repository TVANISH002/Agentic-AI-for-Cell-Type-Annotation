from __future__ import annotations

import pytest

from marker_interpretation_agent_v2.interpretation.evidence_classifier import (
    EvidenceClassifier,
)
from marker_interpretation_agent_v2.schemas import (
    HPAEvidenceSummary,
    InputGene,
    PanelEvidenceCatalog,
    PanelGeneEvidence,
)


def _input_gene(
    gene: str,
    *,
    rank: int = 1,
    log_fold_change: float = 2.0,
) -> InputGene:
    """Create one cluster marker input."""

    return InputGene(
        gene=gene,
        rank_within_cluster=rank,
        log_fold_change=log_fold_change,
    )


def test_direct_marker_support_has_priority_over_hpa_pattern() -> None:
    """Keep curated marker support as the primary category."""

    panel_gene = PanelGeneEvidence(
        gene="EPCAM",
        official_symbol="EPCAM",
        context_matched_cell_types=[
            "Luminal epithelial cell"
        ],
        context_matched_database_evidence=[
            {
                "source": "CellMarker",
                "cell_type": "Luminal epithelial cell",
            },
            {
                "source": "PanglaoDB",
                "cell_type": "Luminal epithelial cell",
            },
        ],
        evidence_sources=[
            "CellMarker",
            "PanglaoDB",
        ],
        hpa=HPAEvidenceSummary(
            queried=True,
            gene_present_in_hpa=True,
            breast_expression_found=True,
            top_cell_type="breast glandular cells",
            top_ncpm=10.0,
            second_cell_type="luminal cells",
            second_ncpm=9.5,
            top_to_second_ncpm_ratio=1.05,
            detected_cell_type_count=2,
            highest_read_count=100.0,
        ),
    )

    result = EvidenceClassifier().classify_gene(
        _input_gene("EPCAM"),
        panel_gene,
    )

    assert (
        result.evidence_category
        == "direct_marker_supported"
    )
    assert result.direct_marker_supported is True

    assert result.supported_cell_types == [
        "Luminal epithelial cell"
    ]

    assert (
        result.statistics.source_agreement_count
        == 2
    )

    assert (
        result.statistics.marker_specificity_score
        == pytest.approx(1.0)
    )


def test_hpa_only_gene_is_expression_only() -> None:
    """Do not convert HPA-only expression into a marker vote."""

    panel_gene = PanelGeneEvidence(
        gene="GENE2",
        official_symbol="GENE2",
        hpa=HPAEvidenceSummary(
            queried=True,
            gene_present_in_hpa=True,
            breast_expression_found=True,
            top_cell_type="fibroblasts",
            top_ncpm=5.0,
            detected_cell_type_count=3,
            highest_read_count=20.0,
        ),
    )

    result = EvidenceClassifier().classify_gene(
        _input_gene("GENE2"),
        panel_gene,
    )

    assert (
        result.evidence_category
        == "expression_only"
    )

    assert (
        result.direct_marker_supported
        is False
    )

    assert result.supported_cell_types == []

    assert (
        result.statistics.source_agreement_count
        == 0
    )

    assert (
        result.statistics.marker_specificity_score
        == pytest.approx(0.0)
    )


def test_missing_hpa_symbol_requires_symbol_review() -> None:
    """Route an exact HPA symbol miss to symbol review."""

    panel_gene = PanelGeneEvidence(
        gene="TEST1",
        official_symbol="TEST1",
        hpa=HPAEvidenceSummary(
            queried=True,
            gene_present_in_hpa=False,
            breast_expression_found=False,
        ),
    )

    result = EvidenceClassifier().classify_gene(
        _input_gene("TEST1"),
        panel_gene,
    )

    assert (
        result.evidence_category
        == "symbol_review_needed"
    )

    assert (
        result.hpa.review_pattern
        == "symbol_not_found"
    )

    assert any(
        "authoritative"
        in warning.casefold()
        for warning in result.warnings
    )


def test_trace_expression_uses_configured_thresholds() -> None:
    """Classify very weak HPA breast expression as trace."""

    panel_gene = PanelGeneEvidence(
        gene="TEST2",
        official_symbol="TEST2",
        hpa=HPAEvidenceSummary(
            queried=True,
            gene_present_in_hpa=True,
            breast_expression_found=True,
            top_cell_type="immune cells",
            top_ncpm=0.10,
            detected_cell_type_count=1,
            highest_read_count=1.0,
        ),
    )

    result = EvidenceClassifier().classify_gene(
        _input_gene("TEST2"),
        panel_gene,
    )

    assert (
        result.evidence_category
        == "trace_expression"
    )

    assert (
        result.hpa.review_pattern
        == "trace_expression"
    )


def test_broad_expression_requires_many_cell_types() -> None:
    """Classify distributed HPA expression as broad."""

    panel_gene = PanelGeneEvidence(
        gene="TEST3",
        official_symbol="TEST3",
        hpa=HPAEvidenceSummary(
            queried=True,
            gene_present_in_hpa=True,
            breast_expression_found=True,
            top_cell_type="fibroblasts",
            top_ncpm=8.0,
            second_cell_type="endothelial cells",
            second_ncpm=6.2,
            top_to_second_ncpm_ratio=1.30,
            detected_cell_type_count=12,
            highest_read_count=50.0,
        ),
    )

    result = EvidenceClassifier().classify_gene(
        _input_gene("TEST3"),
        panel_gene,
    )

    assert (
        result.evidence_category
        == "broad_expression"
    )

    assert (
        result.hpa.review_pattern
        == "broad_expression"
    )


def test_classify_rejects_gene_missing_from_catalog() -> None:
    """Require every cluster gene to exist in the panel catalog."""

    catalog = PanelEvidenceCatalog(
        species="Human",
        tissue="Breast",
        platform="Xenium",
        panel_gene_count=1,
        genes=[
            PanelGeneEvidence(
                gene="EPCAM",
                official_symbol="EPCAM",
                hpa=HPAEvidenceSummary(
                    queried=True,
                    gene_present_in_hpa=True,
                    breast_expression_found=True,
                    top_cell_type=(
                        "breast glandular cells"
                    ),
                    top_ncpm=10.0,
                ),
            )
        ],
        unresolved_genes=[],
    )

    with pytest.raises(
        ValueError,
        match=(
            "(?i)missing from the "
            "panel evidence catalog"
        ),
    ):
        EvidenceClassifier().classify(
            [
                _input_gene(
                    "GATA3"
                )
            ],
            catalog,
        )