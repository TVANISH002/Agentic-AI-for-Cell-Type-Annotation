from __future__ import annotations

from typing import Any

import pytest

from marker_interpretation_agent_v2.panel.panel_catalog_validator import (
    PanelCatalogValidator,
)
from marker_interpretation_agent_v2.panel.panel_evidence_builder import (
    PanelEvidenceBuilder,
)
from marker_interpretation_agent_v2.schemas import (
    PanelEvidenceCatalog,
)


def _agent1_payload() -> dict[str, Any]:
    """Return panel-wide Agent 1 evidence for three genes."""

    return {
        "cluster_id": "panel",
        "species": "Human",
        "tissue": "Breast",
        "platform": "Xenium",
        "input_genes": [
            {"gene": "EPCAM"},
            {"gene": "GENE2"},
            {"gene": "TESTGENE"},
        ],
        "gene_evidence": [
            {
                "gene": "EPCAM",
                "official_symbol": "EPCAM",
                "aliases": ["TACSTD1"],
                "context_matched_cell_types": [
                    "Luminal epithelial cell"
                ],
                "context_matched_database_evidence": [
                    {
                        "source": "CellMarker",
                        "cell_type": "Luminal epithelial cell",
                        "tissue": "Breast",
                    },
                    {
                        "source": "PanglaoDB",
                        "cell_type": "Luminal epithelial cell",
                        "tissue": "Mammary gland",
                    },
                ],
                "database_evidence": [
                    {
                        "source": "CellMarker",
                        "cell_type": "Luminal epithelial cell",
                        "tissue": "Breast",
                    },
                    {
                        "source": "PanglaoDB",
                        "cell_type": "Luminal epithelial cell",
                        "tissue": "Mammary gland",
                    },
                ],
                "evidence_sources": [
                    "CellMarker",
                    "PanglaoDB",
                ],
                "quality_warnings": [],
            },
            {
                "gene": "GENE2",
                "official_symbol": "GENE2",
                "context_matched_cell_types": [],
                "context_matched_database_evidence": [],
                "database_evidence": [],
                "quality_warnings": [],
            },
            {
                "gene": "TESTGENE",
                "official_symbol": "TESTGENE",
                "context_matched_cell_types": [],
                "context_matched_database_evidence": [],
                "database_evidence": [],
                "quality_warnings": [],
            },
        ],
    }


def _hpa_payload() -> dict[str, Any]:
    """Return panel-wide HPA evidence for the same genes."""

    return {
        "source": "Human Protein Atlas",
        "scope": "panel",
        "selected_genes": [
            "EPCAM",
            "GENE2",
            "TESTGENE",
        ],
        "gene_summaries": [
            {
                "gene": "TESTGENE",
                "gene_present_in_hpa": False,
                "breast_expression_found": False,
            },
            {
                "gene": "GENE2",
                "gene_present_in_hpa": True,
                "breast_expression_found": True,
                "top_cell_type": "fibroblasts",
                "top_ncpm": 5.0,
                "detected_cell_type_count": 3,
                "highest_read_count": 20,
            },
            {
                "gene": "EPCAM",
                "gene_present_in_hpa": True,
                "breast_expression_found": True,
                "top_cell_type": "breast glandular cells",
                "top_ncpm": 12.5,
                "detected_cell_type_count": 2,
                "highest_read_count": 100,
            },
        ],
    }


def _build_catalog() -> PanelEvidenceCatalog:
    """Build one validated three-gene panel catalog."""

    return PanelEvidenceBuilder().build(
        _agent1_payload(),
        _hpa_payload(),
        expected_genes=[
            "EPCAM",
            "GENE2",
            "TESTGENE",
        ],
    )


def test_builds_catalog_and_preserves_evidence_boundary() -> None:
    """Keep direct marker support separate from HPA-only evidence."""

    catalog = _build_catalog()

    records = {
        record.gene: record
        for record in catalog.genes
    }

    assert catalog.panel_gene_count == 3

    assert records["EPCAM"].direct_marker_supported is True
    assert records["EPCAM"].hpa.queried is True
    assert (
        records["EPCAM"]
        .hpa.breast_expression_found
        is True
    )

    assert records["GENE2"].direct_marker_supported is False
    assert (
        records["GENE2"]
        .hpa.breast_expression_found
        is True
    )

    assert (
        records["TESTGENE"]
        .direct_marker_supported
        is False
    )
    assert (
        records["TESTGENE"]
        .hpa.gene_present_in_hpa
        is False
    )

    assert catalog.unresolved_genes == [
        "TESTGENE"
    ]
    assert (
        "GENE2"
        not in catalog.unresolved_genes
    )


def test_validator_reports_expected_catalog_counts() -> None:
    """Report deterministic evidence coverage counts."""

    catalog = _build_catalog()

    result = PanelCatalogValidator().validate(
        catalog,
        expected_genes=[
            "EPCAM",
            "GENE2",
            "TESTGENE",
        ],
    )

    payload = result.to_dict()

    assert payload["is_valid"] is True
    assert payload["panel_gene_count"] == 3
    assert payload["expected_gene_count"] == 3
    assert (
        payload["direct_marker_gene_count"]
        == 1
    )
    assert (
        payload["hpa_queried_gene_count"]
        == 3
    )
    assert (
        payload["hpa_gene_match_count"]
        == 2
    )
    assert (
        payload[
            "hpa_breast_expression_gene_count"
        ]
        == 2
    )
    assert (
        payload["unresolved_gene_count"]
        == 1
    )
    assert payload["warnings"] == []


def test_builder_rejects_agent1_hpa_gene_mismatch() -> None:
    """Reject sources that do not cover the same genes."""

    hpa = _hpa_payload()

    hpa["selected_genes"] = [
        "EPCAM",
        "GENE2",
    ]

    hpa["gene_summaries"] = [
        summary
        for summary in hpa["gene_summaries"]
        if summary["gene"] != "TESTGENE"
    ]

    with pytest.raises(
        ValueError,
        match=(
            "(?i)do not match|"
            "missing hpa genes"
        ),
    ):
        PanelEvidenceBuilder().build(
            _agent1_payload(),
            hpa,
            expected_genes=[
                "EPCAM",
                "GENE2",
                "TESTGENE",
            ],
        )