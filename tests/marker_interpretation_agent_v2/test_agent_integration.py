from __future__ import annotations

from pathlib import Path

import pytest

from marker_interpretation_agent_v2.agent import (
    MarkerInterpretationAgent,
)
from marker_interpretation_agent_v2.config import (
    Agent2Config,
)
from marker_interpretation_agent_v2.schemas import (
    HPAEvidenceSummary,
    InputGene,
    PanelEvidenceCatalog,
    PanelGeneEvidence,
)


def _hpa(
    *,
    present: bool = True,
    breast_expression: bool = True,
    top_cell_type: str | None = (
        "breast glandular cells"
    ),
    top_ncpm: float | None = 10.0,
) -> HPAEvidenceSummary:
    """Create reusable HPA evidence for integration tests."""

    return HPAEvidenceSummary(
        queried=True,
        gene_present_in_hpa=present,
        breast_expression_found=(
            breast_expression
        ),
        top_cell_type=(
            top_cell_type
            if breast_expression
            else None
        ),
        top_ncpm=(
            top_ncpm
            if breast_expression
            else None
        ),
        detected_cell_type_count=(
            2
            if breast_expression
            else 0
        ),
        highest_read_count=(
            100.0
            if breast_expression
            else None
        ),
    )


def _direct_gene(
    *,
    gene: str,
    cell_type: str,
    sources: list[str],
) -> PanelGeneEvidence:
    """Create one panel gene with direct marker support."""

    records = [
        {
            "source": source,
            "cell_type": cell_type,
            "tissue": (
                "Breast"
                if source == "CellMarker"
                else "Mammary gland"
            ),
        }
        for source in sources
    ]

    return PanelGeneEvidence(
        gene=gene,
        official_symbol=gene,
        marker_status=(
            "context_marker_supported"
        ),
        context_matched_cell_types=[
            cell_type
        ],
        context_matched_database_evidence=(
            records
        ),
        global_database_evidence=records,
        evidence_sources=sources,
        tissue_contexts=["Breast"],
        hpa=_hpa(),
    )


def _catalog() -> PanelEvidenceCatalog:
    """Build a five-gene reusable panel catalog."""

    genes = [
        _direct_gene(
            gene="EPCAM",
            cell_type=(
                "Luminal epithelial cell"
            ),
            sources=[
                "CellMarker",
                "PanglaoDB",
            ],
        ),
        _direct_gene(
            gene="GATA3",
            cell_type=(
                "Luminal epithelial cell"
            ),
            sources=["CellMarker"],
        ),
        _direct_gene(
            gene="COL1A1",
            cell_type="Fibroblast",
            sources=[
                "CellMarker",
                "PanglaoDB",
            ],
        ),
        PanelGeneEvidence(
            gene="GENE2",
            official_symbol="GENE2",
            marker_status="unresolved",
            hpa=_hpa(
                top_cell_type="fibroblasts",
                top_ncpm=5.0,
            ),
        ),
        PanelGeneEvidence(
            gene="TESTGENE",
            official_symbol="TESTGENE",
            marker_status="unresolved",
            hpa=_hpa(
                present=False,
                breast_expression=False,
                top_cell_type=None,
                top_ncpm=None,
            ),
        ),
    ]

    return PanelEvidenceCatalog(
        species="Human",
        tissue="Breast",
        platform="Xenium",
        panel_gene_count=len(genes),
        genes=genes,
        unresolved_genes=[
            "TESTGENE"
        ],
    )


def _inputs() -> list[InputGene]:
    """Create ranked marker inputs for one cluster."""

    return [
        InputGene(
            gene="EPCAM",
            rank_within_cluster=1,
            log_fold_change=3.2,
        ),
        InputGene(
            gene="GATA3",
            rank_within_cluster=2,
            log_fold_change=3.0,
        ),
        InputGene(
            gene="COL1A1",
            rank_within_cluster=3,
            log_fold_change=2.4,
        ),
        InputGene(
            gene="GENE2",
            rank_within_cluster=4,
            log_fold_change=2.0,
        ),
        InputGene(
            gene="TESTGENE",
            rank_within_cluster=5,
            log_fold_change=1.5,
        ),
    ]


def test_agent_runs_complete_cluster_workflow() -> None:
    """Run classification, ranking, selection, flags, and queueing."""

    result, paths = (
        MarkerInterpretationAgent().run(
            cluster_id=0,
            input_genes=_inputs(),
            catalog=_catalog(),
            write_reports=False,
        )
    )

    assert paths == {}
    assert result.cluster_id == "0"

    assert (
        result.summary.total_input_genes
        == 5
    )

    assert (
        result.summary
        .direct_marker_supported_genes
        == 3
    )

    assert (
        result.summary.hpa_queried_genes
        == 5
    )

    assert (
        result.summary
        .ranked_marker_gene_count
        == 3
    )

    assert (
        result.summary.ranked_candidate_count
        == 2
    )

    # Candidate support counts are 2 and 1.
    # The absolute drop is only 1, so it does
    # not meet the meaningful-drop requirement.
    # Both candidates must therefore be retained.
    assert (
        result.summary.selected_candidate_count
        == 2
    )

    assert (
        result.summary.leading_candidate
        == "Luminal epithelial cell"
    )

    assert (
        result.summary.follow_up_gene_count
        == 1
    )

    assert (
        result.candidate_selection
        .cutoff_method
        == (
            "all_candidates_retained_"
            "no_clear_drop"
        )
    )

    assert (
        result.candidate_selection
        .selected_candidate_cell_types
        == [
            "Luminal epithelial cell",
            "Fibroblast",
        ]
    )

    assert [
        candidate.cell_type
        for candidate
        in result.selected_candidate_cell_types
    ] == [
        "Luminal epithelial cell",
        "Fibroblast",
    ]

    leading = (
        result
        .selected_candidate_cell_types[0]
    )

    assert leading.supporting_genes == [
        "EPCAM",
        "GATA3",
    ]

    assert (
        "GENE2"
        not in leading.supporting_genes
    )

    categories = {
        item.gene: item.evidence_category
        for item
        in result.gene_interpretations
    }

    assert (
        categories["GENE2"]
        == "expression_only"
    )

    assert (
        categories["TESTGENE"]
        == "symbol_review_needed"
    )

    assert [
        item.gene
        for item
        in result.follow_up_gene_queue
    ] == [
        "TESTGENE"
    ]


def test_agent_honors_top_k_without_mutating_catalog() -> None:
    """Interpret only the configured number of leading marker genes."""

    catalog = _catalog()

    config = Agent2Config(
        top_k=2
    )

    result = MarkerInterpretationAgent(
        config
    ).interpret_cluster(
        cluster_id="0",
        input_genes=_inputs(),
        catalog=catalog,
    )

    assert [
        gene.gene
        for gene in result.input_genes
    ] == [
        "EPCAM",
        "GATA3",
    ]

    assert (
        result.summary.total_input_genes
        == 2
    )

    assert (
        result.summary
        .direct_marker_supported_genes
        == 2
    )

    assert (
        catalog.panel_gene_count
        == 5
    )


def test_agent_writes_all_cluster_report_files(
    tmp_path: Path,
) -> None:
    """Write validated JSON and CSV outputs for one cluster."""

    result, paths = (
        MarkerInterpretationAgent().run(
            cluster_id="integration",
            input_genes=_inputs(),
            catalog=_catalog(),
            output_dir=tmp_path,
            write_reports=True,
        )
    )

    assert (
        result.cluster_id
        == "integration"
    )

    assert len(paths) == 7

    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0

    assert (
        paths[
            "marker_evidence_json"
        ].suffix
        == ".json"
    )

    assert (
        "cluster_integration"
        in paths[
            "marker_evidence_json"
        ].name
    )


def test_agent_rejects_catalog_context_mismatch() -> None:
    """Stop when catalog context differs from Agent 2 configuration."""

    catalog = _catalog().model_copy(
        update={
            "tissue": "Lung"
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "(?i)context does not match"
        ),
    ):
        MarkerInterpretationAgent().interpret_cluster(
            cluster_id=0,
            input_genes=_inputs(),
            catalog=catalog,
        )