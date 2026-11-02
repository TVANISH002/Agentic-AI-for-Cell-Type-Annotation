from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lung_annotation_pipeline.fallback_candidate_selector import (
    MarkerOnlyCandidateSelector,
)
from lung_annotation_pipeline.marker_only_catalog_builder import (
    MarkerOnlyPanelCatalogBuilder,
)
from lung_annotation_pipeline.marker_only_report_writer import (
    MarkerOnlyReportWriter,
)
from marker_interpretation_agent_v2.agent import (
    MarkerInterpretationAgent,
)
from marker_interpretation_agent_v2.config import Agent2Config
from marker_interpretation_agent_v2.interpretation.evidence_classifier import (
    EvidenceClassifier,
)
from marker_interpretation_agent_v2.interpretation.explanation_builder import (
    ExplanationBuilder,
)
from marker_interpretation_agent_v2.interpretation.review_flag_builder import (
    ReviewFlagBuilder,
)
from marker_interpretation_agent_v2.panel.gene_panel_loader import (
    GenePanelLoader,
)
from marker_interpretation_agent_v2.paths import PROJECT_ROOT
from marker_interpretation_agent_v2.schemas import (
    CandidateCellType,
    CandidateSelection,
    EvidenceCategory,
    GeneInterpretation,
    MarkerEvidence,
    PanelEvidenceCatalog,
    PanelGeneEvidence,
    ReviewFlag,
)


class MarkerOnlyEvidenceClassifier(EvidenceClassifier):
    """Classify marker evidence without treating disabled HPA as missing."""

    def _warnings(
        self,
        panel_gene: PanelGeneEvidence,
        *,
        category: EvidenceCategory,
        review_pattern: str,
    ) -> list[str]:
        """Build warnings for the CellMarker/PanglaoDB-only lung run."""

        del review_pattern

        warnings = list(panel_gene.warnings)

        if (
            not panel_gene.direct_marker_supported
            and panel_gene.global_database_evidence
        ):
            warnings.append(
                "Global CellMarker/PanglaoDB evidence exists, "
                f"but no {self.config.tissue}-context marker evidence "
                "was found."
            )

        if category == "unresolved":
            warnings.append(
                f"No direct {self.config.tissue}-context marker "
                "evidence was found."
            )

        return self._clean_strings(warnings)


class MarkerOnlyReviewFlagBuilder(ReviewFlagBuilder):
    """Remove HPA-completeness flags when HPA is intentionally disabled."""

    _HPA_ONLY_CODES = {
        "hpa_expression_uncertainty",
        "hpa_not_queried",
        "symbol_review_needed",
    }

    def build(
        self,
        interpretations: Sequence[GeneInterpretation],
        ranked_candidates: Sequence[CandidateCellType],
        selected_candidates: Sequence[CandidateCellType],
        candidate_selection: CandidateSelection,
    ) -> list[ReviewFlag]:
        """Build normal review flags, excluding disabled-HPA warnings."""

        flags = super().build(
            interpretations,
            ranked_candidates,
            selected_candidates,
            candidate_selection,
        )

        retained: list[ReviewFlag] = []

        for flag in flags:
            if flag.code in self._HPA_ONLY_CODES:
                continue

            message = flag.message.replace(
                "breast-context",
                f"{self.config.tissue}-context",
            )

            retained.append(
                flag.model_copy(
                    update={"message": message}
                )
            )

        return retained


class MarkerOnlyExplanationBuilder(ExplanationBuilder):
    """Create tissue-aware explanations without claiming HPA retrieval."""

    def __init__(
        self,
        config: Agent2Config,
    ) -> None:
        """Store the lung-specific Agent 2 configuration."""

        self.config = config

    def _hpa_summary(
        self,
        interpretations: Sequence[GeneInterpretation],
    ) -> str:
        """Omit an HPA summary because HPA was not queried."""

        del interpretations
        return ""

    def build(
        self,
        *,
        cluster_id: str | int,
        interpretations: Sequence[GeneInterpretation],
        ranked_candidates: Sequence[CandidateCellType],
        selected_candidates: Sequence[CandidateCellType],
        candidate_selection: CandidateSelection,
        review_flags: Sequence[ReviewFlag],
    ) -> str:
        """Build the normal explanation with lung marker-only wording."""

        explanation = super().build(
            cluster_id=cluster_id,
            interpretations=interpretations,
            ranked_candidates=ranked_candidates,
            selected_candidates=selected_candidates,
            candidate_selection=candidate_selection,
            review_flags=review_flags,
        )

        hpa_disclaimer = (
            "HPA evidence is complementary healthy-breast expression "
            "evidence and does not independently create candidate marker "
            "votes."
        )

        explanation = explanation.replace(
            "breast-context",
            f"{self.config.tissue}-context",
        ).replace(
            hpa_disclaimer,
            "",
        )

        return " ".join(explanation.split())


def _load_panel_genes(
    path: Path,
) -> list[str]:
    """Load and validate the exact ordered panel-gene list."""

    if not path.exists():
        raise FileNotFoundError(
            f"Panel gene list not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if (
            reader.fieldnames is None
            or "gene" not in reader.fieldnames
        ):
            raise ValueError(
                "Panel gene list must contain a 'gene' column."
            )

        genes = [
            str(row.get("gene", "")).strip().upper()
            for row in reader
        ]

    if not genes or any(not gene for gene in genes):
        raise ValueError(
            "Panel gene list contains a missing gene symbol."
        )

    if len(genes) != len(set(genes)):
        raise ValueError(
            "Panel gene list contains duplicate genes."
        )

    return genes


def _cluster_sort_key(
    value: str,
) -> tuple[int, int | str]:
    """Sort numeric cluster identifiers before text identifiers."""

    try:
        return 0, int(value)

    except ValueError:
        return 1, value.casefold()


def _validate_catalog_coverage(
    loader: GenePanelLoader,
    catalog: PanelEvidenceCatalog,
) -> None:
    """Require marker CSV and catalog to cover the same genes."""

    marker_genes = {
        input_gene.gene.strip().upper()
        for cluster_id in loader.get_cluster_ids()
        for input_gene in loader.get_cluster_markers(
            cluster_id
        )
    }

    catalog_genes = {
        record.gene.strip().upper()
        for record in catalog.genes
    }

    missing = sorted(
        marker_genes - catalog_genes
    )

    extra = sorted(
        catalog_genes - marker_genes
    )

    if missing or extra:
        raise ValueError(
            "Marker CSV and panel catalog do not cover "
            "the same genes. "
            f"Missing from catalog: {missing}; "
            f"extra in catalog: {extra}."
        )


def _join(
    values: Sequence[str],
) -> str:
    """Join report values without losing individual entries."""

    return " | ".join(
        value
        for value in values
        if value
    )


def _final_rows(
    results: Sequence[MarkerEvidence],
) -> list[dict[str, Any]]:
    """Create one concise result row for each lung cluster."""

    rows: list[dict[str, Any]] = []

    for result in results:
        if not result.ranked_candidate_cell_types:
            raise ValueError(
                f"Cluster {result.cluster_id} "
                "has no candidate evidence."
            )

        leading = (
            result.ranked_candidate_cell_types[0]
        )

        selected = (
            result.selected_candidate_cell_types
        )

        warning_codes = sorted(
            flag.code
            for flag in result.review_flags
            if flag.severity == "warning"
        )

        rows.append(
            {
                "cluster_id": (
                    result.cluster_id
                ),
                "leading_candidate": (
                    leading.cell_type
                ),
                "selected_candidate_cell_types": (
                    _join(
                        [
                            candidate.cell_type
                            for candidate in selected
                        ]
                    )
                ),
                "supporting_gene_count": (
                    leading.unique_supporting_gene_count
                ),
                "supporting_markers": (
                    _join(
                        leading.supporting_genes
                    )
                ),
                "both_database_supporting_gene_count": (
                    leading
                    .both_source_supporting_gene_count
                ),
                "both_database_supporting_markers": (
                    _join(
                        leading
                        .both_source_supporting_genes
                    )
                ),
                "evidence_sources": (
                    _join(
                        leading.evidence_sources
                    )
                ),
                "cell_ontology_ids": (
                    _join(
                        leading.cell_ontology_ids
                    )
                ),
                "total_input_genes": (
                    result.summary.total_input_genes
                ),
                "direct_marker_supported_genes": (
                    result.summary
                    .direct_marker_supported_genes
                ),
                "follow_up_gene_count": (
                    result.summary.follow_up_gene_count
                ),
                "human_review_required": (
                    result.summary.human_review_required
                ),
                "review_flag_codes": (
                    _join(warning_codes)
                ),
            }
        )

    return rows


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    """Write a non-empty list of dictionaries to CSV."""

    if not rows:
        raise ValueError(
            "No Agent 2 results were produced."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            extrasaction="raise",
        )

        writer.writeheader()
        writer.writerows(rows)



def _write_all_clusters_threshold_candidates(agent2_output: Path) -> Path:
    """Combine per-cluster minimum-support candidate files into one Agent 2 CSV."""
    clusters_dir = agent2_output / "clusters"
    output_path = agent2_output / "all_clusters_threshold_candidates.csv"

    candidate_paths = sorted(
        clusters_dir.glob(
            "cluster_*/cluster_*_minimum_2_gene_support_candidate_cell_types.csv"
        ),
        key=lambda path: path.as_posix(),
    )

    if not candidate_paths:
        raise FileNotFoundError(
            "No per-cluster minimum-support candidate files were found under "
            f"{clusters_dir}"
        )

    fieldnames = [
        "cluster_id",
        "candidate_rank",
        "cell_type",
        "unique_supporting_gene_count",
        "supporting_genes",
        "both_source_supporting_gene_count",
        "both_source_supporting_genes",
        "specificity_weighted_support",
        "mean_supporting_gene_rank",
        "mean_supporting_log_fold_change",
        "evidence_sources",
        "tissue_contexts",
        "disease_contexts",
        "cell_ontology_ids",
        "selection_method",
        "minimum_supporting_genes",
    ]

    rows = []

    for candidate_path in candidate_paths:
        with candidate_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)

            if reader.fieldnames is None:
                continue

            for row in reader:
                rows.append(
                    {
                        "cluster_id": row.get("cluster_id", ""),
                        "candidate_rank": row.get("candidate_rank", ""),
                        "cell_type": row.get("cell_type", ""),
                        "unique_supporting_gene_count": row.get(
                            "unique_supporting_gene_count", ""
                        ),
                        "supporting_genes": row.get("supporting_genes", ""),
                        "both_source_supporting_gene_count": row.get(
                            "both_source_supporting_gene_count", ""
                        ),
                        "both_source_supporting_genes": row.get(
                            "both_source_supporting_genes", ""
                        ),
                        "specificity_weighted_support": row.get(
                            "specificity_weighted_support", ""
                        ),
                        "mean_supporting_gene_rank": row.get(
                            "mean_supporting_gene_rank", ""
                        ),
                        "mean_supporting_log_fold_change": row.get(
                            "mean_supporting_log_fold_change", ""
                        ),
                        "evidence_sources": row.get("evidence_sources", ""),
                        "tissue_contexts": row.get("tissue_contexts", ""),
                        "disease_contexts": row.get("disease_contexts", ""),
                        "cell_ontology_ids": row.get("cell_ontology_ids", ""),
                        "selection_method": row.get(
                            "selection_method",
                            "minimum_2_gene_support",
                        ),
                        "minimum_supporting_genes": row.get(
                            "minimum_supporting_genes",
                            "2",
                        ),
                    }
                )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def run_agent2(
    marker_csv: Path,
    panel_gene_list_csv: Path,
    agent1_json: Path,
    output_root: Path,
    *,
    top_k: int = 50,
) -> Path:
    """Build marker-only input and interpret all lung clusters."""

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    marker_csv = marker_csv.resolve()
    panel_gene_list_csv = (
        panel_gene_list_csv.resolve()
    )
    agent1_json = agent1_json.resolve()
    output_root = output_root.resolve()

    for path in (
        marker_csv,
        panel_gene_list_csv,
        agent1_json,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required Agent 2 input not found: {path}"
            )

    expected_genes = _load_panel_genes(
        panel_gene_list_csv
    )

    agent2_input = (
        output_root
        / "agent2_input"
    )

    agent2_output = (
        output_root
        / "agent2_output"
    )

    final_csv = (
        output_root
        / "final"
        / "lung_cluster_annotations_pipeline.csv"
    )

    builder = MarkerOnlyPanelCatalogBuilder()

    catalog = builder.build_from_agent1_file(
        agent1_json,
        expected_genes=expected_genes,
    )

    builder.write_marker_only(
        catalog,
        json_path=(
            agent2_input
            / "panel_evidence_catalog.json"
        ),
        csv_path=(
            agent2_input
            / "panel_evidence_catalog.csv"
        ),
        unresolved_path=(
            agent2_input
            / "unresolved_panel_genes.csv"
        ),
    )

    if any(
        record.hpa.queried
        for record in catalog.genes
    ):
        raise ValueError(
            "Marker-only lung catalog unexpectedly "
            "contains HPA queries."
        )

    panel_loader = GenePanelLoader(
        marker_csv
    )

    _validate_catalog_coverage(
        panel_loader,
        catalog,
    )

    config = Agent2Config(
        species=catalog.species,
        tissue=catalog.tissue,
        platform=catalog.platform,
        top_k=top_k,
        hpa_enabled=False,
    )

    agent = MarkerInterpretationAgent(
        config=config,
        classifier=(
            MarkerOnlyEvidenceClassifier(config)
        ),
        candidate_selector=(
            MarkerOnlyCandidateSelector(config)
        ),
        review_flag_builder=(
            MarkerOnlyReviewFlagBuilder(config)
        ),
        explanation_builder=(
            MarkerOnlyExplanationBuilder(config)
        ),
        report_writer=MarkerOnlyReportWriter(),
    )

    cluster_ids = sorted(
        [
            str(cluster_id).strip()
            for cluster_id
            in panel_loader.get_cluster_ids()
        ],
        key=_cluster_sort_key,
    )

    if not cluster_ids:
        raise ValueError(
            "The processed lung marker CSV "
            "contains no clusters."
        )

    results: list[MarkerEvidence] = []

    for cluster_id in cluster_ids:
        input_genes = (
            panel_loader.get_cluster_markers(
                cluster_id
            )
        )

        if not input_genes:
            raise ValueError(
                f"Cluster {cluster_id} "
                "contains no marker genes."
            )

        result, _ = agent.run(
            cluster_id=cluster_id,
            input_genes=input_genes,
            catalog=catalog,
            output_dir=(
                agent2_output
                / "clusters"
                / f"cluster_{cluster_id}"
            ),
            write_reports=True,
        )

        results.append(result)

    rows = _final_rows(
        results
    )

    _write_csv(
        agent2_output
        / "all_clusters_summary.csv",
        rows,
    )

    _write_csv(
        final_csv,
        rows,
    )

    threshold_candidates_csv = _write_all_clusters_threshold_candidates(
        agent2_output
    )

    print(
        "Agent 2 completed successfully."
    )

    print(
        f"Cluster count: {len(results)}"
    )

    for result in results:
        print(
            f"Cluster {result.cluster_id}: "
            f"leading="
            f"{result.summary.leading_candidate}; "
            "direct_marker_genes="
            f"{result.summary.direct_marker_supported_genes}; "
            "human_review_required="
            f"{result.summary.human_review_required}"
        )

    print(
        f"Final lung result: {final_csv}"
    )

    print(
        "Cluster-wise reports: "
        f"{agent2_output / 'clusters'}"
    )

    return final_csv


def main() -> None:
    """Run Agent 2 from the verified lung project folders."""

    root = (
        PROJECT_ROOT
        / "data"
        / "lung_run"
    )

    run_agent2(
        marker_csv=(
            root
            / "processed"
            / "lung_cluster_markers_pipeline.csv"
        ),
        panel_gene_list_csv=(
            root
            / "panel"
            / "panel_gene_list.csv"
        ),
        agent1_json=(
            root
            / "agent1_output"
            / "panel_agent1_knowledge_evidence.json"
        ),
        output_root=root,
        top_k=50,
    )


if __name__ == "__main__":
    main()
