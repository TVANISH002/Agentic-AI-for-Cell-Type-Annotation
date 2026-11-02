from __future__ import annotations

from pathlib import Path
from typing import Any

from marker_interpretation_agent_v2.reporting.report_writer import (
    Agent2ReportWriter,
)
from marker_interpretation_agent_v2.schemas import (
    CandidateCellType,
    MarkerEvidence,
)


HPA_MASTER_FIELDS = {
    "hpa_queried",
    "hpa_gene_present",
    "hpa_breast_expression_found",
    "hpa_top_cell_type",
    "hpa_top_ncpm",
    "hpa_second_cell_type",
    "hpa_second_ncpm",
    "hpa_top_to_second_ncpm_ratio",
    "hpa_detected_cell_type_count",
    "hpa_highest_read_count",
    "hpa_review_pattern",
}


class MarkerOnlyReportWriter(Agent2ReportWriter):
    """Write clear CellMarker/PanglaoDB-only lung reports."""

    MASTER_GENE_FIELDS = [
        field
        for field in Agent2ReportWriter.MASTER_GENE_FIELDS
        if field not in HPA_MASTER_FIELDS
    ]

    CANDIDATE_FIELDS = [
        "cluster_id",
        "candidate_rank",
        "cell_type",
        "selected_for_downstream",
        "selected_by_agent_selector",
        "meets_minimum_2_gene_rule",
        "minimum_supporting_genes",
        "unique_supporting_gene_count",
        "supporting_genes",
        "both_source_supporting_gene_count",
        "both_source_supporting_genes",
        "specificity_weighted_support",
        "mean_supporting_gene_rank",
        "mean_supporting_log_fold_change",
        "evidence_count",
        "evidence_sources",
        "tissue_contexts",
        "disease_contexts",
        "cell_ontology_ids",
        "warnings",
    ]

    @classmethod
    def _master_gene_rows(
        cls,
        result: MarkerEvidence,
    ) -> list[dict[str, Any]]:
        """Remove unused HPA fields from lung gene reports."""

        return [
            {
                field: row[field]
                for field in cls.MASTER_GENE_FIELDS
            }
            for row in super()._master_gene_rows(result)
        ]

    @classmethod
    def _candidate_row(
        cls,
        result: MarkerEvidence,
        candidate: CandidateCellType,
        *,
        sudden_drop_selected_labels: set[str],
        minimum_supporting_genes: int,
    ) -> dict[str, Any]:
        """Create one marker-only candidate row with explicit decisions."""

        base_row = super()._candidate_row(
            result,
            candidate,
            sudden_drop_selected_labels=(
                sudden_drop_selected_labels
            ),
            minimum_supporting_genes=(
                minimum_supporting_genes
            ),
        )

        selected_by_agent = (
            candidate.cell_type.strip().casefold()
            in sudden_drop_selected_labels
        )

        meets_minimum_rule = (
            candidate.unique_supporting_gene_count
            >= minimum_supporting_genes
        )

        return {
            "cluster_id": base_row["cluster_id"],
            "candidate_rank": base_row["candidate_rank"],
            "cell_type": base_row["cell_type"],
            "selected_for_downstream": selected_by_agent,
            "selected_by_agent_selector": selected_by_agent,
            "meets_minimum_2_gene_rule": meets_minimum_rule,
            "minimum_supporting_genes": minimum_supporting_genes,
            "unique_supporting_gene_count": base_row[
                "unique_supporting_gene_count"
            ],
            "supporting_genes": base_row["supporting_genes"],
            "both_source_supporting_gene_count": base_row[
                "both_source_supporting_gene_count"
            ],
            "both_source_supporting_genes": base_row[
                "both_source_supporting_genes"
            ],
            "specificity_weighted_support": base_row[
                "specificity_weighted_support"
            ],
            "mean_supporting_gene_rank": base_row[
                "mean_supporting_gene_rank"
            ],
            "mean_supporting_log_fold_change": base_row[
                "mean_supporting_log_fold_change"
            ],
            "evidence_count": base_row["evidence_count"],
            "evidence_sources": base_row["evidence_sources"],
            "tissue_contexts": base_row["tissue_contexts"],
            "disease_contexts": base_row["disease_contexts"],
            "cell_ontology_ids": base_row["cell_ontology_ids"],
            "warnings": base_row["warnings"],
        }

    @classmethod
    def _threshold_candidate_rows(
        cls,
        result: MarkerEvidence,
    ) -> list[dict[str, Any]]:
        """Make the selected CSV match the selected candidates in JSON."""

        return cls._candidate_rows(
            result,
            result.selected_candidate_cell_types,
        )

    @classmethod
    def _minimum_2_gene_candidate_rows(
        cls,
        result: MarkerEvidence,
    ) -> list[dict[str, Any]]:
        """Keep the professor-requested minimum-two-gene comparison."""

        minimum_supporting_genes = cls._minimum_supporting_genes(
            result
        )

        candidates = [
            candidate
            for candidate in result.ranked_candidate_cell_types
            if (
                candidate.unique_supporting_gene_count
                >= minimum_supporting_genes
            )
        ]

        return cls._candidate_rows(result, candidates)

    def write(
        self,
        result: MarkerEvidence,
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, Path]:
        """Write standard reports plus a clearly named two-gene report."""

        paths = super().write(
            result,
            output_dir=output_dir,
        )

        validated_result = MarkerEvidence.model_validate(
            result.model_dump()
        )

        selected_path = paths[
            "selected_candidate_cell_types_csv"
        ]

        minimum_2_gene_path = selected_path.with_name(
            selected_path.name.replace(
                "selected_candidate_cell_types.csv",
                "minimum_2_gene_support_candidate_cell_types.csv",
            )
        )

        self._write_csv(
            minimum_2_gene_path,
            fieldnames=self.CANDIDATE_FIELDS,
            rows=self._minimum_2_gene_candidate_rows(
                validated_result
            ),
        )

        paths[
            "minimum_2_gene_support_candidate_cell_types_csv"
        ] = minimum_2_gene_path

        return paths
