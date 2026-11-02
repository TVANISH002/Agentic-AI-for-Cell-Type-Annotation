from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from marker_interpretation_agent_v2.config import DEFAULT_CONFIG
from marker_interpretation_agent_v2.paths import (
    get_cluster_output_dir,
    normalize_path,
)
from marker_interpretation_agent_v2.schemas import (
    CandidateCellType,
    MarkerEvidence,
)


class Agent2ReportWriter:
    """Write complete Agent 2 V2 JSON and CSV reports."""

    MASTER_GENE_FIELDS = [
        "cluster_id",
        "gene",
        "rank_within_cluster",
        "score",
        "log_fold_change",
        "p_value",
        "adjusted_p_value",
        "pct_nz_group",
        "pct_nz_reference",
        "direct_marker_supported",
        "evidence_category",
        "supported_cell_types",
        "source_agreement_count",
        "marker_specificity_score",
        "context_evidence_count",
        "evidence_sources",
        "context_evidence_json",
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
        "warnings",
    ]

    RANKED_MARKER_FIELDS = [
        "cluster_id",
        "marker_rank",
        "gene",
        "source_agreement_count",
        "marker_specificity_score",
        "context_evidence_count",
        "cluster_gene_rank",
        "log_fold_change",
        "supported_cell_types",
        "evidence_sources",
        "ranking_reason",
    ]

    CANDIDATE_FIELDS = [
        "cluster_id",
        "candidate_rank",
        "cell_type",
        "selected_for_downstream",
        "selected_by_sudden_drop",
        "selected_by_min_support_threshold",
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
        "hpa_queried_gene_count",
        "hpa_breast_expression_gene_count",
        "hpa_top_cell_types",
        "warnings",
    ]

    FOLLOW_UP_FIELDS = [
        "cluster_id",
        "gene",
        "evidence_category",
        "recommended_action",
        "reason",
        "source_coverage",
        "warnings",
    ]

    REVIEW_FLAG_FIELDS = [
        "cluster_id",
        "code",
        "severity",
        "message",
        "genes",
        "cell_types",
    ]

    @staticmethod
    def _safe_cluster_id(
        cluster_id: str | int,
    ) -> str:
        """Create a filesystem-safe cluster identifier."""

        raw_value = str(cluster_id).strip()

        if not raw_value:
            raise ValueError(
                "cluster_id cannot be empty."
            )

        safe_value = "".join(
            character
            if (
                character.isalnum()
                or character in {"-", "_"}
            )
            else "_"
            for character in raw_value
        )

        if not safe_value:
            raise ValueError(
                "cluster_id does not contain any usable characters."
            )

        return safe_value

    @staticmethod
    def _join(
        values: Sequence[Any],
    ) -> str:
        """Join values using a stable report delimiter."""

        return " | ".join(
            str(value).strip()
            for value in values
            if str(value).strip()
        )

    @staticmethod
    def _json_text(
        value: Any,
    ) -> str:
        """Serialize a value for storage inside a CSV field."""

        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _write_csv(
        path: Path,
        *,
        fieldnames: Sequence[str],
        rows: Sequence[dict[str, Any]],
    ) -> None:
        """Write a CSV with headers even when there are no rows."""

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(fieldnames),
                extrasaction="raise",
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(row)

    @staticmethod
    def _minimum_supporting_genes(
        result: MarkerEvidence,
    ) -> int:
        """Read the reporting threshold from result config or defaults."""

        result_config = getattr(
            result,
            "config",
            None,
        )

        threshold: Any = None

        if isinstance(result_config, dict):
            threshold = result_config.get(
                "candidate_reporting_min_supporting_genes"
            )
        else:
            threshold = getattr(
                result_config,
                "candidate_reporting_min_supporting_genes",
                None,
            )

        if threshold is None:
            threshold = getattr(
                DEFAULT_CONFIG,
                "candidate_reporting_min_supporting_genes",
                None,
            )

        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, int)
            or threshold <= 0
        ):
            raise ValueError(
                "candidate_reporting_min_supporting_genes "
                "must be a positive integer."
            )

        return threshold

    @classmethod
    def _master_gene_rows(
        cls,
        result: MarkerEvidence,
    ) -> list[dict[str, Any]]:
        """Create the complete one-row-per-gene report."""

        input_gene_map = {
            input_gene.gene: input_gene
            for input_gene in result.input_genes
        }

        rows: list[dict[str, Any]] = []

        for interpretation in result.gene_interpretations:
            input_gene = input_gene_map[
                interpretation.gene
            ]

            statistics = interpretation.statistics
            hpa = interpretation.hpa

            rows.append(
                {
                    "cluster_id": result.cluster_id,
                    "gene": interpretation.gene,
                    "rank_within_cluster": (
                        input_gene.rank_within_cluster
                    ),
                    "score": input_gene.score,
                    "log_fold_change": (
                        input_gene.log_fold_change
                    ),
                    "p_value": input_gene.p_value,
                    "adjusted_p_value": (
                        input_gene.adjusted_p_value
                    ),
                    "pct_nz_group": (
                        input_gene.pct_nz_group
                    ),
                    "pct_nz_reference": (
                        input_gene.pct_nz_reference
                    ),
                    "direct_marker_supported": (
                        interpretation
                        .direct_marker_supported
                    ),
                    "evidence_category": (
                        interpretation.evidence_category
                    ),
                    "supported_cell_types": cls._join(
                        interpretation.supported_cell_types
                    ),
                    "source_agreement_count": (
                        statistics
                        .source_agreement_count
                    ),
                    "marker_specificity_score": (
                        statistics
                        .marker_specificity_score
                    ),
                    "context_evidence_count": (
                        statistics
                        .context_evidence_count
                    ),
                    "evidence_sources": cls._join(
                        interpretation.evidence_sources
                    ),
                    "context_evidence_json": (
                        cls._json_text(
                            interpretation.context_evidence
                        )
                    ),
                    "hpa_queried": hpa.queried,
                    "hpa_gene_present": (
                        hpa.gene_present_in_hpa
                    ),
                    "hpa_breast_expression_found": (
                        hpa.breast_expression_found
                    ),
                    "hpa_top_cell_type": (
                        hpa.top_cell_type or ""
                    ),
                    "hpa_top_ncpm": hpa.top_ncpm,
                    "hpa_second_cell_type": (
                        hpa.second_cell_type or ""
                    ),
                    "hpa_second_ncpm": (
                        hpa.second_ncpm
                    ),
                    "hpa_top_to_second_ncpm_ratio": (
                        hpa.top_to_second_ncpm_ratio
                    ),
                    "hpa_detected_cell_type_count": (
                        hpa.detected_cell_type_count
                    ),
                    "hpa_highest_read_count": (
                        hpa.highest_read_count
                    ),
                    "hpa_review_pattern": (
                        hpa.review_pattern or ""
                    ),
                    "warnings": cls._join(
                        interpretation.warnings
                    ),
                }
            )

        return rows

    @classmethod
    def _ranked_marker_rows(
        cls,
        result: MarkerEvidence,
    ) -> list[dict[str, Any]]:
        """Create the ranked direct-marker gene report."""

        return [
            {
                "cluster_id": result.cluster_id,
                "marker_rank": marker.marker_rank,
                "gene": marker.gene,
                "source_agreement_count": (
                    marker.source_agreement_count
                ),
                "marker_specificity_score": (
                    marker.marker_specificity_score
                ),
                "context_evidence_count": (
                    marker.context_evidence_count
                ),
                "cluster_gene_rank": (
                    marker.cluster_gene_rank
                ),
                "log_fold_change": (
                    marker.log_fold_change
                ),
                "supported_cell_types": cls._join(
                    marker.supported_cell_types
                ),
                "evidence_sources": cls._join(
                    marker.evidence_sources
                ),
                "ranking_reason": (
                    marker.ranking_reason
                ),
            }
            for marker in result.ranked_marker_genes
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
        """Create one candidate row with both selection decisions."""

        normalized_label = (
            candidate.cell_type.strip().casefold()
        )

        selected_by_sudden_drop = (
            normalized_label
            in sudden_drop_selected_labels
        )

        selected_by_min_support_threshold = (
            candidate.unique_supporting_gene_count
            >= minimum_supporting_genes
        )

        return {
            "cluster_id": result.cluster_id,
            "candidate_rank": (
                candidate.candidate_rank
            ),
            "cell_type": candidate.cell_type,
            "selected_for_downstream": (
                selected_by_min_support_threshold
            ),
            "selected_by_sudden_drop": (
                selected_by_sudden_drop
            ),
            "selected_by_min_support_threshold": (
                selected_by_min_support_threshold
            ),
            "minimum_supporting_genes": (
                minimum_supporting_genes
            ),
            "unique_supporting_gene_count": (
                candidate
                .unique_supporting_gene_count
            ),
            "supporting_genes": cls._join(
                candidate.supporting_genes
            ),
            "both_source_supporting_gene_count": (
                candidate
                .both_source_supporting_gene_count
            ),
            "both_source_supporting_genes": (
                cls._join(
                    candidate
                    .both_source_supporting_genes
                )
            ),
            "specificity_weighted_support": (
                candidate
                .specificity_weighted_support
            ),
            "mean_supporting_gene_rank": (
                candidate.mean_supporting_gene_rank
            ),
            "mean_supporting_log_fold_change": (
                candidate
                .mean_supporting_log_fold_change
            ),
            "evidence_count": (
                candidate.evidence_count
            ),
            "evidence_sources": cls._join(
                candidate.evidence_sources
            ),
            "tissue_contexts": cls._join(
                candidate.tissue_contexts
            ),
            "disease_contexts": cls._join(
                candidate.disease_contexts
            ),
            "cell_ontology_ids": cls._join(
                candidate.cell_ontology_ids
            ),
            "hpa_queried_gene_count": (
                candidate.hpa_queried_gene_count
            ),
            "hpa_breast_expression_gene_count": (
                candidate
                .hpa_breast_expression_gene_count
            ),
            "hpa_top_cell_types": cls._join(
                candidate.hpa_top_cell_types
            ),
            "warnings": cls._join(
                candidate.warnings
            ),
        }

    @classmethod
    def _candidate_rows(
        cls,
        result: MarkerEvidence,
        candidates: Sequence[CandidateCellType],
    ) -> list[dict[str, Any]]:
        """Create candidate rows using both selection methods."""

        sudden_drop_selected_labels = {
            candidate.cell_type.strip().casefold()
            for candidate
            in result.selected_candidate_cell_types
        }

        minimum_supporting_genes = (
            cls._minimum_supporting_genes(
                result
            )
        )

        return [
            cls._candidate_row(
                result,
                candidate,
                sudden_drop_selected_labels=(
                    sudden_drop_selected_labels
                ),
                minimum_supporting_genes=(
                    minimum_supporting_genes
                ),
            )
            for candidate in candidates
        ]

    @classmethod
    def _ranked_candidate_rows(
        cls,
        result: MarkerEvidence,
    ) -> list[dict[str, Any]]:
        """Create the complete ranked candidate table."""

        return cls._candidate_rows(
            result,
            result.ranked_candidate_cell_types,
        )

    @classmethod
    def _threshold_candidate_rows(
        cls,
        result: MarkerEvidence,
    ) -> list[dict[str, Any]]:
        """Create the threshold-selected candidate table."""

        minimum_supporting_genes = (
            cls._minimum_supporting_genes(
                result
            )
        )

        threshold_candidates = [
            candidate
            for candidate
            in result.ranked_candidate_cell_types
            if (
                candidate.unique_supporting_gene_count
                >= minimum_supporting_genes
            )
        ]

        return cls._candidate_rows(
            result,
            threshold_candidates,
        )

    @classmethod
    def _follow_up_rows(
        cls,
        result: MarkerEvidence,
    ) -> list[dict[str, Any]]:
        """Create the unresolved-gene queue report."""

        return [
            {
                "cluster_id": result.cluster_id,
                "gene": item.gene,
                "evidence_category": (
                    item.evidence_category
                ),
                "recommended_action": (
                    item.recommended_action
                ),
                "reason": item.reason,
                "source_coverage": cls._join(
                    item.source_coverage
                ),
                "warnings": cls._join(
                    item.warnings
                ),
            }
            for item in result.follow_up_gene_queue
        ]

    @classmethod
    def _review_flag_rows(
        cls,
        result: MarkerEvidence,
    ) -> list[dict[str, Any]]:
        """Create the review-flag report."""

        return [
            {
                "cluster_id": result.cluster_id,
                "code": flag.code,
                "severity": flag.severity,
                "message": flag.message,
                "genes": cls._join(
                    flag.genes
                ),
                "cell_types": cls._join(
                    flag.cell_types
                ),
            }
            for flag in result.review_flags
        ]

    def write(
        self,
        result: MarkerEvidence,
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, Path]:
        """Validate and write every Agent 2 V2 report."""

        validated_result = (
            MarkerEvidence.model_validate(
                result.model_dump()
            )
        )

        safe_cluster_id = self._safe_cluster_id(
            validated_result.cluster_id
        )

        resolved_output_dir = (
            normalize_path(output_dir)
            if output_dir is not None
            else get_cluster_output_dir(
                safe_cluster_id
            )
        )

        resolved_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        prefix = f"cluster_{safe_cluster_id}"

        json_path = (
            resolved_output_dir
            / f"{prefix}_marker_evidence.json"
        )

        master_gene_path = (
            resolved_output_dir
            / f"{prefix}_master_gene_evidence.csv"
        )

        ranked_marker_path = (
            resolved_output_dir
            / f"{prefix}_ranked_marker_genes.csv"
        )

        ranked_candidate_path = (
            resolved_output_dir
            / (
                f"{prefix}_ranked_"
                "candidate_cell_types.csv"
            )
        )

        selected_candidate_path = (
            resolved_output_dir
            / (
                f"{prefix}_selected_"
                "candidate_cell_types.csv"
            )
        )

        follow_up_path = (
            resolved_output_dir
            / f"{prefix}_follow_up_gene_queue.csv"
        )

        review_flag_path = (
            resolved_output_dir
            / f"{prefix}_review_flags.csv"
        )

        json_path.write_text(
            validated_result.model_dump_json(
                indent=2
            ),
            encoding="utf-8",
        )

        self._write_csv(
            master_gene_path,
            fieldnames=self.MASTER_GENE_FIELDS,
            rows=self._master_gene_rows(
                validated_result
            ),
        )

        self._write_csv(
            ranked_marker_path,
            fieldnames=self.RANKED_MARKER_FIELDS,
            rows=self._ranked_marker_rows(
                validated_result
            ),
        )

        self._write_csv(
            ranked_candidate_path,
            fieldnames=self.CANDIDATE_FIELDS,
            rows=self._ranked_candidate_rows(
                validated_result
            ),
        )

        self._write_csv(
            selected_candidate_path,
            fieldnames=self.CANDIDATE_FIELDS,
            rows=self._threshold_candidate_rows(
                validated_result
            ),
        )

        self._write_csv(
            follow_up_path,
            fieldnames=self.FOLLOW_UP_FIELDS,
            rows=self._follow_up_rows(
                validated_result
            ),
        )

        self._write_csv(
            review_flag_path,
            fieldnames=self.REVIEW_FLAG_FIELDS,
            rows=self._review_flag_rows(
                validated_result
            ),
        )

        return {
            "marker_evidence_json": json_path,
            "master_gene_evidence_csv": (
                master_gene_path
            ),
            "ranked_marker_genes_csv": (
                ranked_marker_path
            ),
            "ranked_candidate_cell_types_csv": (
                ranked_candidate_path
            ),
            "selected_candidate_cell_types_csv": (
                selected_candidate_path
            ),
            "follow_up_gene_queue_csv": (
                follow_up_path
            ),
            "review_flags_csv": (
                review_flag_path
            ),
        }
        
        
        