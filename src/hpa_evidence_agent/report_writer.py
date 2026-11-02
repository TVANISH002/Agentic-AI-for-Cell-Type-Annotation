
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from hpa_evidence_agent.config import (
    DEFAULT_CONFIG,
    HPAEvidenceConfig,
)
from hpa_evidence_agent.hpa_source_loader import (
    HPASourceLoadResult,
)
from hpa_evidence_agent.schemas import (
    HPAClusterEvidence,
    HPACellTypeSummary,
    HPAExpressionRecord,
    HPAGeneSummary,
    HPAPanelEvidence,
)


class HPAReportWriter:
    """Write independent HPA panel and cluster evidence reports."""

    PANEL_GENE_FIELDS = (
        "gene",
        "gene_present_in_hpa",
        "breast_expression_found",
        "source_status",
        "positive_record_count",
        "detected_cell_type_count",
        "top_breast_cell_type",
        "highest_ncpm",
        "top_cell_type_max_read_count",
        "top_cell_type_cluster_count",
        "second_breast_cell_type",
        "second_highest_ncpm",
        "ncpm_gap_top_vs_second",
        "top_to_second_ncpm_ratio",
        "detected_cell_types",
        "review_status",
        "source",
        "source_release",
        "evidence_scope",
    )

    PANEL_CELL_TYPE_FIELDS = (
        "gene",
        "cell_type_rank",
        "cell_type",
        "is_top_cell_type",
        "max_ncpm",
        "max_read_count",
        "positive_record_count",
        "expressing_cluster_count",
        "clusters",
        "source",
        "source_release",
        "evidence_scope",
    )

    PANEL_EXPRESSION_FIELDS = (
        "gene",
        "ensembl_gene_id",
        "tissue",
        "cell_type",
        "cluster",
        "read_count",
        "ncpm",
        "evidence_type",
        "evidence_scope",
        "source",
        "source_release",
    )

    CLUSTER_GENE_FIELDS = (
        "cluster_id",
        "rank_within_cluster",
        "gene",
        "score",
        "log_fold_change",
        "adjusted_p_value",
        "gene_present_in_hpa",
        "breast_expression_found",
        "source_status",
        "positive_record_count",
        "detected_cell_type_count",
        "top_breast_cell_type",
        "highest_ncpm",
        "top_cell_type_max_read_count",
        "top_cell_type_cluster_count",
        "second_breast_cell_type",
        "second_highest_ncpm",
        "ncpm_gap_top_vs_second",
        "top_to_second_ncpm_ratio",
        "detected_cell_types",
        "review_status",
        "source",
        "source_release",
        "evidence_scope",
    )

    CLUSTER_CELL_TYPE_FIELDS = (
        "cluster_id",
        "rank_within_cluster",
        "gene",
        "hpa_cell_type_rank",
        "cell_type",
        "is_top_cell_type",
        "max_ncpm",
        "max_read_count",
        "positive_record_count",
        "expressing_cluster_count",
        "clusters",
        "source",
        "source_release",
        "evidence_scope",
    )

    CLUSTER_EXPRESSION_FIELDS = (
        "cluster_id",
        "rank_within_cluster",
        "gene",
        "marker_score",
        "marker_log_fold_change",
        "marker_adjusted_p_value",
        "ensembl_gene_id",
        "tissue",
        "cell_type",
        "hpa_cluster",
        "read_count",
        "ncpm",
        "evidence_type",
        "evidence_scope",
        "source",
        "source_release",
    )

    def __init__(
        self,
        config: HPAEvidenceConfig = DEFAULT_CONFIG,
    ) -> None:
        """Store and validate report settings."""

        config.validate()
        self.config = config

    @staticmethod
    def _bool_text(
        value: bool,
    ) -> str:
        """Write CSV boolean values consistently."""

        return "TRUE" if value else "FALSE"

    @staticmethod
    def _join(
        values: Iterable[str],
    ) -> str:
        """Store an ordered list in one CSV cell."""

        return " | ".join(
            str(value)
            for value in values
        )

    @staticmethod
    def _safe_path_token(
        value: str | int,
    ) -> str:
        """Create a safe folder and filename token."""

        cleaned = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(value).strip(),
        )

        cleaned = cleaned.strip(
            "._-"
        )

        if not cleaned:
            raise ValueError(
                "Cluster ID cannot be converted "
                "into a safe path."
            )

        return cleaned

    @staticmethod
    def _write_json(
        path: Path,
        payload: Any,
    ) -> None:
        """Write JSON atomically to avoid partial files."""

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = (
            path.parent
            / f".{path.name}.tmp"
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
            )

            file.write("\n")

        temporary_path.replace(
            path
        )

    @staticmethod
    def _write_csv(
        path: Path,
        *,
        fieldnames: Sequence[str],
        rows: Iterable[
            Mapping[str, Any]
        ],
    ) -> None:
        """Write CSV using a fixed column order."""

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = (
            path.parent
            / f".{path.name}.tmp"
        )

        with temporary_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    fieldnames
                ),
                extrasaction="raise",
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(
                    dict(row)
                )

        temporary_path.replace(
            path
        )

    @staticmethod
    def _validate_clusters(
        clusters: Sequence[
            HPAClusterEvidence
        ],
    ) -> list[HPAClusterEvidence]:
        """Revalidate cluster evidence and reject duplicates."""

        validated: list[
            HPAClusterEvidence
        ] = []

        seen_ids: set[str] = set()

        for cluster in clusters:
            item = (
                HPAClusterEvidence
                .model_validate(
                    cluster.model_dump()
                )
            )

            cluster_key = str(
                item.cluster_id
            )

            if cluster_key in seen_ids:
                raise ValueError(
                    "Duplicate HPA cluster "
                    "evidence for cluster "
                    f"{item.cluster_id}."
                )

            seen_ids.add(
                cluster_key
            )

            validated.append(
                item
            )

        if not validated:
            raise ValueError(
                "cluster_evidence cannot be empty."
            )

        return validated

    def _gene_row(
        self,
        summary: HPAGeneSummary,
        *,
        source: str,
        source_release: str,
    ) -> dict[str, Any]:
        """Flatten one HPA gene summary for CSV."""

        return {
            "gene": summary.gene,
            "gene_present_in_hpa": (
                self._bool_text(
                    summary.gene_present_in_hpa
                )
            ),
            "breast_expression_found": (
                self._bool_text(
                    summary.breast_expression_found
                )
            ),
            "source_status": (
                summary.source_status
            ),
            "positive_record_count": (
                summary.positive_record_count
            ),
            "detected_cell_type_count": (
                summary.detected_cell_type_count
            ),
            "top_breast_cell_type": (
                summary.top_breast_cell_type
                or ""
            ),
            "highest_ncpm": (
                summary.highest_ncpm
            ),
            "top_cell_type_max_read_count": (
                summary
                .top_cell_type_max_read_count
            ),
            "top_cell_type_cluster_count": (
                summary
                .top_cell_type_cluster_count
            ),
            "second_breast_cell_type": (
                summary.second_breast_cell_type
                or ""
            ),
            "second_highest_ncpm": (
                summary.second_highest_ncpm
            ),
            "ncpm_gap_top_vs_second": (
                summary.ncpm_gap_top_vs_second
            ),
            "top_to_second_ncpm_ratio": (
                ""
                if (
                    summary
                    .top_to_second_ncpm_ratio
                    is None
                )
                else (
                    summary
                    .top_to_second_ncpm_ratio
                )
            ),
            "detected_cell_types": (
                self._join(
                    summary.detected_cell_types
                )
            ),
            "review_status": (
                summary.review_status
            ),
            "source": source,
            "source_release": (
                source_release
            ),
            "evidence_scope": (
                self.config.evidence_scope
            ),
        }

    def _cell_type_rows(
        self,
        *,
        gene: str,
        summaries: Sequence[
            HPACellTypeSummary
        ],
        source: str,
        source_release: str,
    ) -> list[dict[str, Any]]:
        """Flatten one gene's HPA cell-type summaries."""

        return [
            {
                "gene": gene,
                "cell_type_rank": rank,
                "cell_type": (
                    summary.cell_type
                ),
                "is_top_cell_type": (
                    self._bool_text(
                        rank == 1
                    )
                ),
                "max_ncpm": (
                    summary.max_ncpm
                ),
                "max_read_count": (
                    summary.max_read_count
                ),
                "positive_record_count": (
                    summary.positive_record_count
                ),
                "expressing_cluster_count": (
                    summary
                    .expressing_cluster_count
                ),
                "clusters": (
                    self._join(
                        summary.clusters
                    )
                ),
                "source": source,
                "source_release": (
                    source_release
                ),
                "evidence_scope": (
                    self.config.evidence_scope
                ),
            }
            for rank, summary in enumerate(
                summaries,
                start=1,
            )
        ]

    def _cluster_gene_rows(
        self,
        cluster: HPAClusterEvidence,
    ) -> list[dict[str, Any]]:
        """Flatten HPA gene evidence for one cluster."""

        rows: list[
            dict[str, Any]
        ] = []

        for item in cluster.genes:
            gene_row = self._gene_row(
                item.hpa,
                source=cluster.source,
                source_release=(
                    cluster.source_release
                ),
            )

            rows.append(
                {
                    "cluster_id": (
                        cluster.cluster_id
                    ),
                    "rank_within_cluster": (
                        item.rank_within_cluster
                    ),
                    "gene": item.gene,
                    "score": (
                        ""
                        if item.score is None
                        else item.score
                    ),
                    "log_fold_change": (
                        ""
                        if (
                            item.log_fold_change
                            is None
                        )
                        else (
                            item.log_fold_change
                        )
                    ),
                    "adjusted_p_value": (
                        ""
                        if (
                            item.adjusted_p_value
                            is None
                        )
                        else (
                            item.adjusted_p_value
                        )
                    ),
                    **{
                        key: value
                        for key, value
                        in gene_row.items()
                        if key != "gene"
                    },
                }
            )

        return rows

    def _cluster_cell_type_rows(
        self,
        cluster: HPAClusterEvidence,
    ) -> list[dict[str, Any]]:
        """Flatten HPA cell-type evidence for one cluster."""

        rows: list[
            dict[str, Any]
        ] = []

        for item in cluster.genes:
            for (
                hpa_rank,
                summary,
            ) in enumerate(
                item.hpa.cell_type_summaries,
                start=1,
            ):
                rows.append(
                    {
                        "cluster_id": (
                            cluster.cluster_id
                        ),
                        "rank_within_cluster": (
                            item.rank_within_cluster
                        ),
                        "gene": item.gene,
                        "hpa_cell_type_rank": (
                            hpa_rank
                        ),
                        "cell_type": (
                            summary.cell_type
                        ),
                        "is_top_cell_type": (
                            self._bool_text(
                                hpa_rank == 1
                            )
                        ),
                        "max_ncpm": (
                            summary.max_ncpm
                        ),
                        "max_read_count": (
                            summary.max_read_count
                        ),
                        "positive_record_count": (
                            summary
                            .positive_record_count
                        ),
                        "expressing_cluster_count": (
                            summary
                            .expressing_cluster_count
                        ),
                        "clusters": (
                            self._join(
                                summary.clusters
                            )
                        ),
                        "source": (
                            cluster.source
                        ),
                        "source_release": (
                            cluster.source_release
                        ),
                        "evidence_scope": (
                            self.config.evidence_scope
                        ),
                    }
                )

        return rows

    def _cluster_expression_rows(
        self,
        cluster: HPAClusterEvidence,
        *,
        records_by_gene: Mapping[
            str,
            Sequence[HPAExpressionRecord],
        ],
    ) -> list[dict[str, Any]]:
        """Create raw HPA expression rows for one cluster."""

        rows: list[
            dict[str, Any]
        ] = []

        for item in cluster.genes:
            gene_records = records_by_gene.get(
                item.gene,
                [],
            )

            for record in gene_records:
                rows.append(
                    {
                        "cluster_id": (
                            cluster.cluster_id
                        ),
                        "rank_within_cluster": (
                            item.rank_within_cluster
                        ),
                        "gene": item.gene,
                        "marker_score": (
                            ""
                            if item.score is None
                            else item.score
                        ),
                        "marker_log_fold_change": (
                            ""
                            if (
                                item.log_fold_change
                                is None
                            )
                            else (
                                item.log_fold_change
                            )
                        ),
                        "marker_adjusted_p_value": (
                            ""
                            if (
                                item.adjusted_p_value
                                is None
                            )
                            else (
                                item.adjusted_p_value
                            )
                        ),
                        "ensembl_gene_id": (
                            record.ensembl_gene_id
                        ),
                        "tissue": (
                            record.tissue
                        ),
                        "cell_type": (
                            record.cell_type
                        ),
                        "hpa_cluster": (
                            record.cluster
                        ),
                        "read_count": (
                            record.read_count
                        ),
                        "ncpm": record.ncpm,
                        "evidence_type": (
                            record.evidence_type
                        ),
                        "evidence_scope": (
                            record.evidence_scope
                        ),
                        "source": (
                            record.source
                        ),
                        "source_release": (
                            cluster.source_release
                        ),
                    }
                )

        return rows

    def _write_per_cluster_reports(
        self,
        *,
        destination: Path,
        clusters: Sequence[
            HPAClusterEvidence
        ],
        records_by_gene: Mapping[
            str,
            Sequence[HPAExpressionRecord],
        ],
    ) -> list[Path]:
        """Write a separate folder and report set for each cluster."""

        written: list[Path] = []

        clusters_root = (
            destination
            / "clusters"
        )

        clusters_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        for cluster in clusters:
            cluster_token = (
                self._safe_path_token(
                    cluster.cluster_id
                )
            )

            cluster_folder = (
                clusters_root
                / f"cluster_{cluster_token}"
            )

            cluster_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            prefix = (
                f"cluster_{cluster_token}"
            )

            cluster_json_path = (
                cluster_folder
                / f"{prefix}_hpa_evidence.json"
            )

            self._write_json(
                cluster_json_path,
                cluster.model_dump(
                    mode="json"
                ),
            )

            written.append(
                cluster_json_path
            )

            cluster_gene_path = (
                cluster_folder
                / f"{prefix}_hpa_gene_summary.csv"
            )

            self._write_csv(
                cluster_gene_path,
                fieldnames=(
                    self.CLUSTER_GENE_FIELDS
                ),
                rows=(
                    self._cluster_gene_rows(
                        cluster
                    )
                ),
            )

            written.append(
                cluster_gene_path
            )

            cluster_cell_type_path = (
                cluster_folder
                / (
                    f"{prefix}_hpa_"
                    "cell_type_summary.csv"
                )
            )

            self._write_csv(
                cluster_cell_type_path,
                fieldnames=(
                    self.CLUSTER_CELL_TYPE_FIELDS
                ),
                rows=(
                    self._cluster_cell_type_rows(
                        cluster
                    )
                ),
            )

            written.append(
                cluster_cell_type_path
            )

            cluster_expression_path = (
                cluster_folder
                / (
                    f"{prefix}_hpa_"
                    "expression_records.csv"
                )
            )

            self._write_csv(
                cluster_expression_path,
                fieldnames=(
                    self.CLUSTER_EXPRESSION_FIELDS
                ),
                rows=(
                    self._cluster_expression_rows(
                        cluster,
                        records_by_gene=(
                            records_by_gene
                        ),
                    )
                ),
            )

            written.append(
                cluster_expression_path
            )

        return written

    def write(
        self,
        *,
        panel_evidence: HPAPanelEvidence,
        cluster_evidence: Sequence[
            HPAClusterEvidence
        ],
        source_result: HPASourceLoadResult,
        output_dir: str | Path | None = None,
    ) -> list[Path]:
        """Write complete independent HPA reports."""

        panel = (
            HPAPanelEvidence
            .model_validate(
                panel_evidence.model_dump()
            )
        )

        clusters = (
            self._validate_clusters(
                cluster_evidence
            )
        )

        if (
            tuple(panel.selected_genes)
            != source_result.selected_genes
        ):
            raise ValueError(
                "Panel evidence genes do not align "
                "with the HPA source result."
            )

        destination = (
            Path(output_dir)
            if output_dir is not None
            else self.config.output_dir
        )

        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        written: list[Path] = []

        records_by_gene: dict[
            str,
            list[HPAExpressionRecord],
        ] = defaultdict(list)

        for record in panel.expression_records:
            records_by_gene[
                record.gene
            ].append(
                record
            )

        panel_json_path = (
            destination
            / "panel_hpa_evidence.json"
        )

        self._write_json(
            panel_json_path,
            panel.model_dump(
                mode="json"
            ),
        )

        written.append(
            panel_json_path
        )

        panel_gene_path = (
            destination
            / "panel_hpa_gene_summary.csv"
        )

        self._write_csv(
            panel_gene_path,
            fieldnames=(
                self.PANEL_GENE_FIELDS
            ),
            rows=(
                self._gene_row(
                    summary,
                    source=panel.source,
                    source_release=(
                        panel.source_release
                    ),
                )
                for summary
                in panel.gene_summaries
            ),
        )

        written.append(
            panel_gene_path
        )

        panel_cell_type_rows: list[
            dict[str, Any]
        ] = []

        for summary in panel.gene_summaries:
            panel_cell_type_rows.extend(
                self._cell_type_rows(
                    gene=summary.gene,
                    summaries=(
                        summary.cell_type_summaries
                    ),
                    source=panel.source,
                    source_release=(
                        panel.source_release
                    ),
                )
            )

        panel_cell_type_path = (
            destination
            / "panel_hpa_cell_type_summary.csv"
        )

        self._write_csv(
            panel_cell_type_path,
            fieldnames=(
                self.PANEL_CELL_TYPE_FIELDS
            ),
            rows=(
                panel_cell_type_rows
            ),
        )

        written.append(
            panel_cell_type_path
        )

        panel_expression_path = (
            destination
            / "panel_hpa_expression_records.csv"
        )

        self._write_csv(
            panel_expression_path,
            fieldnames=(
                self.PANEL_EXPRESSION_FIELDS
            ),
            rows=(
                {
                    **record.model_dump(),
                    "source_release": (
                        panel.source_release
                    ),
                }
                for record
                in panel.expression_records
            ),
        )

        written.append(
            panel_expression_path
        )

        validation_path = (
            destination
            / "panel_hpa_source_validation.json"
        )

        validation_payload = {
            "source": panel.source,
            "source_release": (
                panel.source_release
            ),
            "query_tissue": (
                panel.query_tissue
            ),
            "source_url": (
                panel.source_url
            ),
            "source_path": str(
                source_result.source_path
            ),
            "source_sha256": (
                panel.source_sha256
            ),
            "source_file_size_bytes": (
                panel.source_file_size_bytes
            ),
            "archive_member_name": (
                source_result.archive_member_name
            ),
            "retrieved_at_utc": (
                panel.retrieved_at_utc.isoformat()
            ),
            "requested_gene_count": (
                panel
                .summary
                .requested_gene_count
            ),
            "scanned_row_count": (
                source_result.scanned_row_count
            ),
            "matched_panel_row_count": (
                source_result
                .matched_panel_row_count
            ),
            "breast_expression_found_count": (
                panel
                .summary
                .breast_expression_found_count
            ),
            (
                "gene_found_without_"
                "breast_expression_count"
            ): (
                panel
                .summary
                .gene_found_without_breast_expression_count
            ),
            "gene_not_found_in_hpa_count": (
                panel
                .summary
                .gene_not_found_in_hpa_count
            ),
            "expression_record_count": (
                panel
                .summary
                .expression_record_count
            ),
            "warnings": panel.warnings,
        }

        self._write_json(
            validation_path,
            validation_payload,
        )

        written.append(
            validation_path
        )

        all_clusters_json_path = (
            destination
            / "all_clusters_hpa_evidence.json"
        )

        self._write_json(
            all_clusters_json_path,
            [
                cluster.model_dump(
                    mode="json"
                )
                for cluster in clusters
            ],
        )

        written.append(
            all_clusters_json_path
        )

        all_cluster_gene_rows: list[
            dict[str, Any]
        ] = []

        all_cluster_cell_type_rows: list[
            dict[str, Any]
        ] = []

        for cluster in clusters:
            all_cluster_gene_rows.extend(
                self._cluster_gene_rows(
                    cluster
                )
            )

            all_cluster_cell_type_rows.extend(
                self._cluster_cell_type_rows(
                    cluster
                )
            )

        all_cluster_gene_path = (
            destination
            / "all_clusters_hpa_gene_summary.csv"
        )

        self._write_csv(
            all_cluster_gene_path,
            fieldnames=(
                self.CLUSTER_GENE_FIELDS
            ),
            rows=(
                all_cluster_gene_rows
            ),
        )

        written.append(
            all_cluster_gene_path
        )

        all_cluster_cell_type_path = (
            destination
            / "all_clusters_hpa_cell_type_summary.csv"
        )

        self._write_csv(
            all_cluster_cell_type_path,
            fieldnames=(
                self.CLUSTER_CELL_TYPE_FIELDS
            ),
            rows=(
                all_cluster_cell_type_rows
            ),
        )

        written.append(
            all_cluster_cell_type_path
        )

        written.extend(
            self._write_per_cluster_reports(
                destination=destination,
                clusters=clusters,
                records_by_gene=(
                    records_by_gene
                ),
            )
        )

        return written


def write_hpa_reports(
    *,
    panel_evidence: HPAPanelEvidence,
    cluster_evidence: Sequence[
        HPAClusterEvidence
    ],
    source_result: HPASourceLoadResult,
    config: HPAEvidenceConfig = DEFAULT_CONFIG,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Write all independent HPA reports."""

    return HPAReportWriter(
        config
    ).write(
        panel_evidence=(
            panel_evidence
        ),
        cluster_evidence=(
            cluster_evidence
        ),
        source_result=(
            source_result
        ),
        output_dir=output_dir,
    )


