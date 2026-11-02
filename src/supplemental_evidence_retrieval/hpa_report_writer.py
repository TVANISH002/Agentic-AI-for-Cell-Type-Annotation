from __future__ import annotations

import csv
from pathlib import Path

from .hpa_schemas import HPARetrievalResult


class HPAReportWriter:
    def write_all(
        self,
        result: HPARetrievalResult,
        output_dir: Path,
    ) -> dict[str, Path]:
        """Writes all HPA result files."""

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        prefix = (
            f"cluster_{result.cluster_id}_"
            f"hpa_{result.run_mode}"
        )

        json_path = output_dir / (
            f"{prefix}.json"
        )

        gene_summary_path = output_dir / (
            f"{prefix}_gene_summary.csv"
        )

        cell_type_summary_path = output_dir / (
            f"{prefix}_cell_type_summary.csv"
        )

        expression_records_path = output_dir / (
            f"{prefix}_expression_records.csv"
        )

        json_path.write_text(
            result.model_dump_json(
                indent=2
            ),
            encoding="utf-8",
        )

        self._write_gene_summary(
            result,
            gene_summary_path,
        )

        self._write_cell_type_summary(
            result,
            cell_type_summary_path,
        )

        self._write_expression_records(
            result,
            expression_records_path,
        )

        return {
            "json": json_path,
            "gene_summary_csv": (
                gene_summary_path
            ),
            "cell_type_summary_csv": (
                cell_type_summary_path
            ),
            "expression_records_csv": (
                expression_records_path
            ),
        }

    @staticmethod
    def _write_gene_summary(
        result: HPARetrievalResult,
        output_path: Path,
    ) -> None:
        """Writes one summary row per gene."""

        fieldnames = [
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
        ]

        with output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for gene_result in (
                result.gene_summaries
            ):
                writer.writerow(
                    {
                        "gene": gene_result.gene,
                        "gene_present_in_hpa": (
                            gene_result
                            .gene_present_in_hpa
                        ),
                        "breast_expression_found": (
                            gene_result
                            .breast_expression_found
                        ),
                        "source_status": (
                            gene_result.source_status
                        ),
                        "positive_record_count": (
                            gene_result
                            .positive_record_count
                        ),
                        "detected_cell_type_count": (
                            gene_result
                            .detected_cell_type_count
                        ),
                        "top_breast_cell_type": (
                            gene_result
                            .top_breast_cell_type
                            or ""
                        ),
                        "highest_ncpm": (
                            gene_result.highest_ncpm
                        ),
                        "top_cell_type_max_read_count": (
                            gene_result
                            .top_cell_type_max_read_count
                        ),
                        "top_cell_type_cluster_count": (
                            gene_result
                            .top_cell_type_cluster_count
                        ),
                        "second_breast_cell_type": (
                            gene_result
                            .second_breast_cell_type
                            or ""
                        ),
                        "second_highest_ncpm": (
                            gene_result
                            .second_highest_ncpm
                        ),
                        "ncpm_gap_top_vs_second": (
                            gene_result
                            .ncpm_gap_top_vs_second
                        ),
                        "top_to_second_ncpm_ratio": (
                            gene_result
                            .top_to_second_ncpm_ratio
                            if gene_result
                            .top_to_second_ncpm_ratio
                            is not None
                            else ""
                        ),
                        "detected_cell_types": (
                            " | ".join(
                                gene_result
                                .detected_cell_types
                            )
                        ),
                        "review_status": (
                            gene_result.review_status
                        ),
                        "source": result.source,
                        "source_release": (
                            result.source_release
                        ),
                        "evidence_scope": (
                            "healthy_breast_reference"
                        ),
                    }
                )

    @staticmethod
    def _write_cell_type_summary(
        result: HPARetrievalResult,
        output_path: Path,
    ) -> None:
        """Writes cell-type summaries per gene."""

        fieldnames = [
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
        ]

        with output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for gene_result in (
                result.gene_summaries
            ):
                for rank, summary in enumerate(
                    gene_result
                    .cell_type_summaries,
                    start=1,
                ):
                    writer.writerow(
                        {
                            "gene": gene_result.gene,
                            "cell_type_rank": rank,
                            "cell_type": (
                                summary.cell_type
                            ),
                            "is_top_cell_type": (
                                rank == 1
                            ),
                            "max_ncpm": (
                                summary.max_ncpm
                            ),
                            "max_read_count": (
                                summary
                                .max_read_count
                            ),
                            "positive_record_count": (
                                summary
                                .positive_record_count
                            ),
                            "expressing_cluster_count": (
                                summary
                                .expressing_cluster_count
                            ),
                            "clusters": " | ".join(
                                summary.clusters
                            ),
                            "source": result.source,
                            "source_release": (
                                result
                                .source_release
                            ),
                            "evidence_scope": (
                                "healthy_breast_reference"
                            ),
                        }
                    )

    @staticmethod
    def _write_expression_records(
        result: HPARetrievalResult,
        output_path: Path,
    ) -> None:
        """Writes every positive HPA record."""

        fieldnames = [
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
        ]

        with output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for record in (
                result.expression_records
            ):
                writer.writerow(
                    {
                        "gene": record.gene,
                        "ensembl_gene_id": (
                            record
                            .ensembl_gene_id
                        ),
                        "tissue": record.tissue,
                        "cell_type": (
                            record.cell_type
                        ),
                        "cluster": record.cluster,
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
                        "source": record.source,
                        "source_release": (
                            result.source_release
                        ),
                    }
                )