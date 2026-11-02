from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone

from hpa_evidence_agent.config import (
    DEFAULT_CONFIG,
    HPAEvidenceConfig,
)
from hpa_evidence_agent.hpa_source_loader import (
    HPASourceLoadResult,
    HPASourceRecord,
)
from hpa_evidence_agent.schemas import (
    HPACellTypeSummary,
    HPAExpressionRecord,
    HPAGeneSummary,
    HPAPanelEvidence,
    HPAPanelRunSummary,
)


class HPAEvidenceBuilder:
    """Build validated gene-level evidence from filtered HPA rows."""

    PANEL_WARNINGS = [
        (
            "HPA breast evidence in this workflow comes from "
            "a healthy-breast reference."
        ),
        (
            "Every positive nCPM record is preserved as raw "
            "expression evidence."
        ),
        (
            "The pipeline does not automatically label an nCPM "
            "value as biologically weak, moderate or strong."
        ),
        (
            "Top-versus-second measurements are descriptive and "
            "are not final marker confidence scores."
        ),
        (
            "HPA expression evidence remains separate from "
            "CellMarker and PanglaoDB marker votes."
        ),
    ]

    def __init__(
        self,
        config: HPAEvidenceConfig = DEFAULT_CONFIG,
    ) -> None:
        """Store and validate the independent HPA configuration."""

        config.validate()
        self.config = config

    @staticmethod
    def _sort_cell_type_summaries(
        summaries: Sequence[HPACellTypeSummary],
    ) -> list[HPACellTypeSummary]:
        """Sort cell types by strongest expression with stable ties."""

        return sorted(
            summaries,
            key=lambda item: (
                -item.max_ncpm,
                -item.max_read_count,
                item.cell_type.casefold(),
            ),
        )

    @classmethod
    def _build_cell_type_summaries(
        cls,
        records: Sequence[HPASourceRecord],
    ) -> list[HPACellTypeSummary]:
        """Aggregate one gene's HPA rows into cell-type summaries."""

        records_by_cell_type: dict[
            str,
            list[HPASourceRecord],
        ] = defaultdict(list)

        for record in records:
            records_by_cell_type[
                record.cell_type
            ].append(record)

        summaries: list[HPACellTypeSummary] = []

        for cell_type, cell_type_records in (
            records_by_cell_type.items()
        ):
            clusters = sorted(
                {
                    record.cluster
                    for record in cell_type_records
                },
                key=str.casefold,
            )

            summaries.append(
                HPACellTypeSummary(
                    cell_type=cell_type,
                    max_ncpm=max(
                        record.ncpm
                        for record in cell_type_records
                    ),
                    max_read_count=max(
                        record.read_count
                        for record in cell_type_records
                    ),
                    positive_record_count=len(
                        cell_type_records
                    ),
                    expressing_cluster_count=len(
                        clusters
                    ),
                    clusters=clusters,
                )
            )

        return cls._sort_cell_type_summaries(
            summaries
        )

    @staticmethod
    def _empty_gene_summary(
        gene: str,
        *,
        gene_present_in_hpa: bool,
    ) -> HPAGeneSummary:
        """Create a complete zero-expression summary for one gene."""

        source_status = (
            "gene_found_without_breast_expression"
            if gene_present_in_hpa
            else "gene_not_found_in_hpa"
        )

        review_status = (
            "needs_review"
            if gene_present_in_hpa
            else "symbol_review_needed"
        )

        return HPAGeneSummary(
            gene=gene,
            gene_present_in_hpa=(
                gene_present_in_hpa
            ),
            breast_expression_found=False,
            source_status=source_status,
            positive_record_count=0,
            detected_cell_type_count=0,
            detected_cell_types=[],
            top_breast_cell_type=None,
            highest_ncpm=0.0,
            top_cell_type_max_read_count=0.0,
            top_cell_type_cluster_count=0,
            second_breast_cell_type=None,
            second_highest_ncpm=0.0,
            ncpm_gap_top_vs_second=0.0,
            top_to_second_ncpm_ratio=None,
            cell_type_summaries=[],
            review_status=review_status,
        )

    @classmethod
    def _expressed_gene_summary(
        cls,
        gene: str,
        records: Sequence[HPASourceRecord],
    ) -> HPAGeneSummary:
        """Create one gene summary from its positive breast rows."""

        if not records:
            raise ValueError(
                "records cannot be empty when building an "
                f"expressed gene summary for {gene}."
            )

        cell_type_summaries = (
            cls._build_cell_type_summaries(
                records
            )
        )

        top_summary = cell_type_summaries[0]

        second_summary = (
            cell_type_summaries[1]
            if len(cell_type_summaries) > 1
            else None
        )

        second_highest_ncpm = (
            second_summary.max_ncpm
            if second_summary is not None
            else 0.0
        )

        ncpm_gap = (
            top_summary.max_ncpm
            - second_highest_ncpm
            if second_summary is not None
            else 0.0
        )

        top_to_second_ratio = (
            top_summary.max_ncpm
            / second_summary.max_ncpm
            if (
                second_summary is not None
                and second_summary.max_ncpm > 0.0
            )
            else None
        )

        detected_cell_types = [
            summary.cell_type
            for summary in cell_type_summaries
        ]

        return HPAGeneSummary(
            gene=gene,
            gene_present_in_hpa=True,
            breast_expression_found=True,
            source_status=(
                "breast_expression_found"
            ),
            positive_record_count=len(records),
            detected_cell_type_count=len(
                detected_cell_types
            ),
            detected_cell_types=(
                detected_cell_types
            ),
            top_breast_cell_type=(
                top_summary.cell_type
            ),
            highest_ncpm=top_summary.max_ncpm,
            top_cell_type_max_read_count=(
                top_summary.max_read_count
            ),
            top_cell_type_cluster_count=(
                top_summary.expressing_cluster_count
            ),
            second_breast_cell_type=(
                second_summary.cell_type
                if second_summary is not None
                else None
            ),
            second_highest_ncpm=(
                second_highest_ncpm
            ),
            ncpm_gap_top_vs_second=ncpm_gap,
            top_to_second_ncpm_ratio=(
                top_to_second_ratio
            ),
            cell_type_summaries=(
                cell_type_summaries
            ),
            review_status="needs_review",
        )

    def _expression_record(
        self,
        record: HPASourceRecord,
    ) -> HPAExpressionRecord:
        """Convert one filtered source row into validated evidence."""

        return HPAExpressionRecord(
            gene=record.gene,
            ensembl_gene_id=(
                record.ensembl_gene_id
            ),
            tissue=record.tissue,
            cell_type=record.cell_type,
            cluster=record.cluster,
            read_count=record.read_count,
            ncpm=record.ncpm,
            evidence_type=(
                self.config.evidence_type
            ),
            evidence_scope=(
                self.config.evidence_scope
            ),
            source=self.config.source_name,
        )

    def build(
        self,
        source_result: HPASourceLoadResult,
    ) -> HPAPanelEvidence:
        """Build the complete independent panel-wide HPA result."""

        selected_genes = list(
            source_result.selected_genes
        )

        records_by_gene: dict[
            str,
            list[HPASourceRecord],
        ] = defaultdict(list)

        for record in (
            source_result.breast_expression_records
        ):
            records_by_gene[record.gene].append(
                record
            )

        gene_summaries: list[
            HPAGeneSummary
        ] = []

        expression_records: list[
            HPAExpressionRecord
        ] = []

        for gene in selected_genes:
            gene_records = records_by_gene.get(
                gene,
                [],
            )

            if gene_records:
                gene_summaries.append(
                    self._expressed_gene_summary(
                        gene,
                        gene_records,
                    )
                )

                expression_records.extend(
                    self._expression_record(record)
                    for record in gene_records
                )

                continue

            gene_summaries.append(
                self._empty_gene_summary(
                    gene,
                    gene_present_in_hpa=(
                        gene
                        in source_result
                        .genes_present_in_hpa
                    ),
                )
            )

        breast_expression_found_count = sum(
            summary.breast_expression_found
            for summary in gene_summaries
        )

        found_without_breast_count = sum(
            summary.source_status
            == "gene_found_without_breast_expression"
            for summary in gene_summaries
        )

        gene_not_found_count = sum(
            summary.source_status
            == "gene_not_found_in_hpa"
            for summary in gene_summaries
        )

        run_summary = HPAPanelRunSummary(
            requested_gene_count=len(
                selected_genes
            ),
            breast_expression_found_count=(
                breast_expression_found_count
            ),
            gene_found_without_breast_expression_count=(
                found_without_breast_count
            ),
            gene_not_found_in_hpa_count=(
                gene_not_found_count
            ),
            expression_record_count=len(
                expression_records
            ),
        )

        return HPAPanelEvidence(
            query_tissue=(
                self.config.query_tissue
            ),
            source=self.config.source_name,
            source_release=(
                self.config.source_release
            ),
            source_url=source_result.source_url,
            source_sha256=(
                source_result.source_sha256
            ),
            source_file_size_bytes=(
                source_result
                .source_file_size_bytes
            ),
            retrieved_at_utc=datetime.now(
                timezone.utc
            ),
            selected_genes=selected_genes,
            summary=run_summary,
            gene_summaries=gene_summaries,
            expression_records=(
                expression_records
            ),
            warnings=list(
                self.PANEL_WARNINGS
            ),
        )


def build_hpa_evidence(
    source_result: HPASourceLoadResult,
    config: HPAEvidenceConfig = DEFAULT_CONFIG,
) -> HPAPanelEvidence:
    """Convenience function for building panel-wide HPA evidence."""

    return HPAEvidenceBuilder(
        config
    ).build(
        source_result
    )