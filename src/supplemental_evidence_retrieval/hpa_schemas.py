from __future__ import annotations

from datetime import datetime
from math import isclose
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


HpaRunMode = Literal[
    "pilot",
    "all",
]

HpaGeneStatus = Literal[
    "breast_expression_found",
    "gene_found_without_breast_expression",
    "gene_not_found_in_hpa",
]


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HPAClusterExpressionRecord(StrictSchema):
    gene: str
    ensembl_gene_id: str
    tissue: str
    cell_type: str
    cluster: str

    read_count: float = Field(ge=0)
    ncpm: float = Field(gt=0)

    evidence_type: Literal[
        "single_cell_rna_expression"
    ] = "single_cell_rna_expression"

    evidence_scope: Literal[
        "healthy_breast_reference"
    ] = "healthy_breast_reference"

    source: Literal[
        "Human Protein Atlas"
    ] = "Human Protein Atlas"


class HPACellTypeExpressionSummary(StrictSchema):
    cell_type: str
    max_ncpm: float = Field(gt=0)
    max_read_count: float = Field(ge=0)

    positive_record_count: int = Field(gt=0)
    expressing_cluster_count: int = Field(gt=0)

    clusters: list[str] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_cluster_count(
        self,
    ) -> HPACellTypeExpressionSummary:
        if len(self.clusters) != self.expressing_cluster_count:
            raise ValueError(
                "expressing_cluster_count must match "
                "the number of unique clusters."
            )

        return self


class HPAGeneExpressionSummary(StrictSchema):
    gene: str

    gene_present_in_hpa: bool
    breast_expression_found: bool
    source_status: HpaGeneStatus

    positive_record_count: int = Field(ge=0)
    detected_cell_type_count: int = Field(ge=0)

    detected_cell_types: list[str] = Field(
        default_factory=list
    )

    top_breast_cell_type: str | None = None
    highest_ncpm: float = Field(ge=0)
    top_cell_type_max_read_count: float = Field(ge=0)
    top_cell_type_cluster_count: int = Field(ge=0)

    second_breast_cell_type: str | None = None
    second_highest_ncpm: float = Field(ge=0)

    ncpm_gap_top_vs_second: float = Field(ge=0)

    top_to_second_ncpm_ratio: float | None = Field(
        default=None,
        gt=0,
    )

    cell_type_summaries: list[
        HPACellTypeExpressionSummary
    ] = Field(default_factory=list)

    review_status: Literal[
        "needs_review"
    ] = "needs_review"

    @model_validator(mode="after")
    def validate_gene_summary(
        self,
    ) -> HPAGeneExpressionSummary:
        expected_found = bool(
            self.cell_type_summaries
        )

        if self.breast_expression_found != expected_found:
            raise ValueError(
                "breast_expression_found must match "
                "cell_type_summaries."
            )

        expected_cell_types = [
            summary.cell_type
            for summary in self.cell_type_summaries
        ]

        if self.detected_cell_types != expected_cell_types:
            raise ValueError(
                "detected_cell_types must match "
                "cell_type_summaries."
            )

        if (
            self.detected_cell_type_count
            != len(self.detected_cell_types)
        ):
            raise ValueError(
                "detected_cell_type_count must match "
                "detected_cell_types."
            )

        expected_record_count = sum(
            summary.positive_record_count
            for summary in self.cell_type_summaries
        )

        if self.positive_record_count != expected_record_count:
            raise ValueError(
                "positive_record_count must match "
                "the summarized HPA records."
            )

        if not self.breast_expression_found:
            if self.top_breast_cell_type is not None:
                raise ValueError(
                    "A gene without breast expression cannot "
                    "have a top breast cell type."
                )

            if self.second_breast_cell_type is not None:
                raise ValueError(
                    "A gene without breast expression cannot "
                    "have a second breast cell type."
                )

            if self.highest_ncpm != 0:
                raise ValueError(
                    "A gene without breast expression must "
                    "have highest_ncpm=0."
                )

            if self.second_highest_ncpm != 0:
                raise ValueError(
                    "A gene without breast expression must "
                    "have second_highest_ncpm=0."
                )

            if self.ncpm_gap_top_vs_second != 0:
                raise ValueError(
                    "A gene without breast expression must "
                    "have ncpm_gap_top_vs_second=0."
                )

            if self.top_to_second_ncpm_ratio is not None:
                raise ValueError(
                    "A gene without breast expression cannot "
                    "have a top-to-second ratio."
                )

            return self

        if not self.gene_present_in_hpa:
            raise ValueError(
                "A gene with breast expression must be "
                "present in HPA."
            )

        first_summary = self.cell_type_summaries[0]

        if self.top_breast_cell_type != first_summary.cell_type:
            raise ValueError(
                "top_breast_cell_type must match the "
                "highest-ranked cell-type summary."
            )

        if not isclose(
            self.highest_ncpm,
            first_summary.max_ncpm,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "highest_ncpm must match the first "
                "cell-type summary."
            )

        if not isclose(
            self.top_cell_type_max_read_count,
            first_summary.max_read_count,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "top_cell_type_max_read_count must match "
                "the first cell-type summary."
            )

        if (
            self.top_cell_type_cluster_count
            != first_summary.expressing_cluster_count
        ):
            raise ValueError(
                "top_cell_type_cluster_count must match "
                "the first cell-type summary."
            )

        if len(self.cell_type_summaries) == 1:
            if self.second_breast_cell_type is not None:
                raise ValueError(
                    "A gene with one cell type cannot have "
                    "a second breast cell type."
                )

            if self.second_highest_ncpm != 0:
                raise ValueError(
                    "A gene with one cell type must have "
                    "second_highest_ncpm=0."
                )

            if not isclose(
                self.ncpm_gap_top_vs_second,
                self.highest_ncpm,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "For one cell type, the nCPM gap must "
                    "equal the highest nCPM."
                )

            if self.top_to_second_ncpm_ratio is not None:
                raise ValueError(
                    "A gene with one cell type cannot have "
                    "a top-to-second ratio."
                )

            return self

        second_summary = self.cell_type_summaries[1]

        if self.second_breast_cell_type != second_summary.cell_type:
            raise ValueError(
                "second_breast_cell_type must match the "
                "second-ranked cell-type summary."
            )

        if not isclose(
            self.second_highest_ncpm,
            second_summary.max_ncpm,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "second_highest_ncpm must match the "
                "second-ranked cell-type summary."
            )

        expected_gap = (
            self.highest_ncpm
            - self.second_highest_ncpm
        )

        if not isclose(
            self.ncpm_gap_top_vs_second,
            expected_gap,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "ncpm_gap_top_vs_second is inconsistent."
            )

        expected_ratio = (
            self.highest_ncpm
            / self.second_highest_ncpm
        )

        if self.top_to_second_ncpm_ratio is None:
            raise ValueError(
                "A gene with two or more cell types must "
                "have a top-to-second ratio."
            )

        if not isclose(
            self.top_to_second_ncpm_ratio,
            expected_ratio,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "top_to_second_ncpm_ratio is inconsistent."
            )

        return self


class HPARunSummary(StrictSchema):
    requested_gene_count: int = Field(ge=0)

    breast_expression_found_count: int = Field(ge=0)

    gene_found_without_breast_expression_count: int = Field(
        ge=0
    )

    gene_not_found_in_hpa_count: int = Field(ge=0)

    expression_record_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(
        self,
    ) -> HPARunSummary:
        classified_count = (
            self.breast_expression_found_count
            + self.gene_found_without_breast_expression_count
            + self.gene_not_found_in_hpa_count
        )

        if classified_count != self.requested_gene_count:
            raise ValueError(
                "HPA gene-status counts must equal "
                "requested_gene_count."
            )

        return self


class HPARetrievalResult(StrictSchema):
    cluster_id: int | str
    run_mode: HpaRunMode
    query_tissue: str

    source: Literal[
        "Human Protein Atlas"
    ] = "Human Protein Atlas"

    source_release: str
    source_url: str

    source_sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    source_file_size_bytes: int = Field(gt=0)
    retrieved_at_utc: datetime

    selected_genes: list[str] = Field(
        default_factory=list
    )

    summary: HPARunSummary

    gene_summaries: list[
        HPAGeneExpressionSummary
    ] = Field(default_factory=list)

    expression_records: list[
        HPAClusterExpressionRecord
    ] = Field(default_factory=list)

    warnings: list[str] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_result(
        self,
    ) -> HPARetrievalResult:
        if len(self.selected_genes) != len(
            set(self.selected_genes)
        ):
            raise ValueError(
                "selected_genes must be unique."
            )

        summary_genes = [
            result.gene
            for result in self.gene_summaries
        ]

        if summary_genes != self.selected_genes:
            raise ValueError(
                "gene_summaries must follow the exact "
                "selected_genes order."
            )

        if (
            len(self.selected_genes)
            != self.summary.requested_gene_count
        ):
            raise ValueError(
                "selected_genes count must match "
                "requested_gene_count."
            )

        if (
            len(self.expression_records)
            != self.summary.expression_record_count
        ):
            raise ValueError(
                "expression_records count must match "
                "expression_record_count."
            )

        return self