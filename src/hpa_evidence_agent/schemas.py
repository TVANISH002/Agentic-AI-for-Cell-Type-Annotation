from __future__ import annotations

from collections import Counter
from datetime import datetime
from math import isclose
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


SourceStatus = Literal[
    "breast_expression_found",
    "gene_found_without_breast_expression",
    "gene_not_found_in_hpa",
]

ReviewStatus = Literal[
    "reviewed",
    "needs_review",
    "symbol_review_needed",
]


class StrictModel(BaseModel):
    """Reject unknown fields so report contracts remain predictable."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class HPAExpressionRecord(StrictModel):
    """One positive HPA breast-expression observation."""

    gene: str
    ensembl_gene_id: str
    tissue: str
    cell_type: str
    cluster: str
    read_count: float = Field(ge=0.0)
    ncpm: float = Field(gt=0.0)
    evidence_type: str
    evidence_scope: str
    source: str

    @field_validator("gene")
    @classmethod
    def normalize_gene(cls, value: str) -> str:
        """Store gene symbols in a consistent uppercase format."""

        normalized = value.strip().upper()

        if not normalized:
            raise ValueError(
                "gene cannot be empty."
            )

        return normalized

    @field_validator(
        "ensembl_gene_id",
        "tissue",
        "cell_type",
        "cluster",
        "evidence_type",
        "evidence_scope",
        "source",
    )
    @classmethod
    def require_text(cls, value: str) -> str:
        """Prevent empty text values in evidence records."""

        if not value.strip():
            raise ValueError(
                "Text fields cannot be empty."
            )

        return value.strip()


class HPACellTypeSummary(StrictModel):
    """Aggregated HPA expression for one gene and one cell type."""

    cell_type: str
    max_ncpm: float = Field(gt=0.0)
    max_read_count: float = Field(ge=0.0)
    positive_record_count: int = Field(gt=0)
    expressing_cluster_count: int = Field(gt=0)
    clusters: list[str]

    @field_validator("cell_type")
    @classmethod
    def require_cell_type(cls, value: str) -> str:
        """Prevent empty HPA cell-type names."""

        if not value.strip():
            raise ValueError(
                "cell_type cannot be empty."
            )

        return value.strip()

    @field_validator("clusters")
    @classmethod
    def validate_clusters(
        cls,
        values: list[str],
    ) -> list[str]:
        """Ensure HPA cluster names are clean and unique."""

        cleaned = [
            value.strip()
            for value in values
        ]

        if any(
            not value
            for value in cleaned
        ):
            raise ValueError(
                "clusters cannot contain empty values."
            )

        if len(cleaned) != len(set(cleaned)):
            raise ValueError(
                "clusters must be unique."
            )

        return cleaned

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Check that stored cluster counts match the cluster list."""

        if (
            self.expressing_cluster_count
            != len(self.clusters)
        ):
            raise ValueError(
                "expressing_cluster_count must equal "
                "the number of unique clusters."
            )

        if (
            self.positive_record_count
            < self.expressing_cluster_count
        ):
            raise ValueError(
                "positive_record_count cannot be smaller "
                "than expressing_cluster_count."
            )

        return self


class HPAGeneSummary(StrictModel):
    """Complete independent HPA summary for one panel gene."""

    gene: str
    gene_present_in_hpa: bool
    breast_expression_found: bool
    source_status: SourceStatus

    positive_record_count: int = Field(
        ge=0
    )

    detected_cell_type_count: int = Field(
        ge=0
    )

    detected_cell_types: list[str]

    top_breast_cell_type: str | None = None

    highest_ncpm: float = Field(
        ge=0.0
    )

    top_cell_type_max_read_count: float = Field(
        ge=0.0
    )

    top_cell_type_cluster_count: int = Field(
        ge=0
    )

    second_breast_cell_type: str | None = None

    second_highest_ncpm: float = Field(
        ge=0.0
    )

    ncpm_gap_top_vs_second: float = Field(
        ge=0.0
    )

    top_to_second_ncpm_ratio: float | None = Field(
        default=None,
        ge=0.0,
    )

    cell_type_summaries: list[
        HPACellTypeSummary
    ]

    review_status: ReviewStatus

    @field_validator("gene")
    @classmethod
    def normalize_gene(cls, value: str) -> str:
        """Store gene symbols in uppercase."""

        normalized = value.strip().upper()

        if not normalized:
            raise ValueError(
                "gene cannot be empty."
            )

        return normalized

    @field_validator("detected_cell_types")
    @classmethod
    def validate_detected_cell_types(
        cls,
        values: list[str],
    ) -> list[str]:
        """Ensure detected cell types are clean and unique."""

        cleaned = [
            value.strip()
            for value in values
        ]

        if any(
            not value
            for value in cleaned
        ):
            raise ValueError(
                "detected_cell_types cannot "
                "contain empty values."
            )

        if len(cleaned) != len(set(cleaned)):
            raise ValueError(
                "detected_cell_types must be unique."
            )

        return cleaned

    @field_validator(
        "top_breast_cell_type",
        "second_breast_cell_type",
    )
    @classmethod
    def clean_optional_text(
        cls,
        value: str | None,
    ) -> str | None:
        """Convert empty optional cell-type names to None."""

        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        """Check that the complete gene summary is consistent."""

        if (
            self.detected_cell_type_count
            != len(self.detected_cell_types)
        ):
            raise ValueError(
                "detected_cell_type_count must equal "
                "the number of detected_cell_types."
            )

        summary_cell_types = [
            summary.cell_type
            for summary
            in self.cell_type_summaries
        ]

        if (
            len(summary_cell_types)
            != len(set(summary_cell_types))
        ):
            raise ValueError(
                "cell_type_summaries must contain "
                "unique cell types."
            )

        if (
            summary_cell_types
            != self.detected_cell_types
        ):
            raise ValueError(
                "cell_type_summaries must follow "
                "the same ordered cell-type list "
                "as detected_cell_types."
            )

        if self.breast_expression_found:
            if not self.gene_present_in_hpa:
                raise ValueError(
                    "Breast expression cannot be found "
                    "when the gene is absent from HPA."
                )

            if (
                self.source_status
                != "breast_expression_found"
            ):
                raise ValueError(
                    "source_status must be "
                    "breast_expression_found."
                )

            if self.positive_record_count <= 0:
                raise ValueError(
                    "positive_record_count must be "
                    "positive when breast expression "
                    "is found."
                )

            if self.detected_cell_type_count <= 0:
                raise ValueError(
                    "At least one detected cell type "
                    "is required."
                )

            if not self.top_breast_cell_type:
                raise ValueError(
                    "top_breast_cell_type is required "
                    "when breast expression is found."
                )

            if self.highest_ncpm <= 0:
                raise ValueError(
                    "highest_ncpm must be positive "
                    "when expression is found."
                )

            first_summary = (
                self.cell_type_summaries[0]
            )

            if (
                self.top_breast_cell_type
                != first_summary.cell_type
            ):
                raise ValueError(
                    "top_breast_cell_type must match "
                    "the first cell-type summary."
                )

            if not isclose(
                self.highest_ncpm,
                first_summary.max_ncpm,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "highest_ncpm must match the "
                    "top cell-type summary."
                )

            if self.detected_cell_type_count == 1:
                if (
                    self.second_breast_cell_type
                    is not None
                ):
                    raise ValueError(
                        "second_breast_cell_type must "
                        "be empty when only one cell "
                        "type is detected."
                    )

                if self.second_highest_ncpm != 0.0:
                    raise ValueError(
                        "second_highest_ncpm must be "
                        "zero when only one cell type "
                        "is detected."
                    )

                if (
                    self.top_to_second_ncpm_ratio
                    is not None
                ):
                    raise ValueError(
                        "top_to_second_ncpm_ratio must "
                        "be empty when there is no "
                        "second cell type."
                    )

            else:
                second_summary = (
                    self.cell_type_summaries[1]
                )

                if (
                    self.second_breast_cell_type
                    != second_summary.cell_type
                ):
                    raise ValueError(
                        "second_breast_cell_type must "
                        "match the second cell-type "
                        "summary."
                    )

                if not isclose(
                    self.second_highest_ncpm,
                    second_summary.max_ncpm,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise ValueError(
                        "second_highest_ncpm must "
                        "match the second cell-type "
                        "summary."
                    )

                expected_gap = (
                    self.highest_ncpm
                    - self.second_highest_ncpm
                )

                if not isclose(
                    self.ncpm_gap_top_vs_second,
                    expected_gap,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise ValueError(
                        "ncpm_gap_top_vs_second "
                        "is inconsistent."
                    )

                expected_ratio = (
                    self.highest_ncpm
                    / self.second_highest_ncpm
                    if self.second_highest_ncpm > 0
                    else None
                )

                if expected_ratio is None:
                    if (
                        self.top_to_second_ncpm_ratio
                        is not None
                    ):
                        raise ValueError(
                            "top_to_second_ncpm_ratio "
                            "must be empty when the "
                            "second nCPM is zero."
                        )

                elif (
                    self.top_to_second_ncpm_ratio
                    is None
                    or not isclose(
                        self.top_to_second_ncpm_ratio,
                        expected_ratio,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                ):
                    raise ValueError(
                        "top_to_second_ncpm_ratio "
                        "is inconsistent."
                    )

            return self

        if self.gene_present_in_hpa:
            expected_status: SourceStatus = (
                "gene_found_without_breast_expression"
            )

        else:
            expected_status = (
                "gene_not_found_in_hpa"
            )

        if self.source_status != expected_status:
            raise ValueError(
                "source_status must be "
                f"{expected_status}."
            )

        has_unexpected_expression = any(
            [
                self.positive_record_count != 0,
                self.detected_cell_type_count != 0,
                bool(self.detected_cell_types),
                (
                    self.top_breast_cell_type
                    is not None
                ),
                self.highest_ncpm != 0.0,
                (
                    self.top_cell_type_max_read_count
                    != 0.0
                ),
                (
                    self.top_cell_type_cluster_count
                    != 0
                ),
                (
                    self.second_breast_cell_type
                    is not None
                ),
                self.second_highest_ncpm != 0.0,
                self.ncpm_gap_top_vs_second != 0.0,
                (
                    self.top_to_second_ncpm_ratio
                    is not None
                ),
                bool(self.cell_type_summaries),
            ]
        )

        if has_unexpected_expression:
            raise ValueError(
                "Genes without breast expression "
                "cannot contain positive expression "
                "summaries."
            )

        return self


class HPAPanelRunSummary(StrictModel):
    """Coverage totals for one panel-wide HPA run."""

    requested_gene_count: int = Field(
        ge=0
    )

    breast_expression_found_count: int = Field(
        ge=0
    )

    gene_found_without_breast_expression_count: int = Field(
        ge=0
    )

    gene_not_found_in_hpa_count: int = Field(
        ge=0
    )

    expression_record_count: int = Field(
        ge=0
    )

    @model_validator(mode="after")
    def validate_gene_counts(self) -> Self:
        """Ensure all requested genes are accounted for."""

        covered_total = (
            self.breast_expression_found_count
            + (
                self
                .gene_found_without_breast_expression_count
            )
            + self.gene_not_found_in_hpa_count
        )

        if covered_total != self.requested_gene_count:
            raise ValueError(
                "HPA coverage counts must sum "
                "to requested_gene_count."
            )

        return self


class HPAPanelEvidence(StrictModel):
    """Complete independent HPA result for all panel genes."""

    cluster_id: Literal["panel"] = "panel"
    run_mode: Literal["all"] = "all"

    query_tissue: str
    source: str
    source_release: str
    source_url: str
    source_sha256: str

    source_file_size_bytes: int = Field(
        gt=0
    )

    retrieved_at_utc: datetime

    selected_genes: list[str]

    summary: HPAPanelRunSummary

    gene_summaries: list[
        HPAGeneSummary
    ]

    expression_records: list[
        HPAExpressionRecord
    ]

    warnings: list[str] = Field(
        default_factory=list
    )

    @field_validator("selected_genes")
    @classmethod
    def validate_selected_genes(
        cls,
        values: list[str],
    ) -> list[str]:
        """Normalize and validate the panel gene list."""

        cleaned = [
            value.strip().upper()
            for value in values
        ]

        if any(
            not value
            for value in cleaned
        ):
            raise ValueError(
                "selected_genes cannot contain "
                "empty values."
            )

        if len(cleaned) != len(set(cleaned)):
            raise ValueError(
                "selected_genes must be unique."
            )

        return cleaned

    @field_validator(
        "query_tissue",
        "source",
        "source_release",
        "source_url",
        "source_sha256",
    )
    @classmethod
    def require_text(cls, value: str) -> str:
        """Prevent empty panel metadata."""

        if not value.strip():
            raise ValueError(
                "Text metadata fields cannot be empty."
            )

        return value.strip()

    @field_validator("warnings")
    @classmethod
    def clean_warnings(
        cls,
        values: list[str],
    ) -> list[str]:
        """Remove blank warning messages."""

        cleaned = [
            value.strip()
            for value in values
        ]

        return [
            value
            for value in cleaned
            if value
        ]

    @model_validator(mode="after")
    def validate_panel_alignment(self) -> Self:
        """Check that all panel evidence stays aligned."""

        if (
            self.summary.requested_gene_count
            != len(self.selected_genes)
        ):
            raise ValueError(
                "requested_gene_count must equal "
                "selected_genes length."
            )

        summary_genes = [
            summary.gene
            for summary in self.gene_summaries
        ]

        if summary_genes != self.selected_genes:
            raise ValueError(
                "gene_summaries must align exactly "
                "with selected_genes and preserve "
                "the same order."
            )

        expression_genes = {
            record.gene
            for record in self.expression_records
        }

        if not expression_genes.issubset(
            set(self.selected_genes)
        ):
            raise ValueError(
                "Expression records contain genes "
                "outside the panel."
            )

        status_counts = Counter(
            summary.source_status
            for summary in self.gene_summaries
        )

        expected_counts = {
            "breast_expression_found": (
                self
                .summary
                .breast_expression_found_count
            ),
            "gene_found_without_breast_expression": (
                self
                .summary
                .gene_found_without_breast_expression_count
            ),
            "gene_not_found_in_hpa": (
                self
                .summary
                .gene_not_found_in_hpa_count
            ),
        }

        for (
            status,
            expected_count,
        ) in expected_counts.items():
            if (
                status_counts[status]
                != expected_count
            ):
                raise ValueError(
                    "Panel status count is "
                    f"inconsistent for {status}."
                )

        if (
            self.summary.expression_record_count
            != len(self.expression_records)
        ):
            raise ValueError(
                "expression_record_count must equal "
                "the number of expression_records."
            )

        record_counts = Counter(
            record.gene
            for record in self.expression_records
        )

        for summary in self.gene_summaries:
            if (
                record_counts[summary.gene]
                != summary.positive_record_count
            ):
                raise ValueError(
                    "Gene positive_record_count "
                    "is inconsistent for "
                    f"{summary.gene}."
                )

        return self


class HPAClusterGeneEvidence(StrictModel):
    """One cluster marker joined to its independent HPA evidence."""

    cluster_id: str | int
    gene: str

    rank_within_cluster: int = Field(
        gt=0
    )

    score: float | None = None
    log_fold_change: float | None = None

    adjusted_p_value: float | None = Field(
        default=None,
        ge=0.0,
    )

    hpa: HPAGeneSummary

    @field_validator("gene")
    @classmethod
    def normalize_gene(cls, value: str) -> str:
        """Normalize projected cluster gene symbols."""

        normalized = value.strip().upper()

        if not normalized:
            raise ValueError(
                "gene cannot be empty."
            )

        return normalized

    @model_validator(mode="after")
    def validate_gene_alignment(self) -> Self:
        """Ensure the cluster gene matches its HPA evidence."""

        if self.gene != self.hpa.gene:
            raise ValueError(
                "Cluster gene must match the nested "
                "HPA gene summary."
            )

        return self


class HPAClusterEvidence(StrictModel):
    """Independent HPA evidence projected onto one marker cluster."""

    cluster_id: str | int
    query_tissue: str
    source: str
    source_release: str

    marker_gene_count: int = Field(
        ge=0
    )

    breast_expression_found_count: int = Field(
        ge=0
    )

    gene_found_without_breast_expression_count: int = Field(
        ge=0
    )

    gene_not_found_in_hpa_count: int = Field(
        ge=0
    )

    genes: list[
        HPAClusterGeneEvidence
    ]

    warnings: list[str] = Field(
        default_factory=list
    )

    @field_validator(
        "query_tissue",
        "source",
        "source_release",
    )
    @classmethod
    def require_text(cls, value: str) -> str:
        """Prevent empty cluster metadata."""

        if not value.strip():
            raise ValueError(
                "Text metadata fields cannot be empty."
            )

        return value.strip()

    @field_validator("warnings")
    @classmethod
    def clean_warnings(
        cls,
        values: list[str],
    ) -> list[str]:
        """Remove blank cluster warnings."""

        cleaned = [
            value.strip()
            for value in values
        ]

        return [
            value
            for value in cleaned
            if value
        ]

    @model_validator(mode="after")
    def validate_cluster_alignment(self) -> Self:
        """Check projected cluster evidence for consistency."""

        if self.marker_gene_count != len(self.genes):
            raise ValueError(
                "marker_gene_count must equal "
                "the number of genes."
            )

        ranks = [
            item.rank_within_cluster
            for item in self.genes
        ]

        if len(ranks) != len(set(ranks)):
            raise ValueError(
                "Cluster marker ranks must be unique."
            )

        genes = [
            item.gene
            for item in self.genes
        ]

        if len(genes) != len(set(genes)):
            raise ValueError(
                "Cluster genes must be unique."
            )

        if any(
            str(item.cluster_id)
            != str(self.cluster_id)
            for item in self.genes
        ):
            raise ValueError(
                "Nested cluster IDs must match "
                "the parent cluster ID."
            )

        status_counts = Counter(
            item.hpa.source_status
            for item in self.genes
        )

        expected_counts = {
            "breast_expression_found": (
                self
                .breast_expression_found_count
            ),
            "gene_found_without_breast_expression": (
                self
                .gene_found_without_breast_expression_count
            ),
            "gene_not_found_in_hpa": (
                self
                .gene_not_found_in_hpa_count
            ),
        }

        for (
            status,
            expected_count,
        ) in expected_counts.items():
            if (
                status_counts[status]
                != expected_count
            ):
                raise ValueError(
                    "Cluster status count is "
                    f"inconsistent for {status}."
                )

        covered_total = sum(
            expected_counts.values()
        )

        if covered_total != self.marker_gene_count:
            raise ValueError(
                "Cluster HPA coverage counts must "
                "sum to marker_gene_count."
            )

        return self
   
   
   
   
   
   
   
   
   
   