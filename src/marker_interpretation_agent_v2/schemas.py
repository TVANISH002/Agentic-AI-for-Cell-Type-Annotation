
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


EvidenceCategory = Literal[
    "direct_marker_supported",
    "expression_only",
    "ambiguous_expression",
    "broad_expression",
    "trace_expression",
    "symbol_review_needed",
    "unresolved",
]

ReviewSeverity = Literal["info", "warning"]

FollowUpAction = Literal[
    "none",
    "symbol_review",
    "targeted_literature_retrieval",
    "manual_review",
]


def _normalize_gene(value: str) -> str:
    """Normalize a gene symbol to uppercase."""

    gene = str(value).strip().upper()

    if not gene:
        raise ValueError("Gene symbol cannot be empty.")

    return gene


def _clean_strings(values: list[str] | tuple[str, ...] | None) -> list[str]:
    """Remove blanks and duplicate strings deterministically."""

    if not values:
        return []

    return sorted({str(value).strip() for value in values if str(value).strip()})


def _clean_strings_preserve_order(
    values: list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Remove blanks and duplicates while preserving the input order."""

    if not values:
        return []

    cleaned: list[str] = []
    seen: set[str] = set()

    for value in values:
        item = str(value).strip()
        key = item.casefold()

        if item and key not in seen:
            seen.add(key)
            cleaned.append(item)

    return cleaned


class StrictModel(BaseModel):
    """Base model used by all Agent 2 V2 schemas."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class InputGene(StrictModel):
    """Cluster-level differential-expression information for one gene."""

    gene: str
    rank_within_cluster: int = Field(ge=1)
    score: float | None = None
    log_fold_change: float | None = None
    p_value: float | None = Field(default=None, ge=0.0)
    adjusted_p_value: float | None = Field(default=None, ge=0.0)
    pct_nz_group: float | None = Field(default=None, ge=0.0, le=1.0)
    pct_nz_reference: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("gene")
    @classmethod
    def normalize_gene(cls, value: str) -> str:
        return _normalize_gene(value)


class HPAEvidenceSummary(StrictModel):
    """Healthy-breast expression summary returned by the HPA workflow."""

    queried: bool = False
    gene_present_in_hpa: bool = False
    breast_expression_found: bool = False

    top_cell_type: str | None = None
    top_ncpm: float | None = Field(default=None, ge=0.0)

    second_cell_type: str | None = None
    second_ncpm: float | None = Field(default=None, ge=0.0)

    top_to_second_ncpm_ratio: float | None = Field(default=None, ge=0.0)

    detected_cell_type_count: int = Field(default=0, ge=0)
    highest_read_count: float | None = Field(default=None, ge=0.0)

    review_pattern: str | None = None

    source: str = "Human Protein Atlas"
    source_url: str | None = None

    cell_type_expression: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_consistency(self) -> "HPAEvidenceSummary":
        if not self.queried:
            if self.gene_present_in_hpa or self.breast_expression_found:
                raise ValueError(
                    "An unqueried HPA record cannot report gene or "
                    "breast-expression matches."
                )

        if self.breast_expression_found and not self.gene_present_in_hpa:
            raise ValueError(
                "breast_expression_found requires gene_present_in_hpa=True."
            )

        if self.top_to_second_ncpm_ratio is not None and self.second_ncpm is None:
            raise ValueError("top_to_second_ncpm_ratio requires second_ncpm.")

        self.warnings = _clean_strings(self.warnings)

        return self


class PanelGeneEvidence(StrictModel):
    """Reusable panel-wide evidence for one unique gene."""

    gene: str
    official_symbol: str | None = None

    aliases: list[str] = Field(default_factory=list)

    marker_status: str = "unresolved"

    context_matched_cell_types: list[str] = Field(default_factory=list)
    context_matched_database_evidence: list[dict[str, Any]] = Field(
        default_factory=list
    )
    global_database_evidence: list[dict[str, Any]] = Field(default_factory=list)

    evidence_sources: list[str] = Field(default_factory=list)
    cell_ontology_ids: list[str] = Field(default_factory=list)
    tissue_contexts: list[str] = Field(default_factory=list)
    disease_contexts: list[str] = Field(default_factory=list)

    hpa: HPAEvidenceSummary = Field(default_factory=HPAEvidenceSummary)

    warnings: list[str] = Field(default_factory=list)

    @field_validator("gene")
    @classmethod
    def normalize_gene(cls, value: str) -> str:
        return _normalize_gene(value)

    @field_validator("official_symbol")
    @classmethod
    def normalize_official_symbol(cls, value: str | None) -> str | None:
        if value is None:
            return None

        return _normalize_gene(value)

    @model_validator(mode="after")
    def normalize_lists(self) -> "PanelGeneEvidence":
        self.aliases = [
            _normalize_gene(value) for value in _clean_strings(self.aliases)
        ]
        self.context_matched_cell_types = _clean_strings(
            self.context_matched_cell_types
        )
        self.evidence_sources = _clean_strings(self.evidence_sources)
        self.cell_ontology_ids = _clean_strings(self.cell_ontology_ids)
        self.tissue_contexts = _clean_strings(self.tissue_contexts)
        self.disease_contexts = _clean_strings(self.disease_contexts)
        self.warnings = _clean_strings(self.warnings)

        return self

    @property
    def direct_marker_supported(self) -> bool:
        """Return whether tissue-context marker evidence exists."""

        return bool(self.context_matched_cell_types)


class PanelEvidenceCatalog(StrictModel):
    """One-time evidence catalog shared by every cluster."""

    schema_name: str = "PanelEvidenceCatalog"
    schema_version: str = "1.0"

    species: str
    tissue: str
    platform: str

    panel_gene_count: int = Field(ge=0)

    genes: list[PanelGeneEvidence]

    unresolved_genes: list[str] = Field(default_factory=list)

    source_files: dict[str, str] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_catalog(self) -> "PanelEvidenceCatalog":
        gene_names = [item.gene for item in self.genes]

        if len(gene_names) != len(set(gene_names)):
            raise ValueError("PanelEvidenceCatalog contains duplicate genes.")

        if self.panel_gene_count != len(self.genes):
            raise ValueError(
                "panel_gene_count must equal the number of gene records."
            )

        self.unresolved_genes = sorted(
            {_normalize_gene(value) for value in self.unresolved_genes}
        )
        self.warnings = _clean_strings(self.warnings)

        return self


class GeneStatistics(StrictModel):
    """Derived statistics used for ranking and interpretation."""

    source_agreement_count: int = Field(default=0, ge=0)
    marker_specificity_score: float = Field(default=0.0, ge=0.0)
    context_evidence_count: int = Field(default=0, ge=0)
    cluster_gene_rank: int = Field(ge=1)
    log_fold_change: float | None = None


class GeneInterpretation(StrictModel):
    """Agent 2 interpretation of one cluster marker gene."""

    gene: str
    direct_marker_supported: bool
    evidence_category: EvidenceCategory

    supported_cell_types: list[str] = Field(default_factory=list)
    context_evidence: list[dict[str, Any]] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)

    statistics: GeneStatistics
    hpa: HPAEvidenceSummary

    warnings: list[str] = Field(default_factory=list)

    @field_validator("gene")
    @classmethod
    def normalize_gene(cls, value: str) -> str:
        return _normalize_gene(value)

    @model_validator(mode="after")
    def validate_interpretation(self) -> "GeneInterpretation":
        self.supported_cell_types = _clean_strings(self.supported_cell_types)
        self.evidence_sources = _clean_strings(self.evidence_sources)
        self.warnings = _clean_strings(self.warnings)

        if self.direct_marker_supported and not self.supported_cell_types:
            raise ValueError(
                "direct_marker_supported=True requires supported_cell_types."
            )

        if (
            self.evidence_category == "direct_marker_supported"
            and not self.direct_marker_supported
        ):
            raise ValueError(
                "direct_marker_supported category requires direct marker evidence."
            )

        if (
            self.direct_marker_supported
            and self.evidence_category != "direct_marker_supported"
        ):
            raise ValueError(
                "Direct marker genes must use the direct_marker_supported category."
            )

        return self


class RankedMarkerGene(StrictModel):
    """One directly supported marker gene after ranking."""

    marker_rank: int = Field(ge=1)
    gene: str
    source_agreement_count: int = Field(ge=0)
    marker_specificity_score: float = Field(ge=0.0)
    context_evidence_count: int = Field(ge=0)
    cluster_gene_rank: int = Field(ge=1)
    log_fold_change: float | None = None

    supported_cell_types: list[str] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)

    ranking_reason: str

    @field_validator("gene")
    @classmethod
    def normalize_gene(cls, value: str) -> str:
        return _normalize_gene(value)

    @model_validator(mode="after")
    def normalize_lists(self) -> "RankedMarkerGene":
        self.supported_cell_types = _clean_strings(self.supported_cell_types)
        self.evidence_sources = _clean_strings(self.evidence_sources)

        return self


class CandidateCellType(StrictModel):
    """
    Cluster-level candidate built only from direct marker evidence.

    HPA fields are supplemental summaries and do not independently
    create candidate votes.
    """

    candidate_rank: int = Field(ge=1)
    cell_type: str
    unique_supporting_gene_count: int = Field(ge=1)
    supporting_genes: list[str]

    both_source_supporting_gene_count: int = Field(default=0, ge=0)
    both_source_supporting_genes: list[str] = Field(default_factory=list)

    specificity_weighted_support: float = Field(default=0.0, ge=0.0)
    mean_supporting_gene_rank: float = Field(ge=1.0)
    mean_supporting_log_fold_change: float | None = None

    evidence_count: int = Field(default=0, ge=0)
    evidence_sources: list[str] = Field(default_factory=list)

    tissue_contexts: list[str] = Field(default_factory=list)
    disease_contexts: list[str] = Field(default_factory=list)
    cell_ontology_ids: list[str] = Field(default_factory=list)

    hpa_queried_gene_count: int = Field(default=0, ge=0)
    hpa_breast_expression_gene_count: int = Field(default=0, ge=0)
    hpa_top_cell_types: list[str] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_candidate(self) -> "CandidateCellType":
        self.supporting_genes = sorted(
            {_normalize_gene(value) for value in self.supporting_genes}
        )
        self.both_source_supporting_genes = sorted(
            {_normalize_gene(value) for value in self.both_source_supporting_genes}
        )
        self.evidence_sources = _clean_strings(self.evidence_sources)
        self.tissue_contexts = _clean_strings(self.tissue_contexts)
        self.disease_contexts = _clean_strings(self.disease_contexts)
        self.cell_ontology_ids = _clean_strings(self.cell_ontology_ids)
        self.hpa_top_cell_types = _clean_strings(self.hpa_top_cell_types)
        self.warnings = _clean_strings(self.warnings)

        if self.unique_supporting_gene_count != len(self.supporting_genes):
            raise ValueError(
                "unique_supporting_gene_count must equal supporting_genes length."
            )

        if self.both_source_supporting_gene_count != len(
            self.both_source_supporting_genes
        ):
            raise ValueError(
                "both_source_supporting_gene_count must equal "
                "both_source_supporting_genes length."
            )

        if not set(self.both_source_supporting_genes).issubset(
            self.supporting_genes
        ):
            raise ValueError(
                "both_source_supporting_genes must be a subset "
                "of supporting_genes."
            )

        if self.hpa_breast_expression_gene_count > self.hpa_queried_gene_count:
            raise ValueError(
                "HPA breast-expression count cannot exceed HPA queried count."
            )

        return self


class CandidateSelection(StrictModel):
    """Metadata describing the dynamic candidate cutoff."""

    selected_candidate_count: int = Field(ge=0)
    selected_candidate_cell_types: list[str] = Field(default_factory=list)
    cutoff_index: int | None = Field(default=None, ge=1)
    cutoff_method: str
    support_drop_absolute: int | None = Field(default=None, ge=0)
    support_drop_relative: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_selection(self) -> "CandidateSelection":
        self.selected_candidate_cell_types = _clean_strings_preserve_order(
            self.selected_candidate_cell_types
        )

        if self.selected_candidate_count != len(
            self.selected_candidate_cell_types
        ):
            raise ValueError(
                "selected_candidate_count must equal selected candidate labels."
            )

        return self


class ReviewFlag(StrictModel):
    """Human-review or informational flag produced by Agent 2."""

    code: str
    severity: ReviewSeverity
    message: str
    genes: list[str] = Field(default_factory=list)
    cell_types: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_lists(self) -> "ReviewFlag":
        self.genes = sorted({_normalize_gene(value) for value in self.genes})
        self.cell_types = _clean_strings(self.cell_types)

        return self


class FollowUpGene(StrictModel):
    """Gene routed to a later review or literature step."""

    gene: str
    reason: str
    recommended_action: FollowUpAction
    evidence_category: EvidenceCategory
    source_coverage: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("gene")
    @classmethod
    def normalize_gene(cls, value: str) -> str:
        return _normalize_gene(value)

    @model_validator(mode="after")
    def normalize_lists(self) -> "FollowUpGene":
        self.source_coverage = _clean_strings(self.source_coverage)
        self.warnings = _clean_strings(self.warnings)

        return self


class Agent2Summary(StrictModel):
    """Compact cluster-level summary."""

    total_input_genes: int = Field(ge=0)
    direct_marker_supported_genes: int = Field(ge=0)
    hpa_queried_genes: int = Field(ge=0)
    hpa_gene_matches: int = Field(ge=0)
    hpa_breast_expression_genes: int = Field(ge=0)
    ranked_marker_gene_count: int = Field(ge=0)
    ranked_candidate_count: int = Field(ge=0)
    selected_candidate_count: int = Field(ge=0)
    follow_up_gene_count: int = Field(ge=0)
    leading_candidate: str | None = None
    human_review_required: bool = False


class MarkerEvidence(StrictModel):
    """Complete Marker Interpretation Agent V2 output."""

    schema_name: str = "MarkerEvidence"
    schema_version: str = "2.0"

    cluster_id: str
    species: str
    tissue: str
    platform: str

    input_genes: list[InputGene]
    gene_interpretations: list[GeneInterpretation]
    ranked_marker_genes: list[RankedMarkerGene]
    ranked_candidate_cell_types: list[CandidateCellType]
    selected_candidate_cell_types: list[CandidateCellType]
    candidate_selection: CandidateSelection
    follow_up_gene_queue: list[FollowUpGene]
    review_flags: list[ReviewFlag]
    summary: Agent2Summary
    explanation: str
    config: dict[str, Any]
    provenance: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_result(self) -> "MarkerEvidence":
        input_names = [item.gene for item in self.input_genes]
        interpretation_names = [item.gene for item in self.gene_interpretations]

        if len(input_names) != len(set(input_names)):
            raise ValueError("MarkerEvidence contains duplicate input genes.")

        if len(interpretation_names) != len(set(interpretation_names)):
            raise ValueError(
                "MarkerEvidence contains duplicate gene interpretations."
            )

        if input_names != interpretation_names:
            raise ValueError(
                "Input genes and gene interpretations must use the same order."
            )

        ranked_candidates = {
            item.cell_type: item for item in self.ranked_candidate_cell_types
        }

        for selected in self.selected_candidate_cell_types:
            if selected.cell_type not in ranked_candidates:
                raise ValueError(
                    "Every selected candidate must exist in the ranked candidates."
                )

        direct_marker_genes = {
            item.gene
            for item in self.gene_interpretations
            if item.direct_marker_supported
        }

        for candidate in self.ranked_candidate_cell_types:
            unsupported_genes = set(candidate.supporting_genes) - direct_marker_genes

            if unsupported_genes:
                genes = ", ".join(sorted(unsupported_genes))

                raise ValueError(
                    "Candidate cell types may only use direct-marker genes. "
                    f"Unsupported genes: {genes}"
                )

        self.warnings = _clean_strings(self.warnings)

        return self

