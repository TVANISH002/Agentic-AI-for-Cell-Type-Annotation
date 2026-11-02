from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ContextMatchLevel = Literal["tissue_and_disease", "tissue", "disease", "none"]
RetrievalMethod = Literal["official_symbol", "alias"]


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DEGStatistics(StrictSchema):
    rank_within_cluster: int = Field(gt=0)
    score: float
    log_fold_change: float
    p_value: float = Field(ge=0.0, le=1.0)
    adjusted_p_value: float = Field(ge=0.0, le=1.0)
    percent_expressed_group: float = Field(ge=0.0, le=1.0)
    percent_expressed_reference: float = Field(ge=0.0, le=1.0)
    validation_warnings: list[str] = Field(default_factory=list)

    @field_validator("score", "log_fold_change")
    @classmethod
    def validate_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("DEG statistics must be finite numbers.")
        return value


class InputGene(StrictSchema):
    gene: str
    statistics: DEGStatistics


class SourceEvidence(StrictSchema):
    source: str
    retrieval_method: RetrievalMethod
    reported_marker: str | None = None
    official_symbol: str | None = None
    cell_type: str | None = None
    normalized_cell_type: str | None = None
    tissue: str | None = None
    disease_context: str | None = None
    context_match_level: ContextMatchLevel = "none"
    cell_ontology_id: str | None = None
    description: str | None = None
    pmid: str | None = None
    evidence_title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AtlasEvidence(StrictSchema):
    source: str
    cell_type: str | None = None
    tissue: str | None = None
    cell_ontology_id: str | None = None
    description: str | None = None
    reference_id: str | None = None


class GeneEvidence(StrictSchema):
    gene: str
    official_symbol: str | None = None

    aliases: list[str] = Field(default_factory=list)
    human_only_aliases: list[str] = Field(default_factory=list)
    mixed_species_aliases: list[str] = Field(default_factory=list)

    known_marker_cell_types: list[str] = Field(default_factory=list)
    raw_marker_cell_types: list[str] = Field(default_factory=list)
    context_matched_cell_types: list[str] = Field(default_factory=list)

    tissue_specificity: list[str] = Field(default_factory=list)
    context_matched_tissues: list[str] = Field(default_factory=list)

    disease_contexts: list[str] = Field(default_factory=list)
    context_matched_disease_contexts: list[str] = Field(default_factory=list)

    cell_ontology_ids: list[str] = Field(default_factory=list)
    context_matched_cell_ontology_ids: list[str] = Field(default_factory=list)

    descriptions: list[str] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)

    database_evidence: list[SourceEvidence] = Field(default_factory=list)
    context_matched_database_evidence: list[SourceEvidence] = Field(default_factory=list)
    atlas_evidence: list[AtlasEvidence] = Field(default_factory=list)

    quality_warnings: list[str] = Field(default_factory=list)


class SupportingCellTypeEvidence(StrictSchema):
    cell_type: str
    raw_cell_type_labels: list[str] = Field(default_factory=list)
    supporting_genes: list[str] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)
    tissue_contexts: list[str] = Field(default_factory=list)
    disease_contexts: list[str] = Field(default_factory=list)
    cell_ontology_ids: list[str] = Field(default_factory=list)
    evidence_count: int = Field(ge=0)


class SourceCoverage(StrictSchema):
    total_input_genes: int = Field(ge=0)
    matched_official_symbol: int = Field(ge=0)
    recovered_through_alias: int = Field(ge=0)
    final_matched_genes: int = Field(ge=0)
    unresolved_genes: int = Field(ge=0)
    coverage_percent: float = Field(ge=0.0, le=100.0)
    matched_cellmarker: int = Field(ge=0)
    matched_panglaodb: int = Field(ge=0)
    matched_both: int = Field(ge=0)


class DEGValidationSummary(StrictSchema):
    total_cluster_rows: int = Field(ge=0)
    rows_passing_thresholds: int = Field(ge=0)
    rows_filtered_out: int = Field(ge=0)
    rows_selected: int = Field(ge=0)
    duplicate_genes_removed: int = Field(ge=0)
    min_log_fold_change: float
    max_adjusted_p_value: float = Field(ge=0.0, le=1.0)
    warning_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class ContextDefinition(StrictSchema):
    requested_tissue: str
    tissue_terms: list[str] = Field(default_factory=list)
    disease_terms: list[str] = Field(default_factory=list)


class KnowledgeEvidence(StrictSchema):
    cluster_id: int | str
    species: str
    tissue: str
    platform: str

    context_definition: ContextDefinition
    input_genes: list[InputGene] = Field(default_factory=list)
    gene_evidence: list[GeneEvidence] = Field(default_factory=list)

    supporting_cell_types: list[SupportingCellTypeEvidence] = Field(default_factory=list)
    global_supporting_cell_types: list[SupportingCellTypeEvidence] = Field(default_factory=list)

    source_coverage: SourceCoverage
    deg_validation: DEGValidationSummary

    unresolved_genes: list[str] = Field(default_factory=list)
    data_quality_warnings: list[str] = Field(default_factory=list)
