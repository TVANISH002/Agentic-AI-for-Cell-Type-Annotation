from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


QueryPurpose = Literal[
    "context_overview",
    "candidate_support",
    "candidate_context",
    "cluster_marker_context",
    "candidate_comparison",
    "platform_context",
]

RelevanceLabel = Literal[
    "high",
    "medium",
    "low",
]

PassageSource = Literal[
    "abstract",
]

ReviewSeverity = Literal[
    "info",
    "warning",
    "error",
]


def _clean_strings_preserve_order(
    values: list[str] | None,
) -> list[str]:
    if not values:
        return []

    cleaned: list[str] = []
    seen: set[str] = set()

    for value in values:
        item = " ".join(str(value).split())
        key = item.casefold()

        if item and key not in seen:
            seen.add(key)
            cleaned.append(item)

    return cleaned


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class MarkerGene(StrictModel):
    gene: str
    rank_within_cluster: int = Field(ge=1)
    score: float | None = None
    log_fold_change: float | None = None
    p_value: float | None = Field(
        default=None,
        ge=0.0,
    )
    adjusted_p_value: float | None = Field(
        default=None,
        ge=0.0,
    )
    pct_nz_group: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    pct_nz_reference: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    @field_validator("gene")
    @classmethod
    def normalize_gene(
        cls,
        value: str,
    ) -> str:
        gene = str(value).strip().upper()

        if not gene:
            raise ValueError(
                "gene cannot be empty."
            )

        return gene


class CandidateAnnotation(StrictModel):
    name: str
    upstream_rank: int | None = Field(
        default=None,
        ge=1,
    )
    supporting_genes: list[str] = Field(
        default_factory=list,
    )
    evidence_sources: list[str] = Field(
        default_factory=list,
    )

    @field_validator("name")
    @classmethod
    def validate_name(
        cls,
        value: str,
    ) -> str:
        name = " ".join(str(value).split())

        if not name:
            raise ValueError(
                "candidate name cannot be empty."
            )

        return name

    @field_validator("supporting_genes")
    @classmethod
    def normalize_genes(
        cls,
        values: list[str],
    ) -> list[str]:
        cleaned = _clean_strings_preserve_order(
            values
        )

        return [
            gene.upper()
            for gene in cleaned
        ]

    @field_validator("evidence_sources")
    @classmethod
    def clean_sources(
        cls,
        values: list[str],
    ) -> list[str]:
        return _clean_strings_preserve_order(
            values
        )


class ReviewFlag(StrictModel):
    code: str
    severity: ReviewSeverity = "info"
    message: str
    genes: list[str] = Field(
        default_factory=list,
    )
    cell_types: list[str] = Field(
        default_factory=list,
    )


class LiteratureRequest(StrictModel):
    schema_name: str = "LiteratureRequest"
    schema_version: str = "1.0"

    cluster_id: str
    species: str
    tissue: str
    disease_context: str
    platform: str

    marker_genes: list[MarkerGene]
    candidate_annotations: list[
        CandidateAnnotation
    ]

    review_flags: list[ReviewFlag] = Field(
        default_factory=list,
    )
    spatial_context: str | None = None
    question: str

    upstream_schema_name: str
    upstream_schema_version: str
    source_agent2_file: str


class LiteratureQuery(StrictModel):
    query_id: str
    purpose: QueryPurpose
    query: str

    candidate_annotations: list[str] = Field(
        default_factory=list,
    )
    marker_genes: list[str] = Field(
        default_factory=list,
    )


class LiteratureQueryPlan(StrictModel):
    schema_name: str = "LiteratureQueryPlan"
    schema_version: str = "1.0"

    cluster_id: str
    queries: list[LiteratureQuery]


class PublicationRecord(StrictModel):
    record_key: str
    source: str
    source_id: str

    pmid: str | None = None
    pmcid: str | None = None
    doi: str | None = None

    title: str
    authors: list[str] = Field(
        default_factory=list,
    )
    journal: str | None = None
    publication_year: int | None = Field(
        default=None,
        ge=1600,
        le=2200,
    )
    first_publication_date: str | None = None
    publication_types: list[str] = Field(
        default_factory=list,
    )

    abstract: str | None = None
    is_open_access: bool = False
    is_preprint: bool = False
    cited_by_count: int = Field(
        default=0,
        ge=0,
    )


class QueryRetrievalResult(StrictModel):
    query: LiteratureQuery
    executed_query: str

    hit_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    next_cursor_mark: str | None = None

    papers: list[PublicationRecord] = Field(
        default_factory=list,
    )
    warnings: list[str] = Field(
        default_factory=list,
    )


class RetrievalPilotResult(StrictModel):
    schema_name: str = (
        "LiteratureRetrievalPilot"
    )
    schema_version: str = "1.0"

    request: LiteratureRequest
    query_plan: LiteratureQueryPlan
    query_results: list[
        QueryRetrievalResult
    ]
    unique_papers: list[PublicationRecord]

    warnings: list[str] = Field(
        default_factory=list,
    )


class PaperRelevanceBreakdown(StrictModel):
    species_match: bool = False
    tissue_match: bool = False
    disease_match: bool = False
    platform_match: bool = False
    single_cell_or_spatial_match: bool = False

    primary_study: bool = False
    review_article: bool = False
    abstract_available: bool = False

    matched_candidates: list[str] = Field(
        default_factory=list,
    )
    matched_markers: list[str] = Field(
        default_factory=list,
    )
    matched_query_ids: list[str] = Field(
        default_factory=list,
    )
    matched_query_purposes: list[
        QueryPurpose
    ] = Field(
        default_factory=list,
    )

    score_components: dict[
        str,
        float,
    ] = Field(
        default_factory=dict,
    )


class RankedPaper(StrictModel):
    rank: int = Field(ge=1)
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
    )
    relevance_label: RelevanceLabel

    publication: PublicationRecord
    breakdown: PaperRelevanceBreakdown
    rationale: list[str] = Field(
        default_factory=list,
    )


class PaperRankingResult(StrictModel):
    schema_name: str = "PaperRankingResult"
    schema_version: str = "1.0"

    cluster_id: str
    total_papers: int = Field(ge=0)
    high_relevance_count: int = Field(ge=0)
    medium_relevance_count: int = Field(ge=0)
    low_relevance_count: int = Field(ge=0)

    ranked_papers: list[RankedPaper] = Field(
        default_factory=list,
    )
    ranking_config: dict[
        str,
        float | int | str,
    ] = Field(
        default_factory=dict,
    )
    warnings: list[str] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_counts(
        self,
    ) -> "PaperRankingResult":
        if self.total_papers != len(
            self.ranked_papers
        ):
            raise ValueError(
                "total_papers must equal "
                "len(ranked_papers)."
            )

        relevance_total = (
            self.high_relevance_count
            + self.medium_relevance_count
            + self.low_relevance_count
        )

        if relevance_total != self.total_papers:
            raise ValueError(
                "Relevance counts must sum "
                "to total_papers."
            )

        expected_ranks = list(
            range(
                1,
                self.total_papers + 1,
            )
        )

        observed_ranks = [
            paper.rank
            for paper in self.ranked_papers
        ]

        if observed_ranks != expected_ranks:
            raise ValueError(
                "ranked_papers must contain "
                "sequential ranks."
            )

        return self


class EvidencePassage(StrictModel):
    passage_id: str
    record_key: str

    source_rank: int = Field(ge=1)
    paper_relevance_score: float = Field(
        ge=0.0,
        le=1.0,
    )
    paper_relevance_label: RelevanceLabel

    passage_source: PassageSource = "abstract"
    section: str = "abstract"
    passage_index: int = Field(ge=1)
    text: str

    publication: PublicationRecord

    candidate_mentions: list[str] = Field(
        default_factory=list,
    )
    marker_mentions: list[str] = Field(
        default_factory=list,
    )
    context_matches: list[str] = Field(
        default_factory=list,
    )

    retrieval_score: float = Field(
        ge=0.0,
        le=1.0,
    )
    candidate_scores: dict[
        str,
        float,
    ] = Field(
        default_factory=dict,
    )


class CandidatePassageSet(StrictModel):
    candidate_cell_type: str
    passage_ids: list[str] = Field(
        default_factory=list,
    )
    available_passage_count: int = Field(
        ge=0,
    )


class EvidenceCorpusResult(StrictModel):
    schema_name: str = (
        "LiteratureEvidenceCorpus"
    )
    schema_version: str = "1.0"

    cluster_id: str
    literature_request: LiteratureRequest

    selected_paper_count: int = Field(
        ge=0,
    )
    excluded_no_abstract_count: int = Field(
        ge=0,
    )
    passage_count: int = Field(
        ge=0,
    )

    passages: list[EvidencePassage] = Field(
        default_factory=list,
    )
    candidate_passage_sets: list[
        CandidatePassageSet
    ] = Field(
        default_factory=list,
    )

    corpus_config: dict[
        str,
        float | int | str,
    ] = Field(
        default_factory=dict,
    )

    source_pilot_file: str
    source_ranking_file: str

    warnings: list[str] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_corpus(
        self,
    ) -> "EvidenceCorpusResult":
        if self.passage_count != len(
            self.passages
        ):
            raise ValueError(
                "passage_count must equal "
                "len(passages)."
            )

        passage_ids = [
            passage.passage_id
            for passage in self.passages
        ]

        if len(passage_ids) != len(
            set(passage_ids)
        ):
            raise ValueError(
                "Evidence passage IDs "
                "must be unique."
            )

        known_ids = set(passage_ids)

        for candidate_set in (
            self.candidate_passage_sets
        ):
            unknown_ids = (
                set(candidate_set.passage_ids)
                - known_ids
            )

            if unknown_ids:
                raise ValueError(
                    "Candidate passage sets "
                    "contain unknown passage IDs."
                )

            if (
                candidate_set.available_passage_count
                < len(candidate_set.passage_ids)
            ):
                raise ValueError(
                    "available_passage_count cannot "
                    "be smaller than the number of "
                    "selected passage IDs."
                )

        return self


class CandidateExtractionJob(StrictModel):
    job_id: str
    cluster_id: str
    candidate_cell_type: str
    question: str

    observed_marker_genes: list[str] = Field(
        default_factory=list,
    )
    candidate_supporting_genes: list[str] = Field(
        default_factory=list,
    )
    passage_ids: list[str] = Field(
        default_factory=list,
    )

    prompt: str
    prompt_character_count: int = Field(
        ge=1,
    )


class ExtractionJobPlan(StrictModel):
    schema_name: str = (
        "LiteratureExtractionJobPlan"
    )
    schema_version: str = "1.0"

    cluster_id: str
    job_count: int = Field(ge=0)
    jobs: list[CandidateExtractionJob] = Field(
        default_factory=list,
    )

    source_corpus_file: str
    warnings: list[str] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_jobs(
        self,
    ) -> "ExtractionJobPlan":
        if self.job_count != len(self.jobs):
            raise ValueError(
                "job_count must equal len(jobs)."
            )

        job_ids = [
            job.job_id
            for job in self.jobs
        ]

        if len(job_ids) != len(
            set(job_ids)
        ):
            raise ValueError(
                "Extraction job IDs must be unique."
            )

        return self