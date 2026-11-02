from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from datetime import date

from .schemas import (
    PaperRankingResult,
    PaperRelevanceBreakdown,
    PublicationRecord,
    QueryPurpose,
    RankedPaper,
    RelevanceLabel,
    RetrievalPilotResult,
)


@dataclass(frozen=True, slots=True)
class PaperRankingConfig:
    """
    Transparent and reusable settings for ranking retrieved papers.

    Biological inputs such as species, tissue, disease, marker genes,
    candidate annotations, and cluster ID are obtained dynamically from
    LiteratureRequest. They are not hardcoded here.
    """

    species_weight: float = 1.0
    tissue_weight: float = 1.5
    disease_weight: float = 2.0
    platform_weight: float = 1.0
    method_weight: float = 1.5
    candidate_weight: float = 1.5
    marker_weight: float = 2.0
    primary_study_weight: float = 0.75
    review_article_weight: float = 0.25
    multi_query_weight: float = 1.0
    citation_weight: float = 0.5
    recent_publication_weight: float = 0.5

    preprint_penalty: float = 1.0
    missing_abstract_penalty: float = 1.0

    high_relevance_threshold: float = 0.60
    medium_relevance_threshold: float = 0.35
    recent_year_window: int = 7

    def __post_init__(self) -> None:
        non_negative_values = (
            self.species_weight,
            self.tissue_weight,
            self.disease_weight,
            self.platform_weight,
            self.method_weight,
            self.candidate_weight,
            self.marker_weight,
            self.primary_study_weight,
            self.review_article_weight,
            self.multi_query_weight,
            self.citation_weight,
            self.recent_publication_weight,
            self.preprint_penalty,
            self.missing_abstract_penalty,
        )

        if any(
            value < 0
            for value in non_negative_values
        ):
            raise ValueError(
                "Paper-ranking weights and penalties "
                "cannot be negative."
            )

        if not (
            0
            <= self.medium_relevance_threshold
            <= 1
        ):
            raise ValueError(
                "medium_relevance_threshold must be "
                "between 0 and 1."
            )

        if not (
            0
            <= self.high_relevance_threshold
            <= 1
        ):
            raise ValueError(
                "high_relevance_threshold must be "
                "between 0 and 1."
            )

        if (
            self.medium_relevance_threshold
            >= self.high_relevance_threshold
        ):
            raise ValueError(
                "medium_relevance_threshold must be "
                "lower than high_relevance_threshold."
            )

        if self.recent_year_window <= 0:
            raise ValueError(
                "recent_year_window must be "
                "greater than zero."
            )

    @property
    def maximum_positive_score(self) -> float:
        return (
            self.species_weight
            + self.tissue_weight
            + self.disease_weight
            + self.platform_weight
            + self.method_weight
            + self.candidate_weight
            + self.marker_weight
            + self.primary_study_weight
            + self.review_article_weight
            + self.multi_query_weight
            + self.citation_weight
            + self.recent_publication_weight
        )

    def to_dict(
        self,
    ) -> dict[str, float | int | str]:
        return {
            **asdict(self),
            "maximum_positive_score": (
                self.maximum_positive_score
            ),
        }


DEFAULT_RANKING_CONFIG = PaperRankingConfig()


def _normalized_text(value: str) -> str:
    text = value.casefold()
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"[^a-z0-9+]+", " ", text)

    return " ".join(text.split())


def _contains_phrase(
    text: str,
    phrase: str,
) -> bool:
    normalized_phrase = _normalized_text(phrase)

    if not normalized_phrase:
        return False

    return (
        normalized_phrase in text
        or f"{normalized_phrase}s" in text
    )


def _species_terms(
    species: str,
) -> list[str]:
    normalized_species = _normalized_text(species)

    if normalized_species in {
        "human",
        "humans",
        "homo sapiens",
    }:
        return [
            "human",
            "homo sapiens",
            "patient",
            "patients",
        ]

    if normalized_species in {
        "mouse",
        "mice",
        "mus musculus",
    }:
        return [
            "mouse",
            "mice",
            "murine",
            "mus musculus",
        ]

    return [species]


def _is_review(
    publication: PublicationRecord,
) -> bool:
    publication_types = " ".join(
        publication.publication_types
    ).casefold()

    title = publication.title.casefold()

    return (
        "review" in publication_types
        or "review" in title
    )


def _is_primary_study(
    publication: PublicationRecord,
) -> bool:
    publication_types = " ".join(
        publication.publication_types
    ).casefold()

    return (
        not _is_review(publication)
        and (
            "journal article" in publication_types
            or "research article" in publication_types
            or bool(publication.abstract)
        )
    )


class PaperRanker:
    """
    Rank retrieved papers using metadata and abstract relevance.

    This class ranks retrieval relevance only. It does not treat paper
    counts as biological votes and does not assign a final cell type.
    """

    METHOD_TERMS = (
        "single cell",
        "single-cell",
        "scrna seq",
        "scrna-seq",
        "snrna seq",
        "snrna-seq",
        "spatial transcriptomics",
        "spatial transcriptomic",
        "spatially resolved",
    )

    def __init__(
        self,
        config: PaperRankingConfig = (
            DEFAULT_RANKING_CONFIG
        ),
    ) -> None:
        self.config = config

    def rank(
        self,
        retrieval: RetrievalPilotResult,
    ) -> PaperRankingResult:
        request = retrieval.request

        query_ids_by_paper: dict[
            str,
            list[str],
        ] = {}

        query_purposes_by_paper: dict[
            str,
            list[QueryPurpose],
        ] = {}

        for query_result in retrieval.query_results:
            for paper in query_result.papers:
                query_ids_by_paper.setdefault(
                    paper.record_key,
                    [],
                ).append(
                    query_result.query.query_id
                )

                query_purposes_by_paper.setdefault(
                    paper.record_key,
                    [],
                ).append(
                    query_result.query.purpose
                )

        assessed_papers: list[
            tuple[
                float,
                PublicationRecord,
                PaperRelevanceBreakdown,
                list[str],
            ]
        ] = []

        for publication in retrieval.unique_papers:
            complete_text = " ".join(
                part
                for part in (
                    publication.title,
                    publication.abstract or "",
                )
                if part
            )

            normalized_text = _normalized_text(
                complete_text
            )

            species_match = any(
                _contains_phrase(
                    normalized_text,
                    term,
                )
                for term in _species_terms(
                    request.species
                )
            )

            tissue_match = _contains_phrase(
                normalized_text,
                request.tissue,
            )

            disease_match = _contains_phrase(
                normalized_text,
                request.disease_context,
            )

            platform_match = _contains_phrase(
                normalized_text,
                request.platform,
            )

            method_match = any(
                _contains_phrase(
                    normalized_text,
                    term,
                )
                for term in self.METHOD_TERMS
            )

            matched_candidates = [
                candidate.name
                for candidate
                in request.candidate_annotations
                if _contains_phrase(
                    normalized_text,
                    candidate.name,
                )
            ]

            matched_markers = [
                marker.gene
                for marker in request.marker_genes
                if re.search(
                    (
                        rf"(?<![A-Za-z0-9])"
                        rf"{re.escape(marker.gene)}"
                        rf"(?![A-Za-z0-9])"
                    ),
                    complete_text,
                    flags=re.IGNORECASE,
                )
            ]

            query_ids = list(
                dict.fromkeys(
                    query_ids_by_paper.get(
                        publication.record_key,
                        [],
                    )
                )
            )

            query_purposes = list(
                dict.fromkeys(
                    query_purposes_by_paper.get(
                        publication.record_key,
                        [],
                    )
                )
            )

            primary_study = _is_primary_study(
                publication
            )

            review_article = _is_review(
                publication
            )

            score_components: dict[str, float] = {
                "species": (
                    self.config.species_weight
                    if species_match
                    else 0.0
                ),
                "tissue": (
                    self.config.tissue_weight
                    if tissue_match
                    else 0.0
                ),
                "disease": (
                    self.config.disease_weight
                    if disease_match
                    else 0.0
                ),
                "platform": (
                    self.config.platform_weight
                    if platform_match
                    else 0.0
                ),
                "single_cell_or_spatial": (
                    self.config.method_weight
                    if method_match
                    else 0.0
                ),
                "candidate": (
                    self.config.candidate_weight
                    * min(
                        1.0,
                        len(matched_candidates) / 2.0,
                    )
                ),
                "markers": (
                    self.config.marker_weight
                    * min(
                        1.0,
                        len(matched_markers) / 4.0,
                    )
                ),
                "primary_study": (
                    self.config.primary_study_weight
                    if primary_study
                    else 0.0
                ),
                "review_article": (
                    self.config.review_article_weight
                    if review_article
                    else 0.0
                ),
                "multiple_query_families": (
                    self.config.multi_query_weight
                    * min(
                        1.0,
                        max(
                            0,
                            len(query_purposes) - 1,
                        )
                        / 2.0,
                    )
                ),
                "citations": (
                    self.config.citation_weight
                    * min(
                        1.0,
                        math.log1p(
                            publication.cited_by_count
                        )
                        / math.log(51),
                    )
                ),
                "recent_publication": (
                    self.config.recent_publication_weight
                    if (
                        publication.publication_year
                        is not None
                        and publication.publication_year
                        >= (
                            date.today().year
                            - self.config.recent_year_window
                        )
                    )
                    else 0.0
                ),
                "preprint_penalty": (
                    -self.config.preprint_penalty
                    if publication.is_preprint
                    else 0.0
                ),
                "missing_abstract_penalty": (
                    -self.config.missing_abstract_penalty
                    if not publication.abstract
                    else 0.0
                ),
            }

            raw_score = sum(
                score_components.values()
            )

            normalized_score = min(
                1.0,
                max(
                    0.0,
                    raw_score
                    / self.config.maximum_positive_score,
                ),
            )

            rationale: list[str] = []

            if species_match:
                rationale.append(
                    "Species context matched: "
                    f"{request.species}"
                )

            if tissue_match:
                rationale.append(
                    "Tissue context matched: "
                    f"{request.tissue}"
                )

            if disease_match:
                rationale.append(
                    "Disease context matched: "
                    f"{request.disease_context}"
                )

            if platform_match:
                rationale.append(
                    "Platform matched: "
                    f"{request.platform}"
                )

            if method_match:
                rationale.append(
                    "Single-cell or spatial "
                    "method mentioned"
                )

            if matched_candidates:
                rationale.append(
                    "Candidate mentions: "
                    + ", ".join(
                        matched_candidates
                    )
                )

            if matched_markers:
                rationale.append(
                    "Marker matches: "
                    + ", ".join(
                        matched_markers
                    )
                )

            if len(query_purposes) > 1:
                rationale.append(
                    "Retrieved by "
                    f"{len(query_purposes)} "
                    "query families"
                )

            if publication.is_preprint:
                rationale.append(
                    "Preprint penalty applied"
                )

            if not publication.abstract:
                rationale.append(
                    "Abstract unavailable; "
                    "penalty applied"
                )

            breakdown = PaperRelevanceBreakdown(
                species_match=species_match,
                tissue_match=tissue_match,
                disease_match=disease_match,
                platform_match=platform_match,
                single_cell_or_spatial_match=(
                    method_match
                ),
                primary_study=primary_study,
                review_article=review_article,
                abstract_available=bool(
                    publication.abstract
                ),
                matched_candidates=(
                    matched_candidates
                ),
                matched_markers=matched_markers,
                matched_query_ids=query_ids,
                matched_query_purposes=(
                    query_purposes
                ),
                score_components=(
                    score_components
                ),
            )

            assessed_papers.append(
                (
                    normalized_score,
                    publication,
                    breakdown,
                    rationale,
                )
            )

        assessed_papers.sort(
            key=lambda item: (
                -item[0],
                -(
                    item[1].publication_year
                    or 0
                ),
                item[1].title.casefold(),
            )
        )

        ranked_papers: list[RankedPaper] = []

        for rank, assessed in enumerate(
            assessed_papers,
            start=1,
        ):
            (
                score,
                publication,
                breakdown,
                rationale,
            ) = assessed

            relevance_label: RelevanceLabel

            if (
                score
                >= self.config.high_relevance_threshold
            ):
                relevance_label = "high"

            elif (
                score
                >= self.config.medium_relevance_threshold
            ):
                relevance_label = "medium"

            else:
                relevance_label = "low"

            ranked_papers.append(
                RankedPaper(
                    rank=rank,
                    relevance_score=round(
                        score,
                        6,
                    ),
                    relevance_label=(
                        relevance_label
                    ),
                    publication=publication,
                    breakdown=breakdown,
                    rationale=rationale,
                )
            )

        high_count = sum(
            paper.relevance_label == "high"
            for paper in ranked_papers
        )

        medium_count = sum(
            paper.relevance_label == "medium"
            for paper in ranked_papers
        )

        low_count = sum(
            paper.relevance_label == "low"
            for paper in ranked_papers
        )

        return PaperRankingResult(
            cluster_id=request.cluster_id,
            total_papers=len(ranked_papers),
            high_relevance_count=high_count,
            medium_relevance_count=medium_count,
            low_relevance_count=low_count,
            ranked_papers=ranked_papers,
            ranking_config=(
                self.config.to_dict()
            ),
            warnings=[],
        )