
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .schemas import (
    CandidateAnnotation,
    CandidatePassageSet,
    EvidenceCorpusResult,
    EvidencePassage,
    LiteratureRequest,
    PaperRankingResult,
    RelevanceLabel,
)


@dataclass(frozen=True, slots=True)
class EvidenceCorpusConfig:
    """
    Technical settings for constructing the RAG corpus.

    The cluster, tissue, disease, genes and candidate cell
    types are obtained dynamically from LiteratureRequest.
    """

    min_paper_relevance: RelevanceLabel = "medium"
    maximum_papers: int = 30
    maximum_characters_per_passage: int = 1200
    overlap_sentences: int = 1
    maximum_passages_per_candidate: int = 12
    maximum_passages_per_paper_per_candidate: int = 1
    maximum_context_only_passages_per_candidate: int = 4

    def __post_init__(self) -> None:
        if self.maximum_papers <= 0:
            raise ValueError(
                "maximum_papers must be greater than zero."
            )

        if (
            self.maximum_characters_per_passage
            < 300
        ):
            raise ValueError(
                "maximum_characters_per_passage "
                "must be at least 300."
            )

        if self.overlap_sentences < 0:
            raise ValueError(
                "overlap_sentences cannot be negative."
            )

        if (
            self.maximum_passages_per_candidate
            <= 0
        ):
            raise ValueError(
                "maximum_passages_per_candidate "
                "must be greater than zero."
            )

        if (
            self.maximum_passages_per_paper_per_candidate
            <= 0
        ):
            raise ValueError(
                "maximum_passages_per_paper_per_candidate "
                "must be greater than zero."
            )

        if (
            self.maximum_context_only_passages_per_candidate
            < 0
        ):
            raise ValueError(
                "maximum_context_only_passages_per_candidate "
                "cannot be negative."
            )

    def to_dict(
        self,
    ) -> dict[str, float | int | str]:
        return asdict(self)


DEFAULT_CORPUS_CONFIG = EvidenceCorpusConfig()


_RELEVANCE_ORDER: dict[
    RelevanceLabel,
    int,
] = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


_METHOD_TERMS = (
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


def _normalize(
    value: str,
) -> str:
    text = value.casefold()
    text = re.sub(
        r"[-_/]+",
        " ",
        text,
    )
    text = re.sub(
        r"[^a-z0-9+]+",
        " ",
        text,
    )

    return " ".join(text.split())


def _contains_phrase(
    text: str,
    phrase: str,
) -> bool:
    normalized_phrase = _normalize(phrase)

    return bool(
        normalized_phrase
        and (
            normalized_phrase in text
            or f"{normalized_phrase}s" in text
        )
    )


def _contains_gene(
    text: str,
    gene: str,
) -> bool:
    return bool(
        re.search(
            (
                rf"(?<![A-Za-z0-9])"
                rf"{re.escape(gene)}"
                rf"(?![A-Za-z0-9])"
            ),
            text,
            flags=re.IGNORECASE,
        )
    )


def _split_long_sentence(
    sentence: str,
    maximum: int,
) -> list[str]:
    words = sentence.split()

    pieces: list[str] = []
    current_words: list[str] = []
    current_length = 0

    for word in words:
        additional_length = (
            len(word)
            + (1 if current_words else 0)
        )

        if (
            current_words
            and (
                current_length
                + additional_length
                > maximum
            )
        ):
            pieces.append(
                " ".join(current_words)
            )
            current_words = [word]
            current_length = len(word)

        else:
            current_words.append(word)
            current_length += additional_length

    if current_words:
        pieces.append(
            " ".join(current_words)
        )

    return pieces


def _chunk_abstract(
    abstract: str,
    maximum: int,
    overlap_sentences: int,
) -> list[str]:
    """
    Split an abstract into sentence-aware overlapping passages.
    """

    cleaned_abstract = " ".join(
        abstract.split()
    )

    if not cleaned_abstract:
        return []

    raw_sentences = re.split(
        r"(?<=[.!?])\s+",
        cleaned_abstract,
    )

    sentences: list[str] = []

    for sentence in raw_sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) > maximum:
            sentences.extend(
                _split_long_sentence(
                    sentence=sentence,
                    maximum=maximum,
                )
            )

        else:
            sentences.append(sentence)

    chunks: list[str] = []
    current_sentences: list[str] = []

    for sentence in sentences:
        possible_chunk = " ".join(
            [
                *current_sentences,
                sentence,
            ]
        )

        if (
            current_sentences
            and len(possible_chunk) > maximum
        ):
            chunks.append(
                " ".join(current_sentences)
            )

            if overlap_sentences:
                overlap = current_sentences[
                    -overlap_sentences:
                ]
            else:
                overlap = []

            current_sentences = list(overlap)

            while (
                current_sentences
                and len(
                    " ".join(
                        [
                            *current_sentences,
                            sentence,
                        ]
                    )
                )
                > maximum
            ):
                current_sentences.pop(0)

        current_sentences.append(sentence)

    if current_sentences:
        chunks.append(
            " ".join(current_sentences)
        )

    return list(
        dict.fromkeys(
            chunk
            for chunk in chunks
            if chunk.strip()
        )
    )


def _context_matches(
    text: str,
    request: LiteratureRequest,
) -> list[str]:
    normalized_text = _normalize(text)
    matches: list[str] = []

    context_values = (
        (
            "species",
            request.species,
        ),
        (
            "tissue",
            request.tissue,
        ),
        (
            "disease",
            request.disease_context,
        ),
        (
            "platform",
            request.platform,
        ),
    )

    for label, value in context_values:
        if _contains_phrase(
            normalized_text,
            value,
        ):
            matches.append(label)

    method_match = any(
        _contains_phrase(
            normalized_text,
            term,
        )
        for term in _METHOD_TERMS
    )

    if method_match:
        matches.append(
            "single_cell_or_spatial"
        )

    return matches


def _candidate_score(
    paper_score: float,
    text: str,
    candidate: CandidateAnnotation,
    marker_mentions: list[str],
    context_matches: list[str],
) -> float:
    """
    Calculate passage relevance separately for each candidate.

    This does not classify the passage as supporting or
    contradicting the candidate. It only measures retrieval
    relevance for the later evidence-extraction stage.
    """

    normalized_text = _normalize(text)

    candidate_match = _contains_phrase(
        normalized_text,
        candidate.name,
    )

    candidate_supporting_genes = {
        gene.upper()
        for gene in candidate.supporting_genes
    }

    supporting_gene_matches = (
        candidate_supporting_genes.intersection(
            marker_mentions
        )
    )

    if candidate_supporting_genes:
        supporting_fraction = min(
            1.0,
            len(supporting_gene_matches) / 3.0,
        )
    else:
        supporting_fraction = 0.0

    marker_fraction = min(
        1.0,
        len(marker_mentions) / 4.0,
    )

    context_fraction = min(
        1.0,
        len(context_matches) / 4.0,
    )

    score = (
        0.40 * paper_score
        + 0.25 * float(candidate_match)
        + 0.15 * supporting_fraction
        + 0.10 * marker_fraction
        + 0.10 * context_fraction
    )

    return round(
        min(
            1.0,
            max(0.0, score),
        ),
        6,
    )


class EvidenceCorpusBuilder:
    """
    Construct citation-preserving abstract passages.

    Every passage retains the paper metadata needed to trace
    later biological claims back to PMID, PMCID or DOI.
    """

    def __init__(
        self,
        config: EvidenceCorpusConfig = (
            DEFAULT_CORPUS_CONFIG
        ),
    ) -> None:
        self.config = config

    def build(
        self,
        request: LiteratureRequest,
        ranking: PaperRankingResult,
        source_pilot_file: str,
        source_ranking_file: str,
    ) -> EvidenceCorpusResult:
        minimum_relevance = _RELEVANCE_ORDER[
            self.config.min_paper_relevance
        ]

        eligible_papers = [
            ranked_paper
            for ranked_paper
            in ranking.ranked_papers
            if (
                _RELEVANCE_ORDER[
                    ranked_paper.relevance_label
                ]
                >= minimum_relevance
            )
        ][
            : self.config.maximum_papers
        ]

        excluded_no_abstract_count = sum(
            not bool(
                ranked_paper.publication.abstract
            )
            for ranked_paper in eligible_papers
        )

        passages: list[EvidencePassage] = []

        for ranked_paper in eligible_papers:
            publication = (
                ranked_paper.publication
            )

            if not publication.abstract:
                continue

            chunks = _chunk_abstract(
                abstract=publication.abstract,
                maximum=(
                    self.config
                    .maximum_characters_per_passage
                ),
                overlap_sentences=(
                    self.config.overlap_sentences
                ),
            )

            for passage_index, chunk in enumerate(
                chunks,
                start=1,
            ):
                normalized_chunk = _normalize(chunk)

                candidate_mentions = [
                    candidate.name
                    for candidate
                    in request.candidate_annotations
                    if _contains_phrase(
                        normalized_chunk,
                        candidate.name,
                    )
                ]

                marker_mentions = [
                    marker.gene
                    for marker
                    in request.marker_genes
                    if _contains_gene(
                        chunk,
                        marker.gene,
                    )
                ]

                context_matches = _context_matches(
                    text=chunk,
                    request=request,
                )

                candidate_scores = {
                    candidate.name: _candidate_score(
                        paper_score=(
                            ranked_paper
                            .relevance_score
                        ),
                        text=chunk,
                        candidate=candidate,
                        marker_mentions=(
                            marker_mentions
                        ),
                        context_matches=(
                            context_matches
                        ),
                    )
                    for candidate
                    in request.candidate_annotations
                }

                overall_score = round(
                    min(
                        1.0,
                        (
                            0.50
                            * ranked_paper
                            .relevance_score
                            + 0.20
                            * min(
                                1.0,
                                (
                                    len(
                                        marker_mentions
                                    )
                                    / 4.0
                                ),
                            )
                            + 0.15
                            * min(
                                1.0,
                                (
                                    len(
                                        candidate_mentions
                                    )
                                    / 2.0
                                ),
                            )
                            + 0.15
                            * min(
                                1.0,
                                (
                                    len(
                                        context_matches
                                    )
                                    / 4.0
                                ),
                            )
                        ),
                    ),
                    6,
                )

                passage_id = (
                    f"{publication.record_key}"
                    f":abstract:{passage_index}"
                )

                passages.append(
                    EvidencePassage(
                        passage_id=passage_id,
                        record_key=(
                            publication.record_key
                        ),
                        source_rank=(
                            ranked_paper.rank
                        ),
                        paper_relevance_score=(
                            ranked_paper
                            .relevance_score
                        ),
                        paper_relevance_label=(
                            ranked_paper
                            .relevance_label
                        ),
                        passage_source="abstract",
                        section="abstract",
                        passage_index=passage_index,
                        text=chunk,
                        publication=publication,
                        candidate_mentions=(
                            candidate_mentions
                        ),
                        marker_mentions=(
                            marker_mentions
                        ),
                        context_matches=(
                            context_matches
                        ),
                        retrieval_score=(
                            overall_score
                        ),
                        candidate_scores=(
                            candidate_scores
                        ),
                    )
                )

        candidate_passage_sets: list[
            CandidatePassageSet
        ] = []

        for candidate in (
            request.candidate_annotations
        ):
            candidate_name = candidate.name

            candidate_supporting_genes = {
                gene.upper()
                for gene
                in candidate.supporting_genes
            }

            relevant_passages = [
                passage
                for passage in passages
                if (
                    candidate_name
                    in passage.candidate_mentions
                    or bool(
                        candidate_supporting_genes
                        .intersection(
                            passage.marker_mentions
                        )
                    )
                )
            ]

            def passage_priority(
                passage: EvidencePassage,
            ) -> tuple[int, float, int, int]:
                """
                Prefer direct candidate-and-marker passages.

                Context-only passages remain useful, but they
                must not crowd out discriminating marker evidence.
                """

                candidate_is_mentioned = (
                    candidate_name
                    in passage.candidate_mentions
                )

                supporting_gene_is_mentioned = bool(
                    candidate_supporting_genes.intersection(
                        passage.marker_mentions
                    )
                )

                any_cluster_marker_is_mentioned = bool(
                    passage.marker_mentions
                )

                if (
                    candidate_is_mentioned
                    and supporting_gene_is_mentioned
                ):
                    priority_bucket = 0
                elif (
                    candidate_is_mentioned
                    and any_cluster_marker_is_mentioned
                ):
                    priority_bucket = 1
                elif candidate_is_mentioned:
                    priority_bucket = 2
                else:
                    priority_bucket = 3

                return (
                    priority_bucket,
                    -passage.candidate_scores.get(
                        candidate_name,
                        0.0,
                    ),
                    passage.source_rank,
                    passage.passage_index,
                )

            relevant_passages.sort(
                key=passage_priority
            )

            selected_passages: list[
                EvidencePassage
            ] = []

            paper_passage_counts: dict[
                str,
                int,
            ] = {}

            context_only_count = 0

            for passage in relevant_passages:
                if (
                    len(selected_passages)
                    >= self.config
                    .maximum_passages_per_candidate
                ):
                    break

                paper_passage_count = (
                    paper_passage_counts.get(
                        passage.record_key,
                        0,
                    )
                )

                if (
                    paper_passage_count
                    >= self.config
                    .maximum_passages_per_paper_per_candidate
                ):
                    continue

                is_context_only = (
                    candidate_name
                    in passage.candidate_mentions
                    and not passage.marker_mentions
                )

                if (
                    is_context_only
                    and context_only_count
                    >= self.config
                    .maximum_context_only_passages_per_candidate
                ):
                    continue

                selected_passages.append(
                    passage
                )

                paper_passage_counts[
                    passage.record_key
                ] = paper_passage_count + 1

                if is_context_only:
                    context_only_count += 1

            candidate_passage_sets.append(
                CandidatePassageSet(
                    candidate_cell_type=(
                        candidate_name
                    ),
                    passage_ids=[
                        passage.passage_id
                        for passage
                        in selected_passages
                    ],
                    available_passage_count=len(
                        relevant_passages
                    ),
                )
            )

        warnings: list[str] = []

        if not passages:
            warnings.append(
                "No abstract passages were "
                "available for RAG."
            )

        for candidate_set in (
            candidate_passage_sets
        ):
            if not candidate_set.passage_ids:
                warnings.append(
                    "No candidate-specific passages "
                    "found for "
                    f"{candidate_set.candidate_cell_type}."
                )

        return EvidenceCorpusResult(
            cluster_id=request.cluster_id,
            literature_request=request,
            selected_paper_count=len(
                eligible_papers
            ),
            excluded_no_abstract_count=(
                excluded_no_abstract_count
            ),
            passage_count=len(passages),
            passages=passages,
            candidate_passage_sets=(
                candidate_passage_sets
            ),
            corpus_config=(
                self.config.to_dict()
            ),
            source_pilot_file=(
                source_pilot_file
            ),
            source_ranking_file=(
                source_ranking_file
            ),
            warnings=warnings,
        )