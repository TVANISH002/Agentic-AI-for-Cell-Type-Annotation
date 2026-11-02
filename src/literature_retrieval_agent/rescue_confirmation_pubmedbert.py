from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
)

from literature_retrieval_agent.evidence_corpus_builder import (
    _chunk_abstract,
)
from literature_retrieval_agent.runners.run_pubmedbert_reranker import (
    build_candidate_query,
    embed_texts,
)


def _clean(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


def _candidate_pattern(
    label: str,
):
    """
    Conservative canonical-label matching.

    Candidate labels are supplied dynamically.
    Ontology synonyms are intentionally excluded
    from this eligibility step.
    """

    label = _clean(label)

    words = label.split()

    if not words:
        return None

    escaped = [
        re.escape(word)
        for word in words
    ]

    separator = r"(?:\s+|-)"

    if (
        words[-1].casefold()
        == "cell"
    ):
        prefix = separator.join(
            escaped[:-1]
        )

        if prefix:
            phrase = (
                prefix
                + separator
                + r"cells?"
            )
        else:
            phrase = r"cells?"
    else:
        phrase = separator.join(
            escaped
        )

    return re.compile(
        rf"(?<![A-Za-z0-9])"
        rf"{phrase}"
        rf"(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )


def _gene_pattern(
    gene: str,
):
    return re.compile(
        rf"(?<![A-Za-z0-9])"
        rf"{re.escape(gene)}"
        rf"(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )


def _normalize_embeddings(
    matrix: np.ndarray,
) -> np.ndarray:

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    norms = np.clip(
        norms,
        1e-12,
        None,
    )

    return matrix / norms


def _publication_key(
    paper: Mapping[str, Any],
) -> str:

    return (
        _clean(
            paper.get(
                "record_key"
            )
        )
        or _clean(
            paper.get(
                "pmid"
            )
        )
        or _clean(
            paper.get(
                "doi"
            )
        )
        or _clean(
            paper.get(
                "title"
            )
        )
    )


class RescueConfirmationPubMedBERTReranker:

    def __init__(
        self,
        *,
        model_id: str = (
            "NeuML/"
            "pubmedbert-base-embeddings"
        ),
        top_marker_count: int = 25,
        top_k_passages: int = 10,
        maximum_characters_per_passage: int = 1200,
        overlap_sentences: int = 1,
        maximum_passages_per_paper: int = 1,
        batch_size: int = 16,
        max_length: int = 512,
    ):

        if top_marker_count < 1:
            raise ValueError(
                "top_marker_count must be >= 1."
            )

        if top_k_passages < 1:
            raise ValueError(
                "top_k_passages must be >= 1."
            )

        if (
            maximum_passages_per_paper
            < 1
        ):
            raise ValueError(
                "maximum_passages_per_paper "
                "must be >= 1."
            )

        self.model_id = model_id
        self.top_marker_count = (
            top_marker_count
        )
        self.top_k_passages = (
            top_k_passages
        )
        self.maximum_characters_per_passage = (
            maximum_characters_per_passage
        )
        self.overlap_sentences = (
            overlap_sentences
        )
        self.maximum_passages_per_paper = (
            maximum_passages_per_paper
        )
        self.batch_size = batch_size
        self.max_length = max_length

    def build_candidate_records(
        self,
        pilot: Mapping[str, Any],
        confirmation_plan: Mapping[
            str,
            Any,
        ],
        literature_request: Mapping[
            str,
            Any,
        ],
    ) -> tuple[
        str,
        list[dict],
        list[str],
    ]:

        pilot_request = (
            pilot.get(
                "request",
                {},
            )
            or {}
        )

        pilot_query_plan = (
            pilot.get(
                "query_plan",
                {},
            )
            or {}
        )

        pilot_cluster_id = str(
            pilot.get(
                "cluster_id"
            )
            or (
                pilot_request.get(
                    "cluster_id"
                )
                if isinstance(
                    pilot_request,
                    Mapping,
                )
                else ""
            )
            or (
                pilot_query_plan.get(
                    "cluster_id"
                )
                if isinstance(
                    pilot_query_plan,
                    Mapping,
                )
                else ""
            )
            or ""
        )

        plan_cluster_id = str(
            confirmation_plan.get(
                "cluster_id",
                "",
            )
        )

        request_cluster_id = str(
            literature_request.get(
                "cluster_id",
                "",
            )
        )

        cluster_ids = {
            value
            for value in (
                pilot_cluster_id,
                plan_cluster_id,
                request_cluster_id,
            )
            if value
        }

        if (
            len(cluster_ids) != 1
            or not pilot_cluster_id
            or not plan_cluster_id
            or not request_cluster_id
        ):
            raise ValueError(
                "Cluster mismatch across "
                "confirmation artifacts."
            )

        if confirmation_plan.get(
            "reference_label_used",
            False,
        ):
            raise ValueError(
                "Confirmation plan indicates "
                "reference-label use."
            )

        if confirmation_plan.get(
            "expert_label_used",
            False,
        ):
            raise ValueError(
                "Confirmation plan indicates "
                "expert-label use."
            )

        plan_lookup = {
            item["query_id"]:
                item
            for item
            in confirmation_plan.get(
                "queries",
                [],
            )
        }

        marker_items = (
            literature_request.get(
                "marker_genes",
                [],
            )
        )

        observed_markers = [
            _clean(
                item.get(
                    "gene",
                    "",
                )
            ).upper()
            for item
            in marker_items
            if (
                isinstance(
                    item,
                    Mapping,
                )
                and _clean(
                    item.get(
                        "gene",
                        "",
                    )
                )
            )
        ]

        top_markers = (
            observed_markers[
                :self.top_marker_count
            ]
        )

        marker_patterns = {
            marker:
                _gene_pattern(
                    marker
                )
            for marker
            in top_markers
        }

        candidate_records = []

        for query_result in (
            pilot.get(
                "query_results",
                [],
            )
        ):

            query_object = (
                query_result.get(
                    "query",
                    {},
                )
                or {}
            )

            query_id = _clean(
                query_object.get(
                    "query_id",
                    query_result.get(
                        "query_id",
                        "",
                    ),
                )
            )

            plan_item = (
                plan_lookup.get(
                    query_id
                )
            )

            if not plan_item:
                continue

            candidate_name = _clean(
                plan_item.get(
                    "candidate",
                    "",
                )
            )

            ontology_id = _clean(
                plan_item.get(
                    "ontology_id",
                    "",
                )
            )

            if not candidate_name:
                continue

            candidate_re = (
                _candidate_pattern(
                    candidate_name
                )
            )

            papers = (
                query_result.get(
                    "papers"
                )
                or query_result.get(
                    "publications"
                )
                or query_result.get(
                    "results"
                )
                or query_result.get(
                    "records"
                )
                or []
            )

            candidate_passages = []

            for paper_position, paper in enumerate(
                papers,
                start=1,
            ):

                title = _clean(
                    paper.get(
                        "title",
                        "",
                    )
                )

                abstract = _clean(
                    paper.get(
                        "abstract",
                        paper.get(
                            "abstract_text",
                            "",
                        ),
                    )
                )

                if not abstract:
                    continue

                record_key = (
                    _publication_key(
                        paper
                    )
                )

                if not record_key:
                    continue

                chunks = _chunk_abstract(
                    abstract=abstract,
                    maximum=(
                        self.maximum_characters_per_passage
                    ),
                    overlap_sentences=(
                        self.overlap_sentences
                    ),
                )

                for passage_index, chunk in enumerate(
                    chunks,
                    start=1,
                ):

                    chunk = _clean(
                        chunk
                    )

                    searchable = _clean(
                        f"{title}. {chunk}"
                    )

                    candidate_mentioned = bool(
                        candidate_re
                        and candidate_re.search(
                            searchable
                        )
                    )

                    if not candidate_mentioned:
                        continue

                    markers_mentioned = [
                        marker
                        for marker, pattern
                        in marker_patterns.items()
                        if pattern.search(
                            searchable
                        )
                    ]

                    candidate_passages.append(
                        {
                            "passage_id":
                                (
                                    f"{record_key}:"
                                    f"abstract:"
                                    f"{passage_index}"
                                ),

                            "record_key":
                                record_key,

                            "query_id":
                                query_id,

                            "candidate":
                                candidate_name,

                            "ontology_id":
                                ontology_id,

                            "retrieval_position":
                                paper_position,

                            "passage_index":
                                passage_index,

                            "title":
                                title,

                            "text":
                                chunk,

                            "pmid":
                                paper.get(
                                    "pmid"
                                ),

                            "pmcid":
                                paper.get(
                                    "pmcid"
                                ),

                            "doi":
                                paper.get(
                                    "doi"
                                ),

                            "journal":
                                paper.get(
                                    "journal"
                                ),

                            "publication_year":
                                paper.get(
                                    "publication_year"
                                ),

                            "markers_mentioned":
                                markers_mentioned,
                        }
                    )

            candidate_records.append(
                {
                    "query_id":
                        query_id,

                    "candidate":
                        candidate_name,

                    "ontology_id":
                        ontology_id,

                    "retrieved_paper_count":
                        len(
                            papers
                        ),

                    "passages":
                        candidate_passages,
                }
            )

        #
        # Preserve confirmation-plan ordering.
        #
        order = {
            item["query_id"]:
                index
            for index, item
            in enumerate(
                confirmation_plan.get(
                    "queries",
                    [],
                )
            )
        }

        candidate_records.sort(
            key=lambda item:
                order.get(
                    item["query_id"],
                    10**9,
                )
        )

        return (
            request_cluster_id,
            candidate_records,
            top_markers,
        )

    def rerank(
        self,
        pilot: Mapping[str, Any],
        confirmation_plan: Mapping[
            str,
            Any,
        ],
        literature_request: Mapping[
            str,
            Any,
        ],
    ) -> dict:

        (
            cluster_id,
            candidate_records,
            top_markers,
        ) = self.build_candidate_records(
            pilot,
            confirmation_plan,
            literature_request,
        )

        candidate_queries = []

        for record in candidate_records:

            candidate = {
                "name":
                    record[
                        "candidate"
                    ],

                #
                # Rescue candidates were not
                # generated by Agent 2 and therefore
                # have no Agent 2 support-gene list.
                #
                "supporting_genes":
                    [],
            }

            candidate_queries.append(
                build_candidate_query(
                    request=(
                        literature_request
                    ),
                    candidate=candidate,
                    top_marker_count=(
                        self.top_marker_count
                    ),
                )
            )

        unique_texts = []
        text_to_index = {}

        for record in candidate_records:

            for passage in record[
                "passages"
            ]:

                model_text = _clean(
                    f"{passage['title']}. "
                    f"{passage['text']}"
                )

                passage[
                    "_model_text"
                ] = model_text

                if (
                    model_text
                    not in text_to_index
                ):
                    text_to_index[
                        model_text
                    ] = len(
                        unique_texts
                    )

                    unique_texts.append(
                        model_text
                    )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_id
            )
        )

        model = (
            AutoModel.from_pretrained(
                self.model_id
            ).to(
                device
            )
        )

        query_embeddings = (
            embed_texts(
                texts=candidate_queries,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=(
                    self.batch_size
                ),
                max_length=(
                    self.max_length
                ),
            )
        )

        query_embeddings = (
            _normalize_embeddings(
                query_embeddings
            )
        )

        if unique_texts:

            passage_embeddings = (
                embed_texts(
                    texts=unique_texts,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    batch_size=(
                        self.batch_size
                    ),
                    max_length=(
                        self.max_length
                    ),
                )
            )

            passage_embeddings = (
                _normalize_embeddings(
                    passage_embeddings
                )
            )

        else:

            passage_embeddings = (
                np.empty(
                    (
                        0,
                        query_embeddings.shape[
                            1
                        ],
                    ),
                    dtype=float,
                )
            )

        result_records = []

        for candidate_index, record in enumerate(
            candidate_records
        ):

            passages = record[
                "passages"
            ]

            candidate_query = (
                candidate_queries[
                    candidate_index
                ]
            )

            if not passages:

                result_records.append(
                    {
                        "candidate":
                            record[
                                "candidate"
                            ],

                        "ontology_id":
                            record[
                                "ontology_id"
                            ],

                        "query_id":
                            record[
                                "query_id"
                            ],

                        "candidate_query":
                            candidate_query,

                        "retrieved_paper_count":
                            record[
                                "retrieved_paper_count"
                            ],

                        "eligible_passage_count":
                            0,

                        "eligible_unique_paper_count":
                            0,

                        "selected_top_k":
                            0,

                        "mean_topk_similarity":
                            None,

                        "max_topk_similarity":
                            None,

                        "median_topk_similarity":
                            None,

                        "insufficient_candidate_evidence":
                            True,

                        "top_passages":
                            [],
                    }
                )

                continue

            scored = []

            for passage in passages:

                embedding_index = (
                    text_to_index[
                        passage[
                            "_model_text"
                        ]
                    ]
                )

                similarity = float(
                    np.dot(
                        query_embeddings[
                            candidate_index
                        ],
                        passage_embeddings[
                            embedding_index
                        ],
                    )
                )

                scored.append(
                    {
                        **{
                            key:
                                value
                            for key, value
                            in passage.items()
                            if key
                            != "_model_text"
                        },

                        "similarity":
                            similarity,
                    }
                )

            scored.sort(
                key=lambda item: (
                    -item[
                        "similarity"
                    ],
                    item[
                        "retrieval_position"
                    ],
                    item[
                        "passage_index"
                    ],
                )
            )

            selected = []
            paper_counts = {}

            for item in scored:

                key = item[
                    "record_key"
                ]

                count = (
                    paper_counts.get(
                        key,
                        0,
                    )
                )

                if (
                    count
                    >= self.maximum_passages_per_paper
                ):
                    continue

                selected.append(
                    item
                )

                paper_counts[
                    key
                ] = count + 1

                if (
                    len(selected)
                    >= self.top_k_passages
                ):
                    break

            similarities = np.asarray(
                [
                    item[
                        "similarity"
                    ]
                    for item
                    in selected
                ],
                dtype=float,
            )

            unique_eligible_papers = len(
                {
                    item[
                        "record_key"
                    ]
                    for item
                    in scored
                }
            )

            if len(
                similarities
            ):

                mean_similarity = float(
                    similarities.mean()
                )

                max_similarity = float(
                    similarities.max()
                )

                median_similarity = float(
                    np.median(
                        similarities
                    )
                )

            else:
                mean_similarity = None
                max_similarity = None
                median_similarity = None

            result_records.append(
                {
                    "candidate":
                        record[
                            "candidate"
                        ],

                    "ontology_id":
                        record[
                            "ontology_id"
                        ],

                    "query_id":
                        record[
                            "query_id"
                        ],

                    "candidate_query":
                        candidate_query,

                    "retrieved_paper_count":
                        record[
                            "retrieved_paper_count"
                        ],

                    "eligible_passage_count":
                        len(
                            scored
                        ),

                    "eligible_unique_paper_count":
                        unique_eligible_papers,

                    "selected_top_k":
                        len(
                            selected
                        ),

                    "mean_topk_similarity":
                        mean_similarity,

                    "max_topk_similarity":
                        max_similarity,

                    "median_topk_similarity":
                        median_similarity,

                    "insufficient_candidate_evidence":
                        not bool(
                            selected
                        ),

                    "top_passages":
                        selected,
                }
            )

        return {
            "schema_name":
                (
                    "RescueConfirmation"
                    "PubMedBERTReranking"
                ),

            "schema_version":
                "2.0",

            "cluster_id":
                cluster_id,

            "model_id":
                self.model_id,

            "expert_label_used":
                False,

            "reference_label_used":
                False,

            "candidate_pool_rule": (
                "Each candidate is evaluated "
                "only over passages derived from "
                "that candidate's own target-context "
                "confirmation query, and the canonical "
                "candidate label must be explicitly "
                "present in the title or passage text."
            ),

            "supporting_gene_policy":
                (
                    "supporting_genes=[] "
                    "for ontology-derived "
                    "rescue candidates"
                ),

            "top_marker_count":
                self.top_marker_count,

            "top_k_passages":
                self.top_k_passages,

            "maximum_characters_per_passage":
                (
                    self.maximum_characters_per_passage
                ),

            "overlap_sentences":
                self.overlap_sentences,

            "maximum_passages_per_paper":
                (
                    self.maximum_passages_per_paper
                ),

            "important_note": (
                "PubMedBERT cosine similarity "
                "represents semantic retrieval "
                "relevance, not annotation "
                "confidence or probability."
            ),

            "results":
                result_records,
        }
