from __future__ import annotations

import argparse
import csv
import json
import shutil
import re
from pathlib import Path
from typing import Any

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


def normalize_space(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


def candidate_pattern(label: str):
    """
    Conservative canonical-label matching.

    Allows:
      mast cell / mast cells
      T cell / T-cell / T cells / T-cells

    Ontology synonyms are intentionally not used here.
    """
    label = normalize_space(label)

    words = label.split()

    if not words:
        return None

    escaped = [
        re.escape(word)
        for word in words
    ]

    separator = r"(?:\s+|-)"

    if words[-1].casefold() == "cell":
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


def gene_pattern(gene: str):
    return re.compile(
        rf"(?<![A-Za-z0-9])"
        rf"{re.escape(gene)}"
        rf"(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )


def normalize_embeddings(
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


def publication_key(
    paper: dict[str, Any],
) -> str:
    return (
        normalize_space(
            paper.get("record_key")
        )
        or normalize_space(
            paper.get("pmid")
        )
        or normalize_space(
            paper.get("doi")
        )
        or normalize_space(
            paper.get("title")
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run PubMedBERT reranking over candidate-specific "
            "Agent 3 rescue-confirmation literature."
        )
    )

    parser.add_argument(
        "--pilot-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--confirmation-plan",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--literature-request",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-directory",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--model-id",
        default="NeuML/pubmedbert-base-embeddings",
    )

    parser.add_argument(
        "--top-marker-count",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--top-k-passages",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--maximum-characters-per-passage",
        type=int,
        default=1200,
    )

    parser.add_argument(
        "--overlap-sentences",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--maximum-passages-per-paper",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
    )

    args = parser.parse_args()

    pilot = json.loads(
        args.pilot_json.read_text(
            encoding="utf-8-sig"
        )
    )

    plan = json.loads(
        args.confirmation_plan.read_text(
            encoding="utf-8-sig"
        )
    )

    request = json.loads(
        args.literature_request.read_text(
            encoding="utf-8-sig"
        )
    )

    pilot_request = (
        pilot.get("request", {})
        or {}
    )

    pilot_query_plan = (
        pilot.get("query_plan", {})
        or {}
    )

    pilot_cluster_id = str(
        pilot.get("cluster_id")
        or (
            pilot_request.get("cluster_id")
            if isinstance(
                pilot_request,
                dict,
            )
            else ""
        )
        or (
            pilot_query_plan.get("cluster_id")
            if isinstance(
                pilot_query_plan,
                dict,
            )
            else ""
        )
        or ""
    )

    plan_cluster_id = str(
        plan.get("cluster_id", "")
    )

    request_cluster_id = str(
        request.get("cluster_id", "")
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
            "Cluster mismatch across input artifacts: "
            f"pilot={pilot_cluster_id!r}, "
            f"plan={plan_cluster_id!r}, "
            f"request={request_cluster_id!r}"
        )

    cluster_id = request_cluster_id

    if plan.get(
        "reference_label_used",
        False,
    ):
        raise ValueError(
            "Confirmation plan indicates "
            "reference-label use."
        )

    if plan.get(
        "expert_label_used",
        False,
    ):
        raise ValueError(
            "Confirmation plan indicates "
            "expert-label use."
        )

    plan_lookup = {
        item["query_id"]: item
        for item in plan.get(
            "queries",
            []
        )
    }

    #
    # Cluster markers remain exactly those in the
    # original LiteratureRequest.
    #
    marker_items = request.get(
        "marker_genes",
        []
    )

    observed_markers = [
        normalize_space(
            item.get("gene", "")
        ).upper()
        for item in marker_items
        if (
            isinstance(item, dict)
            and normalize_space(
                item.get("gene", "")
            )
        )
    ]

    top_markers = observed_markers[
        : args.top_marker_count
    ]

    marker_patterns = {
        marker: gene_pattern(marker)
        for marker in top_markers
    }

    candidate_records = []

    #
    # Candidate-specific passage pools.
    #
    for qr in pilot.get(
        "query_results",
        []
    ):
        query_obj = (
            qr.get("query", {})
            or {}
        )

        query_id = normalize_space(
            query_obj.get(
                "query_id",
                qr.get(
                    "query_id",
                    "",
                ),
            )
        )

        plan_item = plan_lookup.get(
            query_id
        )

        if not plan_item:
            continue

        candidate_name = normalize_space(
            plan_item.get(
                "candidate",
                "",
            )
        )

        ontology_id = normalize_space(
            plan_item.get(
                "ontology_id",
                "",
            )
        )

        if not candidate_name:
            continue

        candidate_re = candidate_pattern(
            candidate_name
        )

        papers = (
            qr.get("papers")
            or qr.get("publications")
            or qr.get("results")
            or qr.get("records")
            or []
        )

        candidate_passages = []

        for paper_position, paper in enumerate(
            papers,
            start=1,
        ):
            title = normalize_space(
                paper.get(
                    "title",
                    ""
                )
            )

            abstract = normalize_space(
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

            record_key = publication_key(
                paper
            )

            if not record_key:
                continue

            chunks = _chunk_abstract(
                abstract=abstract,
                maximum=(
                    args.maximum_characters_per_passage
                ),
                overlap_sentences=(
                    args.overlap_sentences
                ),
            )

            for passage_index, chunk in enumerate(
                chunks,
                start=1,
            ):
                chunk = normalize_space(
                    chunk
                )

                searchable = normalize_space(
                    f"{title}. {chunk}"
                )

                #
                # Conservative evidence rule:
                # candidate must actually appear in the
                # title or current abstract passage.
                #
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

                passage_id = (
                    f"{record_key}:"
                    f"abstract:{passage_index}"
                )

                candidate_passages.append(
                    {
                        "passage_id":
                            passage_id,

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
                    len(papers),

                "passages":
                    candidate_passages,
            }
        )

    #
    # Preserve query-plan ordering.
    #
    order = {
        item["query_id"]: index
        for index, item in enumerate(
            plan.get("queries", [])
        )
    }

    candidate_records.sort(
        key=lambda item: order.get(
            item["query_id"],
            10**9,
        )
    )

    #
    # Build PubMedBERT candidate queries.
    #
    candidate_queries = []

    for record in candidate_records:
        candidate = {
            "name":
                record["candidate"],

            #
            # Novel rescue candidates have no
            # Agent 2 supporting-gene list.
            #
            "supporting_genes":
                [],
        }

        candidate_queries.append(
            build_candidate_query(
                request=request,
                candidate=candidate,
                top_marker_count=(
                    args.top_marker_count
                ),
            )
        )

    #
    # Deduplicate identical passage text for model
    # embedding only. Candidate-specific provenance
    # remains separate.
    #
    unique_texts = []
    text_to_index = {}

    for record in candidate_records:
        for passage in record[
            "passages"
        ]:
            model_text = normalize_space(
                f"{passage['title']}. "
                f"{passage['text']}"
            )

            passage[
                "_model_text"
            ] = model_text

            if model_text not in text_to_index:
                text_to_index[
                    model_text
                ] = len(
                    unique_texts
                )

                unique_texts.append(
                    model_text
                )

    print(
        "Cluster:",
        cluster_id,
    )

    print(
        "Candidates:",
        len(candidate_records),
    )

    print(
        "Unique passage texts:",
        len(unique_texts),
    )

    print(
        "Model:",
        args.model_id,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            args.model_id
        )
    )

    model = (
        AutoModel.from_pretrained(
            args.model_id
        ).to(device)
    )

    query_embeddings = embed_texts(
        texts=candidate_queries,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    query_embeddings = normalize_embeddings(
        query_embeddings
    )

    if unique_texts:
        passage_embeddings = embed_texts(
            texts=unique_texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        passage_embeddings = (
            normalize_embeddings(
                passage_embeddings
            )
        )
    else:
        passage_embeddings = np.empty(
            (
                0,
                query_embeddings.shape[1],
            ),
            dtype=float,
        )

    summary_rows = []
    top_rows = []
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
            summary_rows.append(
                {
                    "cluster_id":
                        cluster_id,

                    "candidate":
                        record["candidate"],

                    "ontology_id":
                        record["ontology_id"],

                    "retrieved_papers":
                        record[
                            "retrieved_paper_count"
                        ],

                    "eligible_passages":
                        0,

                    "eligible_unique_papers":
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
                }
            )

            result_records.append(
                {
                    "candidate":
                        record["candidate"],

                    "ontology_id":
                        record["ontology_id"],

                    "query_id":
                        record["query_id"],

                    "candidate_query":
                        candidate_query,

                    "eligible_passage_count":
                        0,

                    "eligible_unique_paper_count":
                        0,

                    "top_passages":
                        [],

                    "insufficient_candidate_evidence":
                        True,
                }
            )

            continue

        scored = []

        for passage in passages:
            passage_embedding_index = (
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
                        passage_embedding_index
                    ],
                )
            )

            scored.append(
                {
                    **{
                        key: value
                        for key, value
                        in passage.items()
                        if key != "_model_text"
                    },

                    "similarity":
                        similarity,
                }
            )

        scored.sort(
            key=lambda item: (
                -item["similarity"],
                item[
                    "retrieval_position"
                ],
                item[
                    "passage_index"
                ],
            )
        )

        #
        # Prevent multiple chunks from one paper
        # dominating the final top-k evidence packet.
        #
        selected = []

        paper_counts = {}

        for item in scored:
            key = item[
                "record_key"
            ]

            count = paper_counts.get(
                key,
                0,
            )

            if (
                count
                >= args.maximum_passages_per_paper
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
                >= args.top_k_passages
            ):
                break

        similarities = np.asarray(
            [
                item["similarity"]
                for item in selected
            ],
            dtype=float,
        )

        unique_eligible_papers = len(
            {
                item["record_key"]
                for item in scored
            }
        )

        if len(similarities):
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

        summary_rows.append(
            {
                "cluster_id":
                    cluster_id,

                "candidate":
                    record["candidate"],

                "ontology_id":
                    record["ontology_id"],

                "retrieved_papers":
                    record[
                        "retrieved_paper_count"
                    ],

                "eligible_passages":
                    len(scored),

                "eligible_unique_papers":
                    unique_eligible_papers,

                "selected_top_k":
                    len(selected),

                "mean_topk_similarity":
                    (
                        round(
                            mean_similarity,
                            6,
                        )
                        if mean_similarity
                        is not None
                        else None
                    ),

                "max_topk_similarity":
                    (
                        round(
                            max_similarity,
                            6,
                        )
                        if max_similarity
                        is not None
                        else None
                    ),

                "median_topk_similarity":
                    (
                        round(
                            median_similarity,
                            6,
                        )
                        if median_similarity
                        is not None
                        else None
                    ),

                "insufficient_candidate_evidence":
                    not bool(selected),
            }
        )

        for rank, item in enumerate(
            selected,
            start=1,
        ):
            row = {
                "cluster_id":
                    cluster_id,

                "candidate":
                    record["candidate"],

                "ontology_id":
                    record["ontology_id"],

                "query_id":
                    record["query_id"],

                "rank_within_candidate":
                    rank,

                "similarity":
                    round(
                        item[
                            "similarity"
                        ],
                        6,
                    ),

                "passage_id":
                    item[
                        "passage_id"
                    ],

                "record_key":
                    item[
                        "record_key"
                    ],

                "retrieval_position":
                    item[
                        "retrieval_position"
                    ],

                "title":
                    item[
                        "title"
                    ],

                "pmid":
                    item[
                        "pmid"
                    ],

                "pmcid":
                    item[
                        "pmcid"
                    ],

                "doi":
                    item[
                        "doi"
                    ],

                "journal":
                    item[
                        "journal"
                    ],

                "publication_year":
                    item[
                        "publication_year"
                    ],

                "markers_mentioned":
                    "; ".join(
                        item[
                            "markers_mentioned"
                        ]
                    ),

                "text":
                    item[
                        "text"
                    ],
            }

            top_rows.append(
                row
            )

        result_records.append(
            {
                "candidate":
                    record["candidate"],

                "ontology_id":
                    record["ontology_id"],

                "query_id":
                    record["query_id"],

                "candidate_query":
                    candidate_query,

                "retrieved_paper_count":
                    record[
                        "retrieved_paper_count"
                    ],

                "eligible_passage_count":
                    len(scored),

                "eligible_unique_paper_count":
                    unique_eligible_papers,

                "selected_top_k":
                    len(selected),

                "mean_topk_similarity":
                    mean_similarity,

                "max_topk_similarity":
                    max_similarity,

                "median_topk_similarity":
                    median_similarity,

                "insufficient_candidate_evidence":
                    not bool(selected),

                "top_passages":
                    [
                        {
                            key: value
                            for key, value
                            in item.items()
                            if key != "_model_text"
                        }
                        for item in selected
                    ],
            }
        )

    #
    # Descriptive ordering for terminal display only.
    # Similarity is semantic relevance, not confidence.
    #
    display_rows = sorted(
        summary_rows,
        key=lambda row: (
            -(
                row[
                    "mean_topk_similarity"
                ]
                if row[
                    "mean_topk_similarity"
                ]
                is not None
                else -1.0
            ),
            row[
                "candidate"
            ].casefold(),
        )
    )

    output_directory = (
        args.output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_directory
        / (
            f"cluster_{cluster_id}_"
            "rescue_confirmation_"
            "pubmedbert_summary.csv"
        )
    )

    top_path = (
        output_directory
        / (
            f"cluster_{cluster_id}_"
            "rescue_confirmation_"
            "pubmedbert_top_passages.csv"
        )
    )

    json_path = (
        output_directory
        / (
            f"cluster_{cluster_id}_"
            "rescue_confirmation_"
            "pubmedbert_results.json"
        )
    )

    if summary_rows:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    summary_rows[0].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                summary_rows
            )

    if top_rows:
        with top_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    top_rows[0].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                top_rows
            )

    output = {
        "schema_name":
            "RescueConfirmationPubMedBERTReranking",

        "schema_version":
            "1.0",

        "cluster_id":
            cluster_id,

        "model_id":
            args.model_id,

        "expert_label_used":
            False,

        "reference_label_used":
            False,

        "candidate_pool_rule": (
            "Each candidate is evaluated only over "
            "passages derived from that candidate's own "
            "target-context confirmation query, and the "
            "canonical candidate label must be explicitly "
            "present in the title or passage text."
        ),

        "supporting_gene_policy":
            "supporting_genes=[] for rescue candidates",

        "top_marker_count":
            args.top_marker_count,

        "top_k_passages":
            args.top_k_passages,

        "maximum_characters_per_passage":
            args.maximum_characters_per_passage,

        "overlap_sentences":
            args.overlap_sentences,

        "maximum_passages_per_paper":
            args.maximum_passages_per_paper,

        "important_note": (
            "PubMedBERT cosine similarity is treated as "
            "semantic retrieval relevance, not annotation "
            "confidence or probability."
        ),

        "results":
            result_records,
    }

    json_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Rescue-confirmation PubMedBERT "
        "reranking completed successfully."
    )

    print(
        "Reference label used:",
        False,
    )

    print()
    print(
        "TOP 25 BY SEMANTIC RELEVANCE "
        "(NOT CONFIDENCE)"
    )

    print("=" * 130)

    for index, row in enumerate(
        display_rows[:25],
        start=1,
    ):
        print(
            f"{index:02d}. "
            f"{row['candidate']} "
            f"| papers="
            f"{row['eligible_unique_papers']} "
            f"| passages="
            f"{row['eligible_passages']} "
            f"| topK="
            f"{row['selected_top_k']} "
            f"| mean="
            f"{row['mean_topk_similarity']} "
            f"| max="
            f"{row['max_topk_similarity']} "
            f"| median="
            f"{row['median_topk_similarity']}"
        )

    print()

    print(
        "Summary:",
        summary_path,
    )

    print(
        "Top passages:",
        top_path,
    )

    print(
        "Results:",
        json_path,
    )


if __name__ == "__main__":
    main()


