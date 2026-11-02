
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_ID = "NeuML/pubmedbert-base-embeddings"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []

    for value in values:
        value = str(value).strip()
        if not value:
            continue

        key = value.upper()
        if key not in seen:
            seen.add(key)
            output.append(value)

    return output


def mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@torch.no_grad()
def embed_texts(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []

    model.eval()

    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[start : start + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        output = model(**encoded)

        pooled = mean_pool(
            output.last_hidden_state,
            encoded["attention_mask"],
        )

        pooled = F.normalize(pooled, p=2, dim=1)

        embeddings.append(
            pooled.cpu().numpy()
        )

    return np.vstack(embeddings)


def build_candidate_query(
    request: dict[str, Any],
    candidate: dict[str, Any],
    top_marker_count: int,
) -> str:
    marker_items = request.get("marker_genes", [])
    all_markers = [
        str(item.get("gene", "")).strip()
        for item in marker_items
        if isinstance(item, dict) and item.get("gene")
    ]

    top_markers = all_markers[:top_marker_count]

    supporting_genes = [
        str(gene).strip()
        for gene in candidate.get("supporting_genes", [])
        if str(gene).strip()
    ]

    marker_context = unique_keep_order(
        supporting_genes + top_markers
    )

    return (
        "Cell type annotation evidence. "
        f"Candidate cell type: {candidate.get('name', '')}. "
        f"Species: {request.get('species', '')}. "
        f"Tissue: {request.get('tissue', '')}. "
        f"Disease context: {request.get('disease_context', '')}. "
        f"Platform: {request.get('platform', '')}. "
        f"Candidate supporting genes: {', '.join(supporting_genes)}. "
        f"Observed cluster markers: {', '.join(marker_context)}. "
        "Find passages that support this candidate cell type or marker program."
    )


def passage_to_text(passage: dict[str, Any]) -> str:
    publication = passage.get("publication", {})
    if not isinstance(publication, dict):
        publication = {}

    title = str(publication.get("title", "") or "")
    text = str(passage.get("text", "") or "")

    return f"{title}. {text}".strip()


def run_pubmedbert_reranking(
    corpus_json: Path,
    output_directory: Path,
    expected_cluster_id: str,
    model_id: str,
    candidate_count: int,
    top_k_passages: int,
    top_marker_count: int,
    batch_size: int,
    max_length: int,
) -> None:
    corpus = load_json(corpus_json)

    cluster_id = str(corpus.get("cluster_id", ""))

    if cluster_id != str(expected_cluster_id):
        raise ValueError(
            "Evidence-corpus cluster ID does not match "
            f"--cluster-id: {cluster_id!r} != "
            f"{str(expected_cluster_id)!r}."
        )

    request = corpus.get("literature_request", {})
    if not isinstance(request, dict):
        raise ValueError("Missing literature_request in evidence corpus.")

    candidates = request.get("candidate_annotations", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("No candidate_annotations found in evidence corpus.")

    if candidate_count > 0:
        candidates = candidates[:candidate_count]

    passages = corpus.get("passages", [])
    if not isinstance(passages, list) or not passages:
        raise ValueError("No passages found in evidence corpus.")

    output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Cluster: {cluster_id}")
    print(f"Candidates: {[candidate.get('name') for candidate in candidates]}")
    print(f"Passages: {len(passages)}")
    print(f"Model: {model_id}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)

    candidate_queries = [
        build_candidate_query(
            request=request,
            candidate=candidate,
            top_marker_count=top_marker_count,
        )
        for candidate in candidates
    ]

    passage_texts = [
        passage_to_text(passage)
        for passage in passages
    ]

    query_embeddings = embed_texts(
        texts=candidate_queries,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )

    passage_embeddings = embed_texts(
        texts=passage_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )

    similarity_matrix = query_embeddings @ passage_embeddings.T

    top_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    result_json: dict[str, Any] = {
        "cluster_id": cluster_id,
        "model_id": model_id,
        "corpus_json": str(corpus_json),
        "candidate_count": len(candidates),
        "passage_count": len(passages),
        "top_k_passages": top_k_passages,
        "top_marker_count": top_marker_count,
        "passage_selection_method": (
            "candidate_eligible_pool_then_pubmedbert"
        ),
        "candidate_eligibility_rule": (
            "candidate mention OR Agent2 supporting-gene mention"
        ),
        "candidate_pool_counts": {},
        "results": [],
    }

    for candidate_index, candidate in enumerate(candidates):
        candidate_name = str(
            candidate.get(
                "name",
                "",
            )
        ).strip()

        if not candidate_name:
            raise ValueError(
                "Candidate annotation contains "
                "an empty name."
            )

        supporting_genes = {
            str(gene).strip().upper()
            for gene in candidate.get(
                "supporting_genes",
                [],
            )
            if str(gene).strip()
        }

        eligible_indices: list[int] = []

        for passage_index, passage in enumerate(
            passages
        ):
            candidate_mentions = {
                str(value).strip().casefold()
                for value in passage.get(
                    "candidate_mentions",
                    [],
                )
                if str(value).strip()
            }

            marker_mentions = {
                str(value).strip().upper()
                for value in passage.get(
                    "marker_mentions",
                    [],
                )
                if str(value).strip()
            }

            candidate_is_mentioned = (
                candidate_name.casefold()
                in candidate_mentions
            )

            supporting_gene_is_mentioned = bool(
                supporting_genes.intersection(
                    marker_mentions
                )
            )

            if (
                candidate_is_mentioned
                or supporting_gene_is_mentioned
            ):
                eligible_indices.append(
                    passage_index
                )

        result_json[
            "candidate_pool_counts"
        ][candidate_name] = len(
            eligible_indices
        )

        if not eligible_indices:
            summary_rows.append(
                {
                    "cluster_id": cluster_id,
                    "candidate": candidate_name,
                    "candidate_upstream_rank": (
                        candidate.get(
                            "upstream_rank"
                        )
                    ),
                    "supporting_genes": "; ".join(
                        str(gene)
                        for gene in candidate.get(
                            "supporting_genes",
                            [],
                        )
                    ),
                    "available_passage_count": 0,
                    "top_k_passages": 0,
                    "mean_similarity": None,
                    "max_similarity": None,
                    "median_similarity": None,
                }
            )

            result_json["results"].append(
                {
                    "candidate": (
                        candidate_name
                    ),
                    "candidate_query": (
                        candidate_queries[
                            candidate_index
                        ]
                    ),
                    "eligible_passage_count": 0,
                    "mean_similarity": None,
                    "max_similarity": None,
                    "median_similarity": None,
                    "top_passages": [],
                    "insufficient_candidate_evidence": (
                        True
                    ),
                }
            )

            continue

        eligible_scores = (
            similarity_matrix[
                candidate_index,
                eligible_indices,
            ]
        )

        local_order = np.argsort(
            eligible_scores
        )[::-1]

        selected_local_indices = (
            local_order[
                :top_k_passages
            ]
        )

        top_indices = [
            eligible_indices[
                int(local_index)
            ]
            for local_index
            in selected_local_indices
        ]

        top_scores = np.asarray(
            [
                similarity_matrix[
                    candidate_index,
                    passage_index,
                ]
                for passage_index
                in top_indices
            ],
            dtype=float,
        )

        mean_similarity = float(
            np.mean(top_scores)
        )

        max_similarity = float(
            np.max(top_scores)
        )

        median_similarity = float(
            np.median(top_scores)
        )

        summary_rows.append(
            {
                "cluster_id": cluster_id,
                "candidate": candidate_name,
                "candidate_upstream_rank": (
                    candidate.get(
                        "upstream_rank"
                    )
                ),
                "supporting_genes": "; ".join(
                    str(gene)
                    for gene in candidate.get(
                        "supporting_genes",
                        [],
                    )
                ),
                "available_passage_count": len(
                    eligible_indices
                ),
                "top_k_passages": len(
                    top_indices
                ),
                "mean_similarity": round(
                    mean_similarity,
                    6,
                ),
                "max_similarity": round(
                    max_similarity,
                    6,
                ),
                "median_similarity": round(
                    median_similarity,
                    6,
                ),
            }
        )

        candidate_result = {
            "candidate": candidate_name,
            "candidate_query": (
                candidate_queries[
                    candidate_index
                ]
            ),
            "eligible_passage_count": len(
                eligible_indices
            ),
            "mean_similarity": (
                mean_similarity
            ),
            "max_similarity": (
                max_similarity
            ),
            "median_similarity": (
                median_similarity
            ),
            "insufficient_candidate_evidence": (
                False
            ),
            "top_passages": [],
        }

        for local_rank, passage_index in enumerate(
            top_indices,
            start=1,
        ):
            passage = passages[
                int(passage_index)
            ]

            publication = passage.get(
                "publication",
                {},
            )

            if not isinstance(
                publication,
                dict,
            ):
                publication = {}

            similarity = float(
                similarity_matrix[
                    candidate_index,
                    passage_index,
                ]
            )

            row = {
                "cluster_id": cluster_id,
                "candidate": candidate_name,
                "rank_within_candidate": (
                    local_rank
                ),
                "similarity": round(
                    similarity,
                    6,
                ),
                "passage_id": passage.get(
                    "passage_id",
                    "",
                ),
                "paper_relevance_score": (
                    passage.get(
                        "paper_relevance_score",
                        "",
                    )
                ),
                "paper_relevance_label": (
                    passage.get(
                        "paper_relevance_label",
                        "",
                    )
                ),
                "source_rank": passage.get(
                    "source_rank",
                    "",
                ),
                "title": publication.get(
                    "title",
                    "",
                ),
                "publication_year": (
                    publication.get(
                        "publication_year",
                        "",
                    )
                ),
                "journal": publication.get(
                    "journal",
                    "",
                ),
                "pmid": publication.get(
                    "pmid",
                    "",
                ),
                "pmcid": publication.get(
                    "pmcid",
                    "",
                ),
                "doi": publication.get(
                    "doi",
                    "",
                ),
                "candidate_mentions": "; ".join(
                    passage.get(
                        "candidate_mentions",
                        [],
                    )
                ),
                "marker_mentions": "; ".join(
                    passage.get(
                        "marker_mentions",
                        [],
                    )
                ),
                "context_matches": "; ".join(
                    passage.get(
                        "context_matches",
                        [],
                    )
                ),
                "text_preview": str(
                    passage.get(
                        "text",
                        "",
                    )
                )[:500],
            }

            top_rows.append(
                row
            )

            candidate_result[
                "top_passages"
            ].append(
                row
            )

        result_json[
            "results"
        ].append(
            candidate_result
        )

    summary_csv = (
        output_directory
        / f"cluster_{cluster_id}_pubmedbert_summary.csv"
    )

    top_passages_csv = (
        output_directory
        / f"cluster_{cluster_id}_pubmedbert_top_passages.csv"
    )

    results_json = (
        output_directory
        / f"cluster_{cluster_id}_pubmedbert_results.json"
    )

    pd.DataFrame(summary_rows).to_csv(
        summary_csv,
        index=False,
    )

    pd.DataFrame(top_rows).to_csv(
        top_passages_csv,
        index=False,
    )

    save_json(
        result_json,
        results_json,
    )

    print("\nSaved outputs:")
    print(summary_csv)
    print(top_passages_csv)
    print(results_json)

    print("\nSummary:")
    print(pd.DataFrame(summary_rows).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PubMedBERT embedding reranking for one cluster evidence corpus."
    )

    parser.add_argument(
        "--cluster-id",
        required=True,
    )

    parser.add_argument(
        "--corpus-json",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
    )

    parser.add_argument(
        "--candidate-count",
        type=int,
        default=3,
        help="Use top N candidate annotations from the corpus. Use 0 for all.",
    )

    parser.add_argument(
        "--top-k-passages",
        type=int,
        default=4,
        help="Number of top passages used to compute mean similarity.",
    )

    parser.add_argument(
        "--top-marker-count",
        type=int,
        default=50,
        help="Number of cluster marker genes included in candidate query.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_pubmedbert_reranking(
        corpus_json=args.corpus_json,
        output_directory=args.output_directory,
        expected_cluster_id=str(args.cluster_id),
        model_id=args.model_id,
        candidate_count=args.candidate_count,
        top_k_passages=args.top_k_passages,
        top_marker_count=args.top_marker_count,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()