from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..extraction_job_builder import (
    ExtractionJobBuilder,
    ExtractionJobConfig,
)
from ..schemas import EvidenceCorpusResult


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8-sig")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object: {path}"
        )

    return payload


def _reranker_rows(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    current_rows = payload.get("results")

    if (
        isinstance(current_rows, list)
        and current_rows
    ):
        return (
            [
                row
                for row in current_rows
                if isinstance(row, dict)
            ],
            "results",
        )

    historical_rows = payload.get(
        "candidate_summaries"
    )

    if (
        isinstance(historical_rows, list)
        and historical_rows
    ):
        return (
            [
                row
                for row in historical_rows
                if isinstance(row, dict)
            ],
            "candidate_summaries",
        )

    raise ValueError(
        "PubMedBERT JSON contains neither "
        "'results' nor 'candidate_summaries'."
    )


def _candidate_name(
    row: dict[str, Any],
) -> str:
    return str(
        row.get("candidate")
        or row.get("candidate_cell_type")
        or ""
    ).strip()


def _top_passage_ids(
    row: dict[str, Any],
) -> list[str]:
    passages = row.get(
        "top_passages",
        [],
    )

    if not isinstance(passages, list):
        return []

    output: list[str] = []
    seen: set[str] = set()

    for passage in passages:
        if not isinstance(passage, dict):
            continue

        passage_id = str(
            passage.get("passage_id", "")
        ).strip()

        if (
            passage_id
            and passage_id not in seen
        ):
            seen.add(passage_id)
            output.append(passage_id)

    return output


def _build_pubmedbert_map(
    payload: dict[str, Any],
) -> tuple[
    dict[str, list[str]],
    str,
]:
    rows, source_format = _reranker_rows(
        payload
    )

    passage_ids_by_candidate: dict[
        str,
        list[str],
    ] = {}

    for row in rows:
        candidate = _candidate_name(row)

        if not candidate:
            raise ValueError(
                "PubMedBERT result contains "
                "a candidate with no name."
            )

        if candidate in passage_ids_by_candidate:
            raise ValueError(
                "Duplicate PubMedBERT candidate: "
                f"{candidate!r}."
            )

        passage_ids = _top_passage_ids(
            row
        )

        if not passage_ids:
            raise ValueError(
                "PubMedBERT candidate has no "
                f"top passages: {candidate!r}."
            )

        passage_ids_by_candidate[
            candidate
        ] = passage_ids

    return (
        passage_ids_by_candidate,
        source_format,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Connect PubMedBERT-selected passages "
            "to the existing grounded extraction "
            "job builder without modifying the "
            "working core modules."
        )
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
        "--pubmedbert-json",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--maximum-passages-per-job",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--allow-candidate-subset",
        action="store_true",
        help=(
            "Allow historical PubMedBERT runs "
            "that evaluated only a subset of "
            "Agent 2 candidates."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    cluster_id = str(
        args.cluster_id
    )

    corpus_path = (
        args.corpus_json.resolve()
    )

    pubmedbert_path = (
        args.pubmedbert_json.resolve()
    )

    output_path = (
        args.output_json.resolve()
    )

    if not corpus_path.is_file():
        raise FileNotFoundError(
            f"Corpus not found: {corpus_path}"
        )

    if not pubmedbert_path.is_file():
        raise FileNotFoundError(
            "PubMedBERT result not found: "
            f"{pubmedbert_path}"
        )

    corpus = (
        EvidenceCorpusResult
        .model_validate_json(
            corpus_path.read_text(
                encoding="utf-8-sig"
            )
        )
    )

    if str(corpus.cluster_id) != cluster_id:
        raise ValueError(
            "Corpus cluster ID does not match "
            f"--cluster-id: {corpus.cluster_id!r} "
            f"!= {cluster_id!r}."
        )

    pubmedbert = _read_json(
        pubmedbert_path
    )

    reranker_cluster = str(
        pubmedbert.get(
            "cluster_id",
            "",
        )
    )

    if reranker_cluster != cluster_id:
        raise ValueError(
            "PubMedBERT cluster ID does not "
            f"match --cluster-id: "
            f"{reranker_cluster!r} != "
            f"{cluster_id!r}."
        )

    (
        passage_ids_by_candidate,
        source_format,
    ) = _build_pubmedbert_map(
        pubmedbert
    )

    request_candidates = [
        candidate.name
        for candidate
        in corpus
        .literature_request
        .candidate_annotations
    ]

    request_candidate_set = set(
        request_candidates
    )

    reranker_candidate_set = set(
        passage_ids_by_candidate
    )

    unexpected_candidates = sorted(
        reranker_candidate_set
        - request_candidate_set
    )

    if unexpected_candidates:
        raise ValueError(
            "PubMedBERT contains candidates "
            "that are not present in the "
            "LiteratureRequest: "
            + ", ".join(
                unexpected_candidates
            )
        )

    missing_candidates = sorted(
        request_candidate_set
        - reranker_candidate_set
    )

    if (
        missing_candidates
        and not args.allow_candidate_subset
    ):
        raise ValueError(
            "PubMedBERT does not contain all "
            "Agent 2 candidates. Missing: "
            + ", ".join(
                missing_candidates
            )
            + ". Use --allow-candidate-subset "
            "only for historical validation runs."
        )

    known_passage_ids = {
        passage.passage_id
        for passage in corpus.passages
    }

    updated_candidate_sets = []

    for candidate_set in (
        corpus.candidate_passage_sets
    ):
        candidate = (
            candidate_set
            .candidate_cell_type
        )

        if (
            candidate
            not in passage_ids_by_candidate
        ):
            continue

        selected_ids = (
            passage_ids_by_candidate[
                candidate
            ]
        )

        unknown_ids = [
            passage_id
            for passage_id in selected_ids
            if passage_id
            not in known_passage_ids
        ]

        if unknown_ids:
            raise ValueError(
                "PubMedBERT references passage "
                "IDs that are absent from the "
                f"corpus for {candidate!r}: "
                + ", ".join(
                    unknown_ids
                )
            )

        updated_candidate_sets.append(
            candidate_set.model_copy(
                update={
                    "passage_ids": (
                        selected_ids
                    ),
                    "available_passage_count": (
                        max(
                            candidate_set
                            .available_passage_count,
                            len(selected_ids),
                        )
                    ),
                }
            )
        )

    if not updated_candidate_sets:
        raise ValueError(
            "No PubMedBERT candidate passages "
            "could be connected to the corpus."
        )

    corpus_payload = (
        corpus.model_dump()
    )

    corpus_payload[
        "candidate_passage_sets"
    ] = [
        candidate_set.model_dump()
        for candidate_set
        in updated_candidate_sets
    ]

    reranked_corpus = (
        EvidenceCorpusResult
        .model_validate(
            corpus_payload
        )
    )

    builder = ExtractionJobBuilder(
        config=ExtractionJobConfig(
            maximum_passages_per_job=(
                args
                .maximum_passages_per_job
            )
        )
    )

    plan = builder.build(
        corpus=reranked_corpus,
        source_corpus_file=str(
            corpus_path
        ),
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        plan.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    audit_path = output_path.with_name(
        output_path.stem
        + "_selection_audit.json"
    )

    audit = {
        "cluster_id": cluster_id,
        "selection_method": (
            "PubMedBERT semantic reranking"
        ),
        "pubmedbert_model_id": (
            pubmedbert.get("model_id")
        ),
        "pubmedbert_result_file": (
            str(pubmedbert_path)
        ),
        "source_corpus_file": (
            str(corpus_path)
        ),
        "pubmedbert_schema_detected": (
            source_format
        ),
        "request_candidates": (
            request_candidates
        ),
        "reranked_candidates": (
            list(
                passage_ids_by_candidate
            )
        ),
        "missing_candidates": (
            missing_candidates
        ),
        "candidate_passage_ids": (
            passage_ids_by_candidate
        ),
        "expert_label_used": False,
    }

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "PubMedBERT -> extraction "
        "handoff completed successfully."
    )
    print(
        f"Cluster: {cluster_id}"
    )
    print(
        f"Jobs created: "
        f"{plan.job_count}"
    )

    for job in plan.jobs:
        print(
            f"{job.candidate_cell_type}: "
            f"{len(job.passage_ids)} "
            "PubMedBERT-selected passages"
        )

    if missing_candidates:
        print(
            "Historical subset warning: "
            + ", ".join(
                missing_candidates
            )
        )

    print(
        f"Extraction jobs: "
        f"{output_path}"
    )
    print(
        f"Selection audit: "
        f"{audit_path}"
    )


if __name__ == "__main__":
    main()
