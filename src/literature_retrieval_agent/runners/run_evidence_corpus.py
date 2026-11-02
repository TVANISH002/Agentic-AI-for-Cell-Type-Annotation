
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import cast

from ..evidence_corpus_builder import (
    EvidenceCorpusBuilder,
    EvidenceCorpusConfig,
)
from ..schemas import (
    EvidenceCorpusResult,
    PaperRankingResult,
    RelevanceLabel,
    RetrievalPilotResult,
)


def _project_root() -> Path:
    """
    Return the existing project root.
    """

    return Path(__file__).resolve().parents[3]


def _cluster_directory(
    cluster_id: str,
) -> Path:
    """
    Return the standard Agent 3 directory for a cluster.

    No cluster identifier is hardcoded.
    """

    return (
        _project_root()
        / "data"
        / "literature_retrieval_agent"
        / "clusters"
        / f"cluster_{cluster_id}"
    )


def _write_json(
    result: EvidenceCorpusResult,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _write_csv(
    result: EvidenceCorpusResult,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "passage_id",
        "source_rank",
        "paper_relevance_score",
        "paper_relevance_label",
        "retrieval_score",
        "title",
        "publication_year",
        "journal",
        "pmid",
        "pmcid",
        "doi",
        "passage_source",
        "passage_index",
        "candidate_mentions",
        "marker_mentions",
        "context_matches",
        "text",
    ]

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for passage in result.passages:
            publication = passage.publication

            writer.writerow(
                {
                    "passage_id": (
                        passage.passage_id
                    ),
                    "source_rank": (
                        passage.source_rank
                    ),
                    "paper_relevance_score": (
                        passage.paper_relevance_score
                    ),
                    "paper_relevance_label": (
                        passage.paper_relevance_label
                    ),
                    "retrieval_score": (
                        passage.retrieval_score
                    ),
                    "title": (
                        publication.title
                    ),
                    "publication_year": (
                        publication.publication_year
                        or ""
                    ),
                    "journal": (
                        publication.journal
                        or ""
                    ),
                    "pmid": (
                        publication.pmid
                        or ""
                    ),
                    "pmcid": (
                        publication.pmcid
                        or ""
                    ),
                    "doi": (
                        publication.doi
                        or ""
                    ),
                    "passage_source": (
                        passage.passage_source
                    ),
                    "passage_index": (
                        passage.passage_index
                    ),
                    "candidate_mentions": "; ".join(
                        passage.candidate_mentions
                    ),
                    "marker_mentions": "; ".join(
                        passage.marker_mentions
                    ),
                    "context_matches": "; ".join(
                        passage.context_matches
                    ),
                    "text": passage.text,
                }
            )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a citation-preserving abstract "
            "evidence corpus for the Literature "
            "Retrieval Agent grounded RAG stage."
        )
    )

    parser.add_argument(
        "--cluster-id",
        required=True,
        help=(
            "Cluster identifier from the existing "
            "Agent 2 and Agent 3 handoff."
        ),
    )

    parser.add_argument(
        "--pilot-json",
        type=Path,
        default=None,
        help=(
            "Optional retrieval-pilot JSON path. "
            "The standard cluster path is used "
            "when omitted."
        ),
    )

    parser.add_argument(
        "--ranking-json",
        type=Path,
        default=None,
        help=(
            "Optional paper-ranking JSON path. "
            "The standard cluster path is used "
            "when omitted."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help=(
            "Optional output directory. The standard "
            "cluster directory is used when omitted."
        ),
    )

    parser.add_argument(
        "--min-paper-relevance",
        choices=(
            "high",
            "medium",
            "low",
        ),
        default="medium",
        help=(
            "Lowest paper-relevance category included "
            "in the evidence corpus."
        ),
    )

    parser.add_argument(
        "--maximum-papers",
        type=int,
        default=30,
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
        "--maximum-passages-per-candidate",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--maximum-passages-per-paper-per-candidate",
        type=int,
        default=1,
        help=(
            "Maximum passages selected from the same "
            "paper for one candidate."
        ),
    )

    parser.add_argument(
        "--maximum-context-only-passages-per-candidate",
        type=int,
        default=4,
        help=(
            "Maximum candidate-only passages that do not "
            "mention any observed cluster marker."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()

    cluster_id = str(args.cluster_id)

    default_directory = _cluster_directory(
        cluster_id
    )

    if args.pilot_json is not None:
        pilot_path = (
            args.pilot_json.resolve()
        )
    else:
        pilot_path = (
            default_directory
            / (
                f"cluster_{cluster_id}"
                "_retrieval_pilot.json"
            )
        )

    if args.ranking_json is not None:
        ranking_path = (
            args.ranking_json.resolve()
        )
    else:
        ranking_path = (
            default_directory
            / (
                f"cluster_{cluster_id}"
                "_ranked_papers.json"
            )
        )

    if not pilot_path.is_file():
        raise FileNotFoundError(
            "Retrieval pilot JSON was not found: "
            f"{pilot_path}"
        )

    if not ranking_path.is_file():
        raise FileNotFoundError(
            "Paper ranking JSON was not found: "
            f"{ranking_path}"
        )

    pilot = (
        RetrievalPilotResult
        .model_validate_json(
            pilot_path.read_text(
                encoding="utf-8-sig"
            )
        )
    )

    ranking = (
        PaperRankingResult
        .model_validate_json(
            ranking_path.read_text(
                encoding="utf-8-sig"
            )
        )
    )

    if (
        str(pilot.request.cluster_id)
        != cluster_id
    ):
        raise ValueError(
            "Pilot JSON cluster ID does not "
            "match --cluster-id."
        )

    if (
        str(ranking.cluster_id)
        != cluster_id
    ):
        raise ValueError(
            "Ranking JSON cluster ID does not "
            "match --cluster-id."
        )

    minimum_relevance = cast(
        RelevanceLabel,
        args.min_paper_relevance,
    )

    config = EvidenceCorpusConfig(
        min_paper_relevance=(
            minimum_relevance
        ),
        maximum_papers=(
            args.maximum_papers
        ),
        maximum_characters_per_passage=(
            args.maximum_characters_per_passage
        ),
        overlap_sentences=(
            args.overlap_sentences
        ),
        maximum_passages_per_candidate=(
            args.maximum_passages_per_candidate
        ),
        maximum_passages_per_paper_per_candidate=(
            args
            .maximum_passages_per_paper_per_candidate
        ),
        maximum_context_only_passages_per_candidate=(
            args
            .maximum_context_only_passages_per_candidate
        ),
    )

    result = EvidenceCorpusBuilder(
        config=config
    ).build(
        request=pilot.request,
        ranking=ranking,
        source_pilot_file=str(
            pilot_path
        ),
        source_ranking_file=str(
            ranking_path
        ),
    )

    if args.output_directory is not None:
        output_directory = (
            args.output_directory.resolve()
        )
    else:
        output_directory = default_directory

    json_path = (
        output_directory
        / (
            f"cluster_{cluster_id}"
            "_evidence_corpus.json"
        )
    )

    csv_path = (
        output_directory
        / (
            f"cluster_{cluster_id}"
            "_evidence_passages.csv"
        )
    )

    _write_json(
        result=result,
        path=json_path,
    )

    _write_csv(
        result=result,
        path=csv_path,
    )

    print(
        "Literature evidence corpus completed "
        "successfully."
    )

    print(
        f"Cluster: {result.cluster_id}"
    )

    print(
        "Papers selected: "
        f"{result.selected_paper_count}"
    )

    print(
        "Papers without abstracts: "
        f"{result.excluded_no_abstract_count}"
    )

    print(
        "Passages created: "
        f"{result.passage_count}"
    )

    for candidate_set in (
        result.candidate_passage_sets
    ):
        print(
            f"{candidate_set.candidate_cell_type}: "
            f"selected="
            f"{len(candidate_set.passage_ids)}, "
            f"available="
            f"{candidate_set.available_passage_count}"
        )

    print(
        f"Warnings: {len(result.warnings)}"
    )

    print(
        f"JSON: {json_path}"
    )

    print(
        f"CSV: {csv_path}"
    )


if __name__ == "__main__":
    main()