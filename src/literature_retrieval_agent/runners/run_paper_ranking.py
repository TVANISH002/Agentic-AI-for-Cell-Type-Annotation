from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ..utf8_console import configure_utf8_console

configure_utf8_console()

from ..paper_ranker import PaperRanker
from ..schemas import (
    PaperRankingResult,
    RetrievalPilotResult,
)


def _project_root() -> Path:
    """
    Return the existing Biological Knowledge Retrieval Agent root.

    File location:
    project_root/src/literature_retrieval_agent/runners/this_file.py
    """
    return Path(__file__).resolve().parents[3]


def _default_pilot_path(
    cluster_id: str,
) -> Path:
    """
    Construct the standard retrieval-pilot path dynamically.

    No cluster number is hardcoded.
    """
    cluster_directory = (
        _project_root()
        / "data"
        / "literature_retrieval_agent"
        / "clusters"
        / f"cluster_{cluster_id}"
    )

    return (
        cluster_directory
        / f"cluster_{cluster_id}_retrieval_pilot.json"
    )


def _write_json(
    result: PaperRankingResult,
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
    result: PaperRankingResult,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "rank",
        "relevance_score",
        "relevance_label",
        "title",
        "publication_year",
        "journal",
        "pmid",
        "pmcid",
        "doi",
        "is_open_access",
        "is_preprint",
        "matched_candidates",
        "matched_markers",
        "matched_query_ids",
        "rationale",
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

        for ranked_paper in result.ranked_papers:
            publication = ranked_paper.publication
            breakdown = ranked_paper.breakdown

            writer.writerow(
                {
                    "rank": ranked_paper.rank,
                    "relevance_score": (
                        ranked_paper.relevance_score
                    ),
                    "relevance_label": (
                        ranked_paper.relevance_label
                    ),
                    "title": publication.title,
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
                    "is_open_access": (
                        publication.is_open_access
                    ),
                    "is_preprint": (
                        publication.is_preprint
                    ),
                    "matched_candidates": "; ".join(
                        breakdown.matched_candidates
                    ),
                    "matched_markers": "; ".join(
                        breakdown.matched_markers
                    ),
                    "matched_query_ids": "; ".join(
                        breakdown.matched_query_ids
                    ),
                    "rationale": "; ".join(
                        ranked_paper.rationale
                    ),
                }
            )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank papers retrieved by the Literature "
            "Retrieval Agent. This command ranks retrieval "
            "relevance; it does not assign a cell type."
        )
    )

    parser.add_argument(
        "--cluster-id",
        required=True,
        help=(
            "Cluster identifier used by the existing "
            "Agent 2 and Agent 3 handoff."
        ),
    )

    parser.add_argument(
        "--pilot-json",
        type=Path,
        default=None,
        help=(
            "Optional path to a retrieval-pilot JSON. "
            "If omitted, the standard Agent 3 output "
            "path is used."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help=(
            "Optional output directory. If omitted, "
            "outputs are saved beside the pilot JSON."
        ),
    )

    parser.add_argument(
        "--show-top",
        type=int,
        default=5,
        help=(
            "Number of highest-ranked paper titles "
            "to print in the terminal."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()

    if args.show_top < 0:
        raise ValueError(
            "--show-top cannot be negative."
        )

    cluster_id = str(args.cluster_id)

    if args.pilot_json is not None:
        pilot_path = args.pilot_json.resolve()
    else:
        pilot_path = _default_pilot_path(
            cluster_id
        ).resolve()

    if not pilot_path.is_file():
        raise FileNotFoundError(
            "Retrieval pilot JSON was not found: "
            f"{pilot_path}"
        )

    pilot = RetrievalPilotResult.model_validate_json(
        pilot_path.read_text(
            encoding="utf-8-sig"
        )
    )

    pilot_cluster_id = str(
        pilot.request.cluster_id
    )

    if pilot_cluster_id != cluster_id:
        raise ValueError(
            "The requested cluster ID does not match "
            "the cluster ID inside the pilot JSON: "
            f"{cluster_id!r} != "
            f"{pilot_cluster_id!r}."
        )

    result = PaperRanker().rank(pilot)

    if args.output_directory is not None:
        output_directory = (
            args.output_directory.resolve()
        )
    else:
        output_directory = pilot_path.parent

    output_stem = (
        f"cluster_{cluster_id}_ranked_papers"
    )

    json_path = (
        output_directory
        / f"{output_stem}.json"
    )

    csv_path = (
        output_directory
        / f"{output_stem}.csv"
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
        "Literature paper ranking completed "
        "successfully."
    )

    print(
        f"Cluster: {result.cluster_id}"
    )

    print(
        f"Papers ranked: {result.total_papers}"
    )

    print(
        "Relevance groups: "
        f"high={result.high_relevance_count}, "
        f"medium={result.medium_relevance_count}, "
        f"low={result.low_relevance_count}"
    )

    print(
        f"JSON: {json_path}"
    )

    print(
        f"CSV: {csv_path}"
    )

    for ranked_paper in result.ranked_papers[
        : args.show_top
    ]:
        print(
            f"#{ranked_paper.rank} "
            f"[{ranked_paper.relevance_label}] "
            f"{ranked_paper.relevance_score:.3f} - "
            f"{ranked_paper.publication.title}"
        )


if __name__ == "__main__":
    main()
