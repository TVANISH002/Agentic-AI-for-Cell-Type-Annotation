from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from literature_retrieval_agent.agent import (
    LiteratureRetrievalAgent,
)
from literature_retrieval_agent.config import (
    DEFAULT_CONFIG,
)
from literature_retrieval_agent.paths import (
    agent2_cluster_json,
    agent3_cluster_output_dir,
)
from literature_retrieval_agent.report_writer import (
    write_pilot_reports,
)
from literature_retrieval_agent.request_builder import (
    build_literature_request,
    load_agent2_handoff,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the retrieval-pilot CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build Agent 3 queries from Agent 2 "
            "output and run a Europe PMC "
            "literature retrieval pilot."
        )
    )

    parser.add_argument(
        "--cluster-id",
        required=True,
        help="Cluster identifier, for example 6.",
    )

    parser.add_argument(
        "--disease-context",
        required=True,
        help=(
            "Disease or experimental context, "
            "for example 'breast cancer'."
        ),
    )

    parser.add_argument(
        "--agent2-json",
        type=Path,
        default=None,
        help=(
            "Optional Agent 2 MarkerEvidence JSON. "
            "When omitted, the standard cluster "
            "output location is used."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional Agent 3 output directory."
        ),
    )

    parser.add_argument(
        "--candidate-count",
        type=int,
        default=3,
        help=(
            "Number of top Agent 2 candidates "
            "included in this run."
        ),
    )

    parser.add_argument(
        "--marker-count",
        type=int,
        default=None,
        help=(
            "Optional number of ranked marker genes "
            "preserved in the literature request. "
            "The default preserves all markers."
        ),
    )

    parser.add_argument(
        "--question",
        default=None,
        help=(
            "Optional specific biological question."
        ),
    )

    parser.add_argument(
        "--spatial-context",
        default=None,
        help=(
            "Optional spatial or histological context."
        ),
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=5,
        help=(
            "Maximum Europe PMC records retrieved "
            "per query during this pilot."
        ),
    )

    parser.add_argument(
        "--max-marker-genes-per-query",
        type=int,
        default=6,
        help=(
            "Maximum number of marker genes "
            "placed in one Europe PMC query."
        ),
    )

    parser.add_argument(
        "--open-access-only",
        action="store_true",
        help=(
            "Restrict retrieval to records marked "
            "open access by Europe PMC."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build and print queries without "
            "calling Europe PMC."
        ),
    )

    return parser


def main() -> None:
    """Execute the retrieval pilot."""

    args = build_parser().parse_args()

    source = (
        args.agent2_json
        if args.agent2_json is not None
        else agent2_cluster_json(
            args.cluster_id
        )
    )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else agent3_cluster_output_dir(
            args.cluster_id
        )
    )

    config = replace(
        DEFAULT_CONFIG,
        page_size=args.page_size,
        max_marker_genes_per_query=(
            args.max_marker_genes_per_query
        ),
        open_access_only=(
            args.open_access_only
        ),
    )

    payload, resolved_source = (
        load_agent2_handoff(source)
    )

    request = build_literature_request(
        payload,
        source_file=resolved_source,
        disease_context=(
            args.disease_context
        ),
        candidate_count=(
            args.candidate_count
        ),
        marker_count=args.marker_count,
        question=args.question,
        spatial_context=args.spatial_context,
        config=config,
    )

    agent = LiteratureRetrievalAgent(
        config
    )

    if args.dry_run:
        query_plan = agent.build_query_plan(
            request
        )

        agent.close()

        print(
            "Literature Retrieval Agent "
            "dry run completed successfully."
        )

        print(
            f"Cluster: {request.cluster_id}"
        )

        print(
            "Markers preserved: "
            f"{len(request.marker_genes)}"
        )

        print(
            "Candidates: "
            f"{[item.name for item in request.candidate_annotations]}"
        )

        print(
            f"Queries generated: "
            f"{len(query_plan.queries)}"
        )

        for query in query_plan.queries:
            print(
                f"{query.query_id}: "
                f"{query.query}"
            )

        return

    result = agent.run_pilot(
        request
    )

    paths = write_pilot_reports(
        result,
        output_dir,
    )

    print(
        "Literature Retrieval Agent "
        "pilot completed successfully."
    )

    print(
        f"Cluster: {request.cluster_id}"
    )

    print(
        "Queries executed: "
        f"{len(result.query_results)}"
    )

    print(
        "Unique papers retrieved: "
        f"{len(result.unique_papers)}"
    )

    print(
        f"Warnings: {len(result.warnings)}"
    )

    for name, path in paths.items():
        print(
            f"{name}: {path}"
        )


if __name__ == "__main__":
    main()