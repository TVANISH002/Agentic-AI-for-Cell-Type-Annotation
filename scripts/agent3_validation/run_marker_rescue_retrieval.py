from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from literature_retrieval_agent.config import DEFAULT_CONFIG
from literature_retrieval_agent.context_retrieval_agent import (
    ContextExpandedRetrievalAgent,
)
from literature_retrieval_agent.europe_pmc_client import (
    EuropePMCClient,
)
from literature_retrieval_agent.report_writer import (
    write_pilot_reports,
)
from literature_retrieval_agent.schemas import (
    LiteratureQuery,
    LiteratureQueryPlan,
    LiteratureRequest,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Execute candidate-independent marker-rescue "
            "queries through the existing Europe PMC client."
        )
    )

    parser.add_argument(
        "--literature-request",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--rescue-plan",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--open-access-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    request_path = args.literature_request.resolve()
    rescue_path = args.rescue_plan.resolve()
    output_directory = args.output_directory.resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    request = LiteratureRequest.model_validate_json(
        request_path.read_text(
            encoding="utf-8-sig"
        )
    )

    rescue = json.loads(
        rescue_path.read_text(
            encoding="utf-8-sig"
        )
    )

    if str(rescue.get("cluster_id")) != str(
        request.cluster_id
    ):
        raise ValueError(
            "Marker-rescue plan cluster does not "
            "match LiteratureRequest cluster."
        )

    #
    # Leakage audit:
    # upstream candidate names must not appear
    # in any rescue query.
    #
    candidate_names = [
        c.name
        for c in request.candidate_annotations
    ]

    leakage = []

    for item in rescue["queries"]:
        query_text = str(
            item["query"]
        ).casefold()

        for candidate in candidate_names:
            if candidate.casefold() in query_text:
                leakage.append(
                    {
                        "query_id":
                            item["query_id"],
                        "candidate":
                            candidate,
                    }
                )

    if leakage:
        raise ValueError(
            "Candidate leakage detected in "
            f"marker-rescue queries: {leakage}"
        )

    #
    # Convert validation rescue queries into
    # the existing Agent 3 LiteratureQuery schema.
    #
    # We reuse cluster_marker_context as the
    # existing schema's candidate-independent
    # marker-query purpose. The external rescue
    # plan preserves the more specific query family.
    #
    queries = []

    for item in rescue["queries"]:
        queries.append(
            LiteratureQuery(
                query_id=item["query_id"],
                purpose="cluster_marker_context",
                query=item["query"],
                candidate_annotations=[],
                marker_genes=[
                    str(x).upper()
                    for x in item.get(
                        "markers",
                        []
                    )
                ],
            )
        )

    query_plan = LiteratureQueryPlan(
        schema_name="LiteratureQueryPlan",
        schema_version="1.0",
        cluster_id=str(request.cluster_id),
        queries=queries,
    )

    config = replace(
        DEFAULT_CONFIG,
        page_size=args.page_size,
        open_access_only=(
            args.open_access_only
        ),
    )

    client = EuropePMCClient(
        config
    )

    result = ContextExpandedRetrievalAgent(
        client=client,
    ).run(
        request,
        query_plan,
    )

    paths = write_pilot_reports(
        result,
        output_directory,
    )

    manifest = {
        "schema_name":
            "MarkerRescueRetrievalManifest",
        "schema_version":
            "1.0",
        "cluster_id":
            str(request.cluster_id),
        "source_literature_request":
            str(request_path),
        "source_rescue_plan":
            str(rescue_path),
        "candidate_names_used_in_queries":
            False,
        "expert_label_used":
            False,
        "page_size":
            args.page_size,
        "open_access_only":
            args.open_access_only,
        "query_count":
            len(result.query_results),
        "unique_paper_count":
            len(result.unique_papers),
        "warning_count":
            len(result.warnings),
        "query_summary": [
            {
                "query_id":
                    qr.query.query_id,
                "markers":
                    qr.query.marker_genes,
                "hit_count":
                    qr.hit_count,
                "returned_count":
                    qr.returned_count,
                "warnings":
                    qr.warnings,
            }
            for qr in result.query_results
        ],
    }

    manifest_path = (
        output_directory
        / (
            f"cluster_{request.cluster_id}_"
            "marker_rescue_retrieval_manifest.json"
        )
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "Marker-rescue Europe PMC retrieval "
        "completed successfully."
    )
    print(
        "Cluster:",
        request.cluster_id,
    )
    print(
        "Queries executed:",
        len(result.query_results),
    )
    print(
        "Unique papers retrieved:",
        len(result.unique_papers),
    )
    print(
        "Warnings:",
        len(result.warnings),
    )
    print(
        "Candidate leakage:",
        False,
    )
    print()

    for qr in result.query_results:
        print(
            f"{qr.query.query_id}: "
            f"hits={qr.hit_count}, "
            f"returned={qr.returned_count}"
        )

    print()

    for name, path in paths.items():
        print(
            f"{name}: {path}"
        )

    print(
        "manifest:",
        manifest_path,
    )


if __name__ == "__main__":
    main()
