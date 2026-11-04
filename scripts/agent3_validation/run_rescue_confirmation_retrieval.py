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
            "Execute target-context confirmation queries "
            "for ontology-derived Agent 3 rescue candidates."
        )
    )

    parser.add_argument(
        "--literature-request",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--confirmation-plan",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-directory",
        required=True,
        type=Path,
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
    plan_path = args.confirmation_plan.resolve()
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

    plan = json.loads(
        plan_path.read_text(
            encoding="utf-8-sig"
        )
    )

    if str(plan.get("cluster_id")) != str(
        request.cluster_id
    ):
        raise ValueError(
            "Confirmation-plan cluster does not "
            "match LiteratureRequest cluster."
        )

    if plan.get(
        "reference_label_used",
        False,
    ):
        raise ValueError(
            "Confirmation plan reports reference-label use."
        )

    if plan.get(
        "expert_label_used",
        False,
    ):
        raise ValueError(
            "Confirmation plan reports expert-label use."
        )

    plan_queries = plan.get(
        "queries",
        [],
    )

    if not plan_queries:
        raise ValueError(
            "Confirmation query plan contains no queries."
        )

    #
    # Convert our validation confirmation plan into
    # the repository's existing LiteratureQuery schema.
    #
    queries = []

    for item in plan_queries:
        candidate = str(
            item.get(
                "candidate",
                "",
            )
        ).strip()

        query_text = str(
            item.get(
                "query",
                "",
            )
        ).strip()

        query_id = str(
            item.get(
                "query_id",
                "",
            )
        ).strip()

        markers = [
            str(x).strip().upper()
            for x in item.get(
                "markers",
                [],
            )
            if str(x).strip()
        ]

        if not candidate:
            raise ValueError(
                f"{query_id}: missing candidate."
            )

        if not query_text:
            raise ValueError(
                f"{query_id}: missing query."
            )

        #
        # Important provenance check:
        # candidate must occur in its own
        # target-context confirmation query.
        #
        if candidate.casefold() not in query_text.casefold():
            raise ValueError(
                f"{query_id}: candidate is absent "
                "from confirmation query."
            )

        queries.append(
            LiteratureQuery(
                query_id=query_id,
                purpose="cluster_marker_context",
                query=query_text,
                candidate_annotations=[
                    candidate
                ],
                marker_genes=markers,
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
        open_access_only=args.open_access_only,
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
            "RescueCandidateConfirmationRetrievalManifest",

        "schema_version":
            "1.0",

        "cluster_id":
            str(request.cluster_id),

        "source_literature_request":
            str(request_path),

        "source_confirmation_plan":
            str(plan_path),

        "expert_label_used":
            False,

        "reference_label_used":
            False,

        "candidate_source":
            "ontology_grounded_marker_rescue_discovery",

        "query_count":
            len(result.query_results),

        "unique_paper_count":
            len(result.unique_papers),

        "warning_count":
            len(result.warnings),

        "page_size":
            args.page_size,

        "open_access_only":
            args.open_access_only,

        "query_results": [
            {
                "query_id":
                    qr.query.query_id,

                "candidate":
                    (
                        qr.query.candidate_annotations[0]
                        if qr.query.candidate_annotations
                        else ""
                    ),

                "marker_genes":
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
            "rescue_confirmation_manifest.json"
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
        "Rescue candidate confirmation retrieval "
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
        "Unique papers:",
        len(result.unique_papers),
    )

    print(
        "Warnings:",
        len(result.warnings),
    )

    print(
        "Reference label used:",
        False,
    )

    print()

    print(
        "CONFIRMATION QUERY RESULTS"
    )

    print("=" * 120)

    for qr in result.query_results:
        candidate = (
            qr.query.candidate_annotations[0]
            if qr.query.candidate_annotations
            else ""
        )

        print(
            f"{qr.query.query_id}: "
            f"{candidate} "
            f"| hits={qr.hit_count} "
            f"| returned={qr.returned_count}"
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
