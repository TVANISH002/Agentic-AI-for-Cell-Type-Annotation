from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from literature_retrieval_agent.config import DEFAULT_CONFIG
from literature_retrieval_agent.context_expansion import (
    ContextQueryConfig,
    load_context_expansion_terms,
)
from literature_retrieval_agent.context_query_builder import (
    ContextExpandedQueryBuilder,
)
from literature_retrieval_agent.context_report_writer import (
    write_context_plan_reports,
    write_context_retrieval_reports,
)
from literature_retrieval_agent.context_retrieval_agent import (
    ContextExpandedRetrievalAgent,
)
from literature_retrieval_agent.europe_pmc_client import EuropePMCClient
from literature_retrieval_agent.schemas import LiteratureRequest


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_request_path(cluster_id: str) -> Path:
    return (
        _project_root()
        / "data"
        / "literature_retrieval_agent"
        / "clusters"
        / f"cluster_{cluster_id}"
        / f"cluster_{cluster_id}_literature_request.json"
    )


def _default_output_directory(cluster_id: str) -> Path:
    return (
        _project_root()
        / "data"
        / "literature_retrieval_agent"
        / "clusters"
        / f"cluster_{cluster_id}_retrieval_v2_context"
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run generalized context-expanded Europe PMC retrieval without "
            "changing the baseline retrieval or model outputs."
        )
    )
    parser.add_argument("--cluster-id", required=True)
    parser.add_argument("--request-json", type=Path, default=None)
    parser.add_argument("--context-terms-json", type=Path, default=None)
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--page-size", type=int, default=10)
    parser.add_argument("--marker-batch-size", type=int, default=6)
    parser.add_argument("--max-marker-batches", type=int, default=4)
    parser.add_argument(
        "--open-access-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use Europe PMC OPEN_ACCESS:Y filtering. Enabled by default; "
            "use --no-open-access-only only for a documented comparison."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build, audit, print, and save the v2 query plan only.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    cluster_id = str(args.cluster_id)
    request_path = (
        args.request_json.resolve()
        if args.request_json is not None
        else _default_request_path(cluster_id).resolve()
    )
    output_directory = (
        args.output_directory.resolve()
        if args.output_directory is not None
        else _default_output_directory(cluster_id).resolve()
    )

    if not request_path.is_file():
        raise FileNotFoundError(
            f"LiteratureRequest JSON was not found: {request_path}"
        )

    request = LiteratureRequest.model_validate_json(
        request_path.read_text(encoding="utf-8-sig")
    )

    if str(request.cluster_id) != cluster_id:
        raise ValueError(
            "Requested cluster does not match LiteratureRequest cluster: "
            f"{cluster_id!r} != {request.cluster_id!r}."
        )

    expansion_terms = load_context_expansion_terms(args.context_terms_json)
    query_config = ContextQueryConfig(
        marker_batch_size=args.marker_batch_size,
        max_marker_batches=args.max_marker_batches,
    )
    retrieval_config = replace(
        DEFAULT_CONFIG,
        page_size=args.page_size,
        open_access_only=args.open_access_only,
        query_plan_schema_version=query_config.query_plan_schema_version,
        retrieval_schema_version=query_config.retrieval_schema_version,
    )
    builder = ContextExpandedQueryBuilder(
        query_config=query_config,
        expansion_terms=expansion_terms,
    )
    query_plan, audit = builder.build_with_audit(request)

    if not audit.valid:
        raise ValueError(
            "Context query audit failed: " + "; ".join(audit.errors)
        )

    if args.dry_run:
        paths = write_context_plan_reports(
            request=request,
            query_plan=query_plan,
            audit=audit,
            expansion_terms=expansion_terms,
            query_config=query_config,
            retrieval_config=retrieval_config,
            output_directory=output_directory,
        )
        print("Context-expanded retrieval dry run completed successfully.")
    else:
        client = EuropePMCClient(retrieval_config)
        result = ContextExpandedRetrievalAgent(
            client=client,
            query_config=query_config,
        ).run(request, query_plan)
        paths = write_context_retrieval_reports(
            result=result,
            audit=audit,
            expansion_terms=expansion_terms,
            query_config=query_config,
            retrieval_config=retrieval_config,
            output_directory=output_directory,
        )
        print("Context-expanded retrieval completed successfully.")
        print(f"Unique papers retrieved: {len(result.unique_papers)}")
        print(f"Warnings: {len(result.warnings)}")

    print(f"Cluster: {request.cluster_id}")
    print(f"Queries generated: {len(query_plan.queries)}")
    print(f"Open-access only: {retrieval_config.open_access_only}")
    print(
        "Candidate query counts: "
        + ", ".join(
            f"{name}={count}"
            for name, count in audit.candidate_query_counts.items()
        )
    )
    print(f"Fairness audit valid: {audit.valid}")
    print(f"Expert label used as query input: {audit.expert_label_used_as_query_input}")

    for name, path in paths.items():
        print(f"{name}: {path}")

    if args.dry_run:
        for query in query_plan.queries:
            print(f"{query.query_id}: {query.query}")


if __name__ == "__main__":
    main()
