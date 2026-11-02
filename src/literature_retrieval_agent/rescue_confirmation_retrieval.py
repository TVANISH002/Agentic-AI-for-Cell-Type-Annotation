from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from literature_retrieval_agent.config import (
    DEFAULT_CONFIG,
)
from literature_retrieval_agent.context_retrieval_agent import (
    ContextExpandedRetrievalAgent,
)
from literature_retrieval_agent.europe_pmc_client import (
    EuropePMCClient,
)
from literature_retrieval_agent.schemas import (
    LiteratureQuery,
    LiteratureQueryPlan,
    LiteratureRequest,
)


def build_confirmation_query_plan(
    request: LiteratureRequest,
    confirmation_plan: Mapping[str, Any],
) -> LiteratureQueryPlan:

   
    # Cluster consistency
    

    if str(
        confirmation_plan.get(
            "cluster_id"
        )
    ) != str(
        request.cluster_id
    ):
        raise ValueError(
            "Confirmation-plan cluster does not "
            "match LiteratureRequest cluster."
        )

    
    # Leakage guards
   

    if confirmation_plan.get(
        "reference_label_used",
        False,
    ):
        raise ValueError(
            "Confirmation plan reports "
            "reference-label use."
        )

    if confirmation_plan.get(
        "expert_label_used",
        False,
    ):
        raise ValueError(
            "Confirmation plan reports "
            "expert-label use."
        )

    plan_queries = (
        confirmation_plan.get(
            "queries",
            [],
        )
    )

    if not plan_queries:
        raise ValueError(
            "Confirmation query plan "
            "contains no queries."
        )

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
            str(value)
            .strip()
            .upper()
            for value
            in item.get(
                "markers",
                [],
            )
            if str(value).strip()
        ]

        if not query_id:
            raise ValueError(
                "Confirmation query is "
                "missing query_id."
            )

        if not candidate:
            raise ValueError(
                f"{query_id}: "
                "missing candidate."
            )

        if not query_text:
            raise ValueError(
                f"{query_id}: "
                "missing query."
            )

        
        # Candidate-specific confirmation must actually contain the candidate.
        
        if (
            candidate.casefold()
            not in query_text.casefold()
        ):
            raise ValueError(
                f"{query_id}: candidate is "
                "absent from confirmation query."
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

    return LiteratureQueryPlan(
        schema_name="LiteratureQueryPlan",
        schema_version="1.0",
        cluster_id=str(
            request.cluster_id
        ),
        queries=queries,
    )


def run_confirmation_retrieval(
    request: LiteratureRequest,
    confirmation_plan: Mapping[str, Any],
    *,
    page_size: int = 10,
    open_access_only: bool = True,
):

    if page_size < 1:
        raise ValueError(
            "page_size must be >= 1."
        )

    query_plan = (
        build_confirmation_query_plan(
            request,
            confirmation_plan,
        )
    )

    config = replace(
        DEFAULT_CONFIG,
        page_size=page_size,
        open_access_only=(
            open_access_only
        ),
    )

    
    # Retrieval itself is agnostic to tissue, species, cluster, candidate identity, and marker identity.
    
    with EuropePMCClient(
        config
    ) as client:

        result = (
            ContextExpandedRetrievalAgent(
                client=client,
            ).run(
                request,
                query_plan,
            )
        )

    manifest = {
        "schema_name":
            (
                "RescueCandidateConfirmation"
                "RetrievalManifest"
            ),

        "schema_version":
            "2.0",

        "cluster_id":
            str(
                request.cluster_id
            ),

        "expert_label_used":
            False,

        "reference_label_used":
            False,

        "candidate_source":
            (
                "ontology_grounded_"
                "marker_rescue_discovery"
            ),

        "query_count":
            len(
                result.query_results
            ),

        "unique_paper_count":
            len(
                result.unique_papers
            ),

        "warning_count":
            len(
                result.warnings
            ),

        "page_size":
            page_size,

        "open_access_only":
            open_access_only,

        "query_results": [
            {
                "query_id":
                    qr.query.query_id,

                "candidate":
                    (
                        qr.query
                        .candidate_annotations[0]
                        if qr.query
                        .candidate_annotations
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
            for qr
            in result.query_results
        ],
    }

    return result, manifest






