from __future__ import annotations

from literature_retrieval_agent.schemas import (
    LiteratureQuery,
    LiteratureQueryPlan,
    LiteratureRequest,
    PublicationRecord,
    QueryRetrievalResult,
    RetrievalPilotResult,
)


def _request_signature(request: LiteratureRequest) -> tuple[object, ...]:
    return (
        str(request.cluster_id),
        request.species.casefold(),
        request.tissue.casefold(),
        request.disease_context.casefold(),
        request.platform.casefold(),
        tuple(marker.gene for marker in request.marker_genes),
        tuple(
            candidate.name.casefold()
            for candidate in request.candidate_annotations
        ),
    )


def _prefer_richer_record(
    current: PublicationRecord,
    candidate: PublicationRecord,
) -> PublicationRecord:
    current_score = (
        int(bool(current.abstract)),
        len(current.abstract or ""),
        len(current.authors),
        int(current.is_open_access),
    )
    candidate_score = (
        int(bool(candidate.abstract)),
        len(candidate.abstract or ""),
        len(candidate.authors),
        int(candidate.is_open_access),
    )

    return candidate if candidate_score > current_score else current


def _namespaced_query_id(
    original_query_id: str,
    pilot_index: int,
    used_query_ids: set[str],
) -> str:
    """Return a deterministic unused ID for a conflicting execution.

    Context-expanded retrievals can intentionally execute the same planned
    query with different retrieval filters, for example OPEN_ACCESS:Y in one
    pilot and no open-access restriction in another.  Those executions must
    both be retained, but they cannot share a query ID in the merged result.
    """

    base = f"{original_query_id}__input_{pilot_index}"
    candidate = base
    suffix = 2

    while candidate in used_query_ids:
        candidate = f"{base}_{suffix}"
        suffix += 1

    return candidate


def merge_retrieval_pilots(
    pilots: list[RetrievalPilotResult],
) -> RetrievalPilotResult:
    """Merge baseline and context retrieval without duplicating papers."""

    if len(pilots) < 2:
        raise ValueError("At least two retrieval pilots are required to merge.")

    reference_request = pilots[0].request
    reference_signature = _request_signature(reference_request)

    for index, pilot in enumerate(pilots[1:], start=2):
        if _request_signature(pilot.request) != reference_signature:
            raise ValueError(
                "Retrieval inputs are not comparable. Request mismatch at "
                f"input {index}."
            )

    queries_by_id: dict[str, LiteratureQuery] = {}
    results_by_id: dict[str, QueryRetrievalResult] = {}
    papers_by_key: dict[str, PublicationRecord] = {}
    warnings: list[str] = []

    for pilot_index, pilot in enumerate(pilots, start=1):
        assigned_ids: dict[str, str] = {}

        # Query results are processed first because the executed query records
        # retrieval-time filters that may not appear in the planned query.
        for query_result in pilot.query_results:
            original_id = query_result.query.query_id
            assigned_id = original_id
            existing_result = results_by_id.get(original_id)

            if existing_result is not None:
                same_execution = (
                    existing_result.query.query == query_result.query.query
                    and existing_result.executed_query
                    == query_result.executed_query
                )

                if same_execution:
                    assigned_ids[original_id] = original_id
                    continue

                assigned_id = _namespaced_query_id(
                    original_query_id=original_id,
                    pilot_index=pilot_index,
                    used_query_ids=set(queries_by_id) | set(results_by_id),
                )

            assigned_ids[original_id] = assigned_id
            assigned_query = query_result.query.model_copy(
                update={"query_id": assigned_id}
            )
            assigned_result = query_result.model_copy(
                update={"query": assigned_query}
            )
            queries_by_id[assigned_id] = assigned_query
            results_by_id[assigned_id] = assigned_result

        # Preserve planned queries that have no result while keeping their IDs
        # consistent with any result processed above.
        for query in pilot.query_plan.queries:
            assigned_id = assigned_ids.get(query.query_id)

            if assigned_id is None:
                assigned_id = query.query_id
                existing_query = queries_by_id.get(assigned_id)

                if (
                    existing_query is not None
                    and existing_query.query != query.query
                ):
                    assigned_id = _namespaced_query_id(
                        original_query_id=query.query_id,
                        pilot_index=pilot_index,
                        used_query_ids=(
                            set(queries_by_id) | set(results_by_id)
                        ),
                    )

            if assigned_id not in queries_by_id:
                queries_by_id[assigned_id] = query.model_copy(
                    update={"query_id": assigned_id}
                )

        for paper in pilot.unique_papers:
            existing_paper = papers_by_key.get(paper.record_key)

            if existing_paper is None:
                papers_by_key[paper.record_key] = paper
            else:
                papers_by_key[paper.record_key] = _prefer_richer_record(
                    existing_paper,
                    paper,
                )

        warnings.extend(
            f"input_{pilot_index}: {warning}" for warning in pilot.warnings
        )

    queries = list(queries_by_id.values())
    query_results = [
        results_by_id[query.query_id]
        for query in queries
        if query.query_id in results_by_id
    ]
    unique_papers = sorted(
        papers_by_key.values(),
        key=lambda paper: (
            -(paper.publication_year or 0),
            paper.title.casefold(),
        ),
    )

    return RetrievalPilotResult(
        schema_name="LiteratureRetrievalPilot",
        schema_version="2.1-merged",
        request=reference_request,
        query_plan=LiteratureQueryPlan(
            schema_name="LiteratureQueryPlan",
            schema_version="2.1-merged",
            cluster_id=reference_request.cluster_id,
            queries=queries,
        ),
        query_results=query_results,
        unique_papers=unique_papers,
        warnings=warnings,
    )
