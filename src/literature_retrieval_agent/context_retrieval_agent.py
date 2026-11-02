from __future__ import annotations

from literature_retrieval_agent.context_expansion import ContextQueryConfig
from literature_retrieval_agent.europe_pmc_client import EuropePMCClient
from literature_retrieval_agent.schemas import (
    LiteratureQueryPlan,
    LiteratureRequest,
    PublicationRecord,
    QueryRetrievalResult,
    RetrievalPilotResult,
)


def _prefer_richer_record(
    current: PublicationRecord,
    candidate: PublicationRecord,
) -> PublicationRecord:
    """Keep the version with an abstract and richer citation metadata."""

    current_score = (
        int(bool(current.abstract)),
        len(current.abstract or ""),
        len(current.authors),
        int(bool(current.doi)),
    )
    candidate_score = (
        int(bool(candidate.abstract)),
        len(candidate.abstract or ""),
        len(candidate.authors),
        int(bool(candidate.doi)),
    )

    return candidate if candidate_score > current_score else current


class ContextExpandedRetrievalAgent:
    """Execute a pre-audited context-expanded plan against Europe PMC."""

    def __init__(
        self,
        client: EuropePMCClient,
        query_config: ContextQueryConfig | None = None,
        *,
        owns_client: bool = True,
    ) -> None:
        self.client = client
        self.query_config = query_config or ContextQueryConfig()
        self.owns_client = owns_client

    def close(self) -> None:
        if self.owns_client:
            self.client.close()

    def run(
        self,
        request: LiteratureRequest,
        query_plan: LiteratureQueryPlan,
    ) -> RetrievalPilotResult:
        if str(query_plan.cluster_id) != str(request.cluster_id):
            raise ValueError(
                "Query-plan cluster does not match LiteratureRequest cluster."
            )

        query_results: list[QueryRetrievalResult] = []
        warnings: list[str] = []

        try:
            for query in query_plan.queries:
                try:
                    query_results.append(self.client.search(query))
                except RuntimeError as exc:
                    message = f"{query.query_id} failed: {exc}"
                    warnings.append(message)
                    query_results.append(
                        QueryRetrievalResult(
                            query=query,
                            executed_query=query.query,
                            hit_count=0,
                            returned_count=0,
                            papers=[],
                            warnings=[message],
                        )
                    )
        finally:
            self.close()

        unique_by_key: dict[str, PublicationRecord] = {}

        for query_result in query_results:
            for paper in query_result.papers:
                existing = unique_by_key.get(paper.record_key)

                if existing is None:
                    unique_by_key[paper.record_key] = paper
                else:
                    unique_by_key[paper.record_key] = _prefer_richer_record(
                        existing,
                        paper,
                    )

        unique_papers = sorted(
            unique_by_key.values(),
            key=lambda paper: (
                -(paper.publication_year or 0),
                paper.title.casefold(),
            ),
        )

        if not unique_papers:
            warnings.append(
                "No publications were retrieved by context-expanded v2."
            )

        return RetrievalPilotResult(
            schema_name="LiteratureRetrievalPilot",
            schema_version=self.query_config.retrieval_schema_version,
            request=request,
            query_plan=query_plan,
            query_results=query_results,
            unique_papers=unique_papers,
            warnings=warnings,
        )
