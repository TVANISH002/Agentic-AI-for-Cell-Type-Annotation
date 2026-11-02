from __future__ import annotations

from literature_retrieval_agent.config import (
    DEFAULT_CONFIG,
    LiteratureRetrievalConfig,
)
from literature_retrieval_agent.europe_pmc_client import (
    EuropePMCClient,
)
from literature_retrieval_agent.query_builder import (
    LiteratureQueryBuilder,
)
from literature_retrieval_agent.schemas import (
    LiteratureQueryPlan,
    LiteratureRequest,
    PublicationRecord,
    QueryRetrievalResult,
    RetrievalPilotResult,
)


class LiteratureRetrievalAgent:
    """
    Plan and retrieve citation-grounded evidence.

    This agent does not assign the final cell type.
    """

    def __init__(
        self,
        config: LiteratureRetrievalConfig = DEFAULT_CONFIG,
        client: EuropePMCClient | None = None,
    ) -> None:
        self.config = config

        self.query_builder = (
            LiteratureQueryBuilder(config)
        )

        self.client = (
            client
            if client is not None
            else EuropePMCClient(config)
        )

        self._owns_client = client is None

    def close(self) -> None:
        """Close the internally created API client."""

        if self._owns_client:
            self.client.close()

    def build_query_plan(
        self,
        request: LiteratureRequest,
    ) -> LiteratureQueryPlan:
        """Build queries without calling Europe PMC."""

        return self.query_builder.build(
            request
        )

    def run_pilot(
        self,
        request: LiteratureRequest,
    ) -> RetrievalPilotResult:
        """Run deterministic literature retrieval."""

        query_plan = self.build_query_plan(
            request
        )

        query_results: list[
            QueryRetrievalResult
        ] = []

        warnings: list[str] = []

        try:
            for query in query_plan.queries:
                try:
                    result = self.client.search(
                        query
                    )

                    query_results.append(
                        result
                    )

                except RuntimeError as exc:
                    message = (
                        f"{query.query_id} failed: "
                        f"{exc}"
                    )

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

        unique_by_key: dict[
            str,
            PublicationRecord,
        ] = {}

        for result in query_results:
            for paper in result.papers:
                unique_by_key.setdefault(
                    paper.record_key,
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
                "No publications were retrieved "
                "for this request."
            )

        return RetrievalPilotResult(
            schema_name=(
                self.config.retrieval_schema_name
            ),
            schema_version=(
                self.config.retrieval_schema_version
            ),
            request=request,
            query_plan=query_plan,
            query_results=query_results,
            unique_papers=unique_papers,
            warnings=warnings,
        )