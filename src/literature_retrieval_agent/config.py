from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LiteratureRetrievalConfig:
    """
    Technical settings for Agent 3.

    Biological context is supplied dynamically at runtime.
    """

    europe_pmc_endpoint: str = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    )

    result_type: str = "core"
    page_size: int = 5
    timeout_seconds: float = 30.0

    max_retries: int = 3
    retry_backoff_seconds: float = 1.0

    synonym_expansion: bool = False
    open_access_only: bool = False

    max_marker_genes_per_query: int = 6
    max_candidate_supporting_genes: int = 5

    user_agent: str = "ai-cell-annotation-agents/0.1"

    request_schema_name: str = "LiteratureRequest"
    request_schema_version: str = "1.0"

    query_plan_schema_name: str = "LiteratureQueryPlan"
    query_plan_schema_version: str = "1.0"

    retrieval_schema_name: str = "LiteratureRetrievalPilot"
    retrieval_schema_version: str = "1.0"

    def __post_init__(self) -> None:
        text_fields = (
            "europe_pmc_endpoint",
            "result_type",
            "user_agent",
            "request_schema_name",
            "request_schema_version",
            "query_plan_schema_name",
            "query_plan_schema_version",
            "retrieval_schema_name",
            "retrieval_schema_version",
        )

        for field_name in text_fields:
            value = " ".join(
                str(getattr(self, field_name)).split()
            )

            if not value:
                raise ValueError(
                    f"{field_name} cannot be empty."
                )

            object.__setattr__(
                self,
                field_name,
                value,
            )

        if not 1 <= self.page_size <= 1000:
            raise ValueError(
                "page_size must be between 1 and 1000."
            )

        if self.timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than zero."
            )

        if self.max_retries <= 0:
            raise ValueError(
                "max_retries must be greater than zero."
            )

        if self.retry_backoff_seconds < 0:
            raise ValueError(
                "retry_backoff_seconds cannot be negative."
            )

        if self.max_marker_genes_per_query <= 0:
            raise ValueError(
                "max_marker_genes_per_query must be "
                "greater than zero."
            )

        if self.max_candidate_supporting_genes <= 0:
            raise ValueError(
                "max_candidate_supporting_genes must be "
                "greater than zero."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration snapshot."""

        return asdict(self)


DEFAULT_CONFIG = LiteratureRetrievalConfig()