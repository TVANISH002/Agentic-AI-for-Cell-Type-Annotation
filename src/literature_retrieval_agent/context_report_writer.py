from __future__ import annotations

import json
from pathlib import Path

from literature_retrieval_agent.config import LiteratureRetrievalConfig
from literature_retrieval_agent.context_expansion import (
    ContextExpansionTerms,
    ContextQueryAudit,
    ContextQueryConfig,
)
from literature_retrieval_agent.schemas import (
    LiteratureQueryPlan,
    LiteratureRequest,
    RetrievalPilotResult,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    temporary_path.replace(path)


def write_context_plan_reports(
    *,
    request: LiteratureRequest,
    query_plan: LiteratureQueryPlan,
    audit: ContextQueryAudit,
    expansion_terms: ContextExpansionTerms,
    query_config: ContextQueryConfig,
    retrieval_config: LiteratureRetrievalConfig,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Write dry-run proof without calling a model or Europe PMC."""

    directory = Path(output_directory).expanduser().resolve()
    prefix = f"cluster_{request.cluster_id}_context_v2"

    paths = {
        "request": directory / f"{prefix}_literature_request.json",
        "query_plan": directory / f"{prefix}_query_plan.json",
        "query_audit": directory / f"{prefix}_query_audit.json",
        "run_configuration": directory / f"{prefix}_configuration.json",
    }

    _write_json(paths["request"], request.model_dump(mode="json"))
    _write_json(paths["query_plan"], query_plan.model_dump(mode="json"))
    _write_json(paths["query_audit"], audit.model_dump(mode="json"))
    _write_json(
        paths["run_configuration"],
        {
            "strategy": query_config.to_dict(),
            "retrieval": retrieval_config.to_dict(),
            "context_expansion_terms": expansion_terms.model_dump(mode="json"),
        },
    )

    return paths


def write_context_retrieval_reports(
    *,
    result: RetrievalPilotResult,
    audit: ContextQueryAudit,
    expansion_terms: ContextExpansionTerms,
    query_config: ContextQueryConfig,
    retrieval_config: LiteratureRetrievalConfig,
    output_directory: str | Path,
) -> dict[str, Path]:
    paths = write_context_plan_reports(
        request=result.request,
        query_plan=result.query_plan,
        audit=audit,
        expansion_terms=expansion_terms,
        query_config=query_config,
        retrieval_config=retrieval_config,
        output_directory=output_directory,
    )

    directory = Path(output_directory).expanduser().resolve()
    prefix = f"cluster_{result.request.cluster_id}_context_v2"
    paths.update(
        {
            "retrieved_papers": directory
            / f"{prefix}_retrieved_papers.json",
            "pilot_result": directory / f"{prefix}_retrieval_pilot.json",
        }
    )

    _write_json(
        paths["retrieved_papers"],
        [paper.model_dump(mode="json") for paper in result.unique_papers],
    )
    _write_json(paths["pilot_result"], result.model_dump(mode="json"))

    return paths
