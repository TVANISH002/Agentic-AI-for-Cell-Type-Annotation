from __future__ import annotations

import json
from pathlib import Path

from literature_retrieval_agent.schemas import (
    RetrievalPilotResult,
)


def _write_json(
    path: Path,
    payload: object,
) -> None:
    """Write one JSON file atomically."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )

        handle.write("\n")

    temporary_path.replace(path)


def write_pilot_reports(
    result: RetrievalPilotResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write auditable pilot output files."""

    directory = (
        Path(output_dir)
        .expanduser()
        .resolve()
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = (
        f"cluster_{result.request.cluster_id}"
    )

    paths = {
        "request": (
            directory
            / f"{prefix}_literature_request.json"
        ),
        "query_plan": (
            directory
            / f"{prefix}_query_plan.json"
        ),
        "retrieved_papers": (
            directory
            / f"{prefix}_retrieved_papers.json"
        ),
        "pilot_result": (
            directory
            / f"{prefix}_retrieval_pilot.json"
        ),
    }

    _write_json(
        paths["request"],
        result.request.model_dump(
            mode="json"
        ),
    )

    _write_json(
        paths["query_plan"],
        result.query_plan.model_dump(
            mode="json"
        ),
    )

    _write_json(
        paths["retrieved_papers"],
        [
            paper.model_dump(mode="json")
            for paper in result.unique_papers
        ],
    )

    _write_json(
        paths["pilot_result"],
        result.model_dump(
            mode="json"
        ),
    )

    return paths