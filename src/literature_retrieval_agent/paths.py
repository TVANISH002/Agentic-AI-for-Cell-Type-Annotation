from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

AGENT2_DATA_DIR = (
    DATA_DIR
    / "agent2_v2"
    / "clusters"
)

AGENT3_DATA_DIR = (
    DATA_DIR
    / "literature_retrieval_agent"
)

AGENT3_CLUSTER_DIR = (
    AGENT3_DATA_DIR
    / "clusters"
)


def agent2_cluster_json(
    cluster_id: str | int,
) -> Path:
    """Return the Agent 2 JSON path for one cluster."""

    cluster = str(cluster_id).strip()

    if not cluster:
        raise ValueError(
            "cluster_id cannot be empty."
        )

    return (
        AGENT2_DATA_DIR
        / f"cluster_{cluster}"
        / f"cluster_{cluster}_marker_evidence.json"
    )


def agent3_cluster_output_dir(
    cluster_id: str | int,
) -> Path:
    """Return the Agent 3 output directory for one cluster."""

    cluster = str(cluster_id).strip()

    if not cluster:
        raise ValueError(
            "cluster_id cannot be empty."
        )

    return (
        AGENT3_CLUSTER_DIR
        / f"cluster_{cluster}"
    )