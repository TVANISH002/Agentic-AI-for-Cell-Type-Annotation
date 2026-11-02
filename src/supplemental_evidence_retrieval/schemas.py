
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


Agent1EvidenceStatus = Literal[
    "database_evidence_without_context",
    "unresolved_in_agent1",
]


RetrievalStatus = Literal[
    "pending",
    "in_progress",
    "completed",
    "failed",
    "not_required",
]


class SupplementalEvidenceQueueSummary(
    StrictSchema
):
    total_input_genes: int = Field(
        ge=0
    )
    context_supported_gene_count: int = Field(
        ge=0
    )
    context_missing_gene_count: int = Field(
        ge=0
    )
    database_evidence_without_context_count: int = Field(
        ge=0
    )
    agent1_unresolved_gene_count: int = Field(
        ge=0
    )
    pilot_gene_count: int = Field(
        ge=0
    )


class SupplementalEvidenceQueueItem(
    StrictSchema
):
    priority: int = Field(
        gt=0
    )
    gene: str
    rank_within_cluster: int = Field(
        gt=0
    )
    score: float
    log_fold_change: float
    agent1_database_evidence_found: bool
    agent1_unresolved: bool
    agent1_evidence_status: Agent1EvidenceStatus
    queue_reason: str
    pilot_selected: bool = False
    retrieval_status: RetrievalStatus = (
        "pending"
    )


class SupplementalEvidenceQueue(
    StrictSchema
):
    cluster_id: int | str
    species: str
    tissue: str
    platform: str

    queue_stage: Literal[
        "supplemental_evidence_queue"
    ] = "supplemental_evidence_queue"

    summary: SupplementalEvidenceQueueSummary

    pilot_genes: list[str] = Field(
        default_factory=list
    )

    items: list[
        SupplementalEvidenceQueueItem
    ] = Field(
        default_factory=list
    )

    warnings: list[str] = Field(
        default_factory=list
    )




