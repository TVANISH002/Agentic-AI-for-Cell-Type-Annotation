from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import (
    Agent1EvidenceStatus,
    SupplementalEvidenceQueue,
    SupplementalEvidenceQueueItem,
    SupplementalEvidenceQueueSummary,
)


class SupplementalEvidenceQueueBuilder:
    def __init__(
        self,
        pilot_genes: list[str] | None = None,
    ) -> None:
        self.pilot_genes = {
            self._gene_key(gene)
            for gene in (
                pilot_genes or []
            )
        }

    @staticmethod
    def _gene_key(
        value: Any,
    ) -> str:
        return str(
            value
        ).strip().upper()

    @staticmethod
    def _load_json(
        path: Path | str,
    ) -> dict[str, Any]:
        source = Path(
            path
        ).resolve()

        if not source.exists():
            raise FileNotFoundError(
                f"Agent 1 JSON not found: {source}"
            )

        with source.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = json.load(
                handle
            )

        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                "Agent 1 JSON must contain one top-level object."
            )

        return data

    @staticmethod
    def _validate_agent1(
        data: dict[str, Any],
    ) -> None:
        required = {
            "cluster_id",
            "species",
            "tissue",
            "platform",
            "input_genes",
            "gene_evidence",
        }

        missing = required - set(
            data
        )

        if missing:
            raise ValueError(
                "Agent 1 JSON is missing required keys: "
                f"{sorted(missing)}"
            )

        if len(
            data["input_genes"]
        ) != len(
            data["gene_evidence"]
        ):
            raise ValueError(
                "Agent 1 input_genes and gene_evidence counts differ."
            )

        input_genes = [
            str(
                item["gene"]
            ).strip().upper()
            for item in data[
                "input_genes"
            ]
        ]

        evidence_genes = [
            str(
                item["gene"]
            ).strip().upper()
            for item in data[
                "gene_evidence"
            ]
        ]

        if input_genes != evidence_genes:
            raise ValueError(
                "Agent 1 input gene order does not match gene evidence order."
            )

        if len(
            input_genes
        ) != len(
            set(input_genes)
        ):
            raise ValueError(
                "Agent 1 contains duplicate genes."
            )

    @staticmethod
    def _statistics(
        input_gene: dict[str, Any],
    ) -> tuple[int, float, float]:
        stats = input_gene.get(
            "statistics",
            {},
        )

        return (
            int(
                stats[
                    "rank_within_cluster"
                ]
            ),
            float(
                stats["score"]
            ),
            float(
                stats[
                    "log_fold_change"
                ]
            ),
        )

    def build_from_dict(
        self,
        agent1: dict[str, Any],
    ) -> SupplementalEvidenceQueue:
        self._validate_agent1(
            agent1
        )

        evidence_map = {
            self._gene_key(
                item["gene"]
            ): item
            for item in agent1[
                "gene_evidence"
            ]
        }

        items: list[
            SupplementalEvidenceQueueItem
        ] = []

        context_supported_count = 0

        for input_gene in agent1[
            "input_genes"
        ]:
            gene = self._gene_key(
                input_gene["gene"]
            )

            evidence = evidence_map[
                gene
            ]

            context_cell_types = list(
                evidence.get(
                    "context_matched_cell_types",
                    [],
                )
            )

            database_evidence = list(
                evidence.get(
                    "database_evidence",
                    [],
                )
            )

            if context_cell_types:
                context_supported_count += 1
                continue

            database_found = bool(
                database_evidence
            )

            (
                rank,
                score,
                log_fold_change,
            ) = self._statistics(
                input_gene
            )

            status: Agent1EvidenceStatus

            if database_found:
                status = (
                    "database_evidence_without_context"
                )

                reason = (
                    "Agent 1 found database evidence, "
                    "but no breast-context cell-type evidence."
                )

            else:
                status = (
                    "unresolved_in_agent1"
                )

                reason = (
                    "Agent 1 found no database evidence "
                    "for this gene."
                )

            items.append(
                SupplementalEvidenceQueueItem(
                    priority=len(
                        items
                    )
                    + 1,
                    gene=gene,
                    rank_within_cluster=rank,
                    score=score,
                    log_fold_change=(
                        log_fold_change
                    ),
                    agent1_database_evidence_found=(
                        database_found
                    ),
                    agent1_unresolved=(
                        not database_found
                    ),
                    agent1_evidence_status=(
                        status
                    ),
                    queue_reason=reason,
                    pilot_selected=(
                        gene
                        in self.pilot_genes
                    ),
                    retrieval_status=(
                        "pending"
                    ),
                )
            )

        items.sort(
            key=lambda item: (
                item.rank_within_cluster,
                item.gene,
            )
        )

        for priority, item in enumerate(
            items,
            start=1,
        ):
            item.priority = priority

        database_without_context_count = sum(
            item.agent1_evidence_status
            == "database_evidence_without_context"
            for item in items
        )

        unresolved_count = sum(
            item.agent1_evidence_status
            == "unresolved_in_agent1"
            for item in items
        )

        selected_pilot_genes = [
            item.gene
            for item in items
            if item.pilot_selected
        ]

        summary = (
            SupplementalEvidenceQueueSummary(
                total_input_genes=len(
                    agent1["input_genes"]
                ),
                context_supported_gene_count=(
                    context_supported_count
                ),
                context_missing_gene_count=len(
                    items
                ),
                database_evidence_without_context_count=(
                    database_without_context_count
                ),
                agent1_unresolved_gene_count=(
                    unresolved_count
                ),
                pilot_gene_count=len(
                    selected_pilot_genes
                ),
            )
        )

        if (
            summary.total_input_genes
            != summary.context_supported_gene_count
            + summary.context_missing_gene_count
        ):
            raise ValueError(
                "Context-supported and context-missing counts "
                "do not reconstruct the Agent 1 input count."
            )

        if (
            summary.context_missing_gene_count
            != summary.database_evidence_without_context_count
            + summary.agent1_unresolved_gene_count
        ):
            raise ValueError(
                "Supplemental queue categories do not "
                "reconstruct the context-missing count."
            )

        return SupplementalEvidenceQueue(
            cluster_id=agent1[
                "cluster_id"
            ],
            species=str(
                agent1["species"]
            ),
            tissue=str(
                agent1["tissue"]
            ),
            platform=str(
                agent1["platform"]
            ),
            summary=summary,
            pilot_genes=(
                selected_pilot_genes
            ),
            items=items,
            warnings=[
                (
                    "Queue membership means breast-context evidence "
                    "is missing from the current Agent 1 sources. "
                    "It does not mean the gene is unrelated to breast biology."
                ),
                (
                    "No external biological source retrieval or RAG "
                    "has been performed at this stage."
                ),
            ],
        )

    def build_from_file(
        self,
        agent1_json: Path | str,
    ) -> SupplementalEvidenceQueue:
        return self.build_from_dict(
            self._load_json(
                agent1_json
            )
        )