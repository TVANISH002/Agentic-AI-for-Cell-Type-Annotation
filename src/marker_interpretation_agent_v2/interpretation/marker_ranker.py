from __future__ import annotations

from collections.abc import Sequence

from marker_interpretation_agent_v2.schemas import (
    GeneInterpretation,
    RankedMarkerGene,
)


class MarkerRanker:
    """
    Rank cluster marker genes using direct breast-context evidence.

    Ranking priority:

    1. More supporting databases
    2. Greater marker specificity
    3. Better original DEG rank
    4. Higher log-fold change
    5. Gene symbol for deterministic tie-breaking

    HPA-only genes are excluded because HPA expression does not
    independently establish a curated marker relationship.
    """

    @staticmethod
    def _normalize_gene(value: str) -> str:
        """Normalize one gene symbol."""

        gene = str(value).strip().upper()

        if not gene:
            raise ValueError("Gene symbol cannot be empty.")

        return gene

    @staticmethod
    def _sort_key(
        interpretation: GeneInterpretation,
    ) -> tuple[int, float, int, float, str]:
        """Return the deterministic marker-ranking key."""

        statistics = interpretation.statistics

        log_fold_change_key = (
            -statistics.log_fold_change
            if statistics.log_fold_change is not None
            else float("inf")
        )

        return (
            -statistics.source_agreement_count,
            -statistics.marker_specificity_score,
            statistics.cluster_gene_rank,
            log_fold_change_key,
            interpretation.gene,
        )

    @staticmethod
    def _ranking_reason(
        interpretation: GeneInterpretation,
    ) -> str:
        """Create a concise explanation of the marker rank."""

        statistics = interpretation.statistics

        source_count = (
            statistics.source_agreement_count
        )

        cell_type_count = len(
            set(interpretation.supported_cell_types)
        )

        if source_count >= 2:
            source_text = (
                f"supported by {source_count} marker databases"
            )
        elif source_count == 1:
            source_text = (
                "supported by 1 marker database"
            )
        else:
            source_text = (
                "supported by context-matched marker evidence"
            )

        if cell_type_count == 1:
            specificity_text = (
                "specific to 1 supported cell type"
            )
        else:
            specificity_text = (
                f"distributed across {cell_type_count} "
                "supported cell types"
            )

        log_fold_change = (
            statistics.log_fold_change
        )

        if log_fold_change is None:
            fold_change_text = (
                "log-fold change unavailable"
            )
        else:
            fold_change_text = (
                f"log-fold change {log_fold_change:.4f}"
            )

        return (
            f"{source_text}; {specificity_text}; "
            f"original DEG rank "
            f"{statistics.cluster_gene_rank}; "
            f"{fold_change_text}."
        )

    def rank(
        self,
        interpretations: Sequence[
            GeneInterpretation
        ],
    ) -> list[RankedMarkerGene]:
        """Rank all direct-marker-supported genes."""

        gene_names = [
            self._normalize_gene(
                interpretation.gene
            )
            for interpretation in interpretations
        ]

        duplicate_genes = sorted(
            {
                gene
                for gene in gene_names
                if gene_names.count(gene) > 1
            }
        )

        if duplicate_genes:
            raise ValueError(
                "Gene interpretations contain "
                f"duplicate genes: {duplicate_genes}"
            )

        direct_markers = [
            interpretation
            for interpretation in interpretations
            if interpretation.direct_marker_supported
        ]

        for interpretation in direct_markers:
            if (
                interpretation.evidence_category
                != "direct_marker_supported"
            ):
                raise ValueError(
                    "A direct marker gene must use the "
                    "direct_marker_supported category. "
                    f"Gene: {interpretation.gene}"
                )

            if not interpretation.supported_cell_types:
                raise ValueError(
                    "A direct marker gene must contain "
                    "at least one supported cell type. "
                    f"Gene: {interpretation.gene}"
                )

        sorted_markers = sorted(
            direct_markers,
            key=self._sort_key,
        )

        ranked_markers: list[
            RankedMarkerGene
        ] = []

        for marker_rank, interpretation in enumerate(
            sorted_markers,
            start=1,
        ):
            statistics = (
                interpretation.statistics
            )

            ranked_markers.append(
                RankedMarkerGene(
                    marker_rank=marker_rank,
                    gene=interpretation.gene,
                    source_agreement_count=(
                        statistics.source_agreement_count
                    ),
                    marker_specificity_score=(
                        statistics.marker_specificity_score
                    ),
                    context_evidence_count=(
                        statistics.context_evidence_count
                    ),
                    cluster_gene_rank=(
                        statistics.cluster_gene_rank
                    ),
                    log_fold_change=(
                        statistics.log_fold_change
                    ),
                    supported_cell_types=list(
                        interpretation.supported_cell_types
                    ),
                    evidence_sources=list(
                        interpretation.evidence_sources
                    ),
                    ranking_reason=(
                        self._ranking_reason(
                            interpretation
                        )
                    ),
                )
            )

        return ranked_markers