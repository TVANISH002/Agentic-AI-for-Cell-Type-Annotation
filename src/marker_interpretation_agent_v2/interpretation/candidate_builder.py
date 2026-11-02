from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import mean
from typing import Any
from ..schemas import CandidateCellType, GeneInterpretation

from marker_interpretation_agent_v2.schemas import (
    CandidateCellType,
    GeneInterpretation,
)


@dataclass(slots=True)
class _CandidateAggregate:
    """Internal accumulator for one candidate cell type."""

    display_name: str

    supporting_genes: set[str] = field(
        default_factory=set
    )

    both_source_genes: set[str] = field(
        default_factory=set
    )

    specificity_scores: dict[str, float] = field(
        default_factory=dict
    )

    gene_ranks: dict[str, int] = field(
        default_factory=dict
    )

    log_fold_changes: dict[str, float] = field(
        default_factory=dict
    )

    evidence_count: int = 0

    evidence_sources: set[str] = field(
        default_factory=set
    )

    tissue_contexts: set[str] = field(
        default_factory=set
    )

    disease_contexts: set[str] = field(
        default_factory=set
    )

    cell_ontology_ids: set[str] = field(
        default_factory=set
    )

    hpa_queried_genes: set[str] = field(
        default_factory=set
    )

    hpa_breast_expression_genes: set[str] = field(
        default_factory=set
    )

    hpa_top_cell_types: set[str] = field(
        default_factory=set
    )


class CandidateBuilder:
    """
    Build cluster-level candidate cell types.

    Only direct breast-context marker genes contribute candidate votes.
    HPA evidence is summarized for those genes but does not independently
    create or increase candidate marker support.
    """

    @staticmethod
    def _normalize_gene(value: str) -> str:
        """Normalize one gene symbol."""

        gene = str(value).strip().upper()

        if not gene:
            raise ValueError(
                "Gene symbol cannot be empty."
            )

        return gene

    @staticmethod
    def _normalize_cell_type(
        value: str,
    ) -> str:
        """Normalize whitespace in one cell-type label."""

        cell_type = " ".join(
            str(value).strip().split()
        )

        if not cell_type:
            raise ValueError(
                "Cell-type label cannot be empty."
            )

        return cell_type

    @classmethod
    def _candidate_key(
        cls,
        value: str,
    ) -> str:
        """Create a case-insensitive candidate key."""

        return cls._normalize_cell_type(
            value
        ).casefold()

    @staticmethod
    def _split_values(
        value: Any,
    ) -> list[str]:
        """Convert a value into unique clean strings."""

        if value is None:
            return []

        if isinstance(value, str):
            raw_values: Sequence[Any] = (
                value.replace(",", "|").split("|")
            )
        elif (
            isinstance(value, Sequence)
            and not isinstance(
                value,
                (
                    bytes,
                    bytearray,
                ),
            )
        ):
            raw_values = value
        else:
            raw_values = [value]

        return sorted(
            {
                str(item).strip()
                for item in raw_values
                if (
                    item is not None
                    and str(item).strip()
                )
            },
            key=str.casefold,
        )

    @classmethod
    def _record_cell_type(
        cls,
        record: Mapping[str, Any],
    ) -> str | None:
        """Extract a cell-type label from one evidence record."""

        value = (
            record.get(
                "normalized_cell_type"
            )
            or record.get("cell_type")
            or record.get("cell_name")
        )

        if (
            value is None
            or not str(value).strip()
        ):
            return None

        return cls._normalize_cell_type(
            str(value)
        )

    @classmethod
    def _matching_records(
        cls,
        interpretation: GeneInterpretation,
        cell_type: str,
    ) -> list[dict[str, Any]]:
        """Return context records matching one candidate."""

        target_key = cls._candidate_key(
            cell_type
        )

        matches: list[
            dict[str, Any]
        ] = []

        for record in interpretation.context_evidence:
            record_cell_type = (
                cls._record_cell_type(
                    record
                )
            )

            if record_cell_type is None:
                continue

            if (
                cls._candidate_key(
                    record_cell_type
                )
                == target_key
            ):
                matches.append(
                    dict(record)
                )

        return matches

    @classmethod
    def _add_record_metadata(
        cls,
        aggregate: _CandidateAggregate,
        record: Mapping[str, Any],
    ) -> None:
        """Add biological metadata from one evidence record."""

        source = (
            record.get("source")
            or record.get("database")
        )

        if (
            source is not None
            and str(source).strip()
        ):
            aggregate.evidence_sources.add(
                str(source).strip()
            )

        for key in (
            "tissue",
            "tissue_type",
            "tissue_context",
        ):
            aggregate.tissue_contexts.update(
                cls._split_values(
                    record.get(key)
                )
            )

        for key in (
            "disease",
            "disease_context",
            "cancer_type",
        ):
            aggregate.disease_contexts.update(
                cls._split_values(
                    record.get(key)
                )
            )

        for key in (
            "cell_ontology_id",
            "cell_ontology_ids",
            "cl_id",
        ):
            aggregate.cell_ontology_ids.update(
                cls._split_values(
                    record.get(key)
                )
            )

    @classmethod
    def _add_gene_support(
        cls,
        aggregate: _CandidateAggregate,
        interpretation: GeneInterpretation,
        matching_records: Sequence[
            Mapping[str, Any]
        ],
    ) -> None:
        """Add one direct marker gene to a candidate."""

        gene = cls._normalize_gene(
            interpretation.gene
        )

        statistics = (
            interpretation.statistics
        )

        aggregate.supporting_genes.add(
            gene
        )

        aggregate.specificity_scores[
            gene
        ] = statistics.marker_specificity_score

        aggregate.gene_ranks[
            gene
        ] = statistics.cluster_gene_rank

        if (
            statistics.log_fold_change
            is not None
        ):
            aggregate.log_fold_changes[
                gene
            ] = statistics.log_fold_change

        gene_sources = {
            str(source).strip()
            for source
            in interpretation.evidence_sources
            if str(source).strip()
        }

        aggregate.evidence_sources.update(
            gene_sources
        )

        if len(gene_sources) >= 2:
            aggregate.both_source_genes.add(
                gene
            )

        for record in matching_records:
            aggregate.evidence_count += 1

            cls._add_record_metadata(
                aggregate,
                record,
            )

        if interpretation.hpa.queried:
            aggregate.hpa_queried_genes.add(
                gene
            )

        if (
            interpretation.hpa
            .breast_expression_found
        ):
            aggregate.hpa_breast_expression_genes.add(
                gene
            )

        if interpretation.hpa.top_cell_type:
            aggregate.hpa_top_cell_types.add(
                interpretation.hpa.top_cell_type
            )

    @staticmethod
    def _mean_log_fold_change(
        aggregate: _CandidateAggregate,
    ) -> float | None:
        """Calculate mean available log-fold change."""

        if not aggregate.log_fold_changes:
            return None

        return mean(
            aggregate.log_fold_changes.values()
        )

    @staticmethod
    def _candidate_warnings(
        aggregate: _CandidateAggregate,
    ) -> list[str]:
        """Create candidate-level warnings."""

        warnings: list[str] = []

        if (
            len(aggregate.supporting_genes)
            == 1
        ):
            warnings.append(
                "Candidate is supported by only "
                "one direct-marker gene."
            )

        if aggregate.evidence_count == 0:
            warnings.append(
                "Candidate has a supported label, but "
                "no matching context evidence records "
                "were available for metadata aggregation."
            )

        return warnings

    @classmethod
    def _to_candidate(
        cls,
        aggregate: _CandidateAggregate,
    ) -> CandidateCellType:
        """Convert one aggregate into a candidate schema."""

        supporting_genes = sorted(
            aggregate.supporting_genes
        )

        both_source_genes = sorted(
            aggregate.both_source_genes
        )

        return CandidateCellType(
            candidate_rank=1,
            cell_type=aggregate.display_name,
            unique_supporting_gene_count=len(
                supporting_genes
            ),
            supporting_genes=(
                supporting_genes
            ),
            both_source_supporting_gene_count=len(
                both_source_genes
            ),
            both_source_supporting_genes=(
                both_source_genes
            ),
            specificity_weighted_support=sum(
                aggregate.specificity_scores.values()
            ),
            mean_supporting_gene_rank=mean(
                aggregate.gene_ranks.values()
            ),
            mean_supporting_log_fold_change=(
                cls._mean_log_fold_change(
                    aggregate
                )
            ),
            evidence_count=(
                aggregate.evidence_count
            ),
            evidence_sources=sorted(
                aggregate.evidence_sources,
                key=str.casefold,
            ),
            tissue_contexts=sorted(
                aggregate.tissue_contexts,
                key=str.casefold,
            ),
            disease_contexts=sorted(
                aggregate.disease_contexts,
                key=str.casefold,
            ),
            cell_ontology_ids=sorted(
                aggregate.cell_ontology_ids,
                key=str.casefold,
            ),
            hpa_queried_gene_count=len(
                aggregate.hpa_queried_genes
            ),
            hpa_breast_expression_gene_count=len(
                aggregate.hpa_breast_expression_genes
            ),
            hpa_top_cell_types=sorted(
                aggregate.hpa_top_cell_types,
                key=str.casefold,
            ),
            warnings=(
                cls._candidate_warnings(
                    aggregate
                )
            ),
        )

    @staticmethod
    def _sort_key(
        candidate: CandidateCellType,
    ) -> tuple[
        int,
        int,
        float,
        float,
        float,
        int,
        str,
    ]:
        """Return the deterministic candidate ranking key."""

        log_fold_change_key = (
            -candidate.mean_supporting_log_fold_change
            if (
                candidate.mean_supporting_log_fold_change
                is not None
            )
            else float("inf")
        )

        return (
            -candidate.unique_supporting_gene_count,
            -candidate.both_source_supporting_gene_count,
            -candidate.specificity_weighted_support,
            candidate.mean_supporting_gene_rank,
            log_fold_change_key,
            -candidate.evidence_count,
            candidate.cell_type.casefold(),
        )

    def build(
        self,
        interpretations: Sequence[
            GeneInterpretation
        ],
    ) -> list[CandidateCellType]:
        """Build and rank candidates from direct markers."""

        gene_names = [
            self._normalize_gene(
                interpretation.gene
            )
            for interpretation
            in interpretations
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
                "duplicate genes: "
                f"{duplicate_genes}"
            )

        aggregates: dict[
            str,
            _CandidateAggregate,
        ] = {}

        for interpretation in interpretations:
            if (
                not interpretation
                .direct_marker_supported
            ):
                continue

            if (
                interpretation.evidence_category
                != "direct_marker_supported"
            ):
                raise ValueError(
                    "Direct marker genes must use "
                    "the direct_marker_supported "
                    "category. "
                    f"Gene: {interpretation.gene}"
                )

            for cell_type in (
                interpretation
                .supported_cell_types
            ):
                display_name = (
                    self._normalize_cell_type(
                        cell_type
                    )
                )

                candidate_key = (
                    self._candidate_key(
                        display_name
                    )
                )

                if candidate_key not in aggregates:
                    aggregates[
                        candidate_key
                    ] = _CandidateAggregate(
                        display_name=display_name
                    )

                aggregate = aggregates[
                    candidate_key
                ]

                matching_records = (
                    self._matching_records(
                        interpretation,
                        display_name,
                    )
                )

                self._add_gene_support(
                    aggregate,
                    interpretation,
                    matching_records,
                )

        candidates = sorted(
            [
                self._to_candidate(
                    aggregate
                )
                for aggregate
                in aggregates.values()
            ],
            key=self._sort_key,
        )

        ranked_candidates: list[
            CandidateCellType
        ] = []

        for rank, candidate in enumerate(
            candidates,
            start=1,
        ):
            ranked_candidates.append(
                CandidateCellType.model_validate(
                    {
                        **candidate.model_dump(),
                        "candidate_rank": rank,
                    }
                )
            )

        return ranked_candidates