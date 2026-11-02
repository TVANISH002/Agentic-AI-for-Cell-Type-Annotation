from __future__ import annotations

from collections.abc import Sequence
from math import isclose

from marker_interpretation_agent_v2.config import (
    Agent2Config,
    DEFAULT_CONFIG,
)
from marker_interpretation_agent_v2.schemas import (
    CandidateCellType,
    CandidateSelection,
    GeneInterpretation,
    ReviewFlag,
)


class ReviewFlagBuilder:
    """Build transparent review and informational flags for one cluster."""

    def __init__(
        self,
        config: Agent2Config | None = None,
    ) -> None:
        """Initialize the flag builder with validated configuration."""

        self.config = config or DEFAULT_CONFIG

    @staticmethod
    def _normalize_gene(value: str) -> str:
        """Normalize one gene symbol."""

        gene = str(value).strip().upper()

        if not gene:
            raise ValueError("Gene symbol cannot be empty.")

        return gene

    @staticmethod
    def _normalize_cell_type(value: str) -> str:
        """Normalize one candidate cell-type label."""

        cell_type = " ".join(
            str(value).strip().split()
        )

        if not cell_type:
            raise ValueError(
                "Cell-type label cannot be empty."
            )

        return cell_type

    @classmethod
    def _validate_interpretations(
        cls,
        interpretations: Sequence[
            GeneInterpretation
        ],
    ) -> None:
        """Require unique gene interpretations."""

        genes = [
            cls._normalize_gene(
                interpretation.gene
            )
            for interpretation in interpretations
        ]

        duplicates = sorted(
            {
                gene
                for gene in genes
                if genes.count(gene) > 1
            }
        )

        if duplicates:
            raise ValueError(
                "Gene interpretations contain "
                f"duplicate genes: {duplicates}"
            )

    @classmethod
    def _validate_candidates(
        cls,
        candidates: Sequence[
            CandidateCellType
        ],
    ) -> None:
        """Require unique candidate labels and consecutive ranks."""

        ordered = sorted(
            candidates,
            key=lambda candidate: (
                candidate.candidate_rank
            ),
        )

        ranks = [
            candidate.candidate_rank
            for candidate in ordered
        ]

        expected_ranks = list(
            range(1, len(ordered) + 1)
        )

        if ranks != expected_ranks:
            raise ValueError(
                "Candidate ranks must be consecutive "
                "and start at 1. "
                f"Received ranks: {ranks}"
            )

        labels = [
            cls._normalize_cell_type(
                candidate.cell_type
            ).casefold()
            for candidate in ordered
        ]

        duplicates = sorted(
            {
                label
                for label in labels
                if labels.count(label) > 1
            }
        )

        if duplicates:
            raise ValueError(
                "Ranked candidates contain duplicate "
                f"labels: {duplicates}"
            )

    @staticmethod
    def _selected_candidate_labels(
        selected_candidates: Sequence[
            CandidateCellType
        ],
    ) -> set[str]:
        """Return normalized selected-candidate labels."""

        return {
            " ".join(
                candidate.cell_type.strip().split()
            ).casefold()
            for candidate in selected_candidates
        }

    @staticmethod
    def _top_candidates_tied(
        candidates: Sequence[
            CandidateCellType
        ],
    ) -> bool:
        """Return whether the two leading candidates have equal support."""

        if len(candidates) < 2:
            return False

        first = candidates[0]
        second = candidates[1]

        return (
            first.unique_supporting_gene_count
            == second.unique_supporting_gene_count
            and first.both_source_supporting_gene_count
            == second.both_source_supporting_gene_count
            and isclose(
                first.specificity_weighted_support,
                second.specificity_weighted_support,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        )

    def build(
        self,
        interpretations: Sequence[
            GeneInterpretation
        ],
        ranked_candidates: Sequence[
            CandidateCellType
        ],
        selected_candidates: Sequence[
            CandidateCellType
        ],
        candidate_selection: CandidateSelection,
    ) -> list[ReviewFlag]:
        """Build deterministic review and informational flags."""

        self._validate_interpretations(
            interpretations
        )

        self._validate_candidates(
            ranked_candidates
        )

        ranked_label_map = {
            self._normalize_cell_type(
                candidate.cell_type
            ).casefold(): candidate
            for candidate in ranked_candidates
        }

        selected_labels = (
            self._selected_candidate_labels(
                selected_candidates
            )
        )

        unknown_selected = sorted(
            selected_labels
            - set(ranked_label_map)
        )

        if unknown_selected:
            raise ValueError(
                "Selected candidates must exist in "
                "the ranked candidates. "
                f"Unknown labels: {unknown_selected}"
            )

        if (
            candidate_selection
            .selected_candidate_count
            != len(selected_candidates)
        ):
            raise ValueError(
                "CandidateSelection count does not "
                "match the selected candidate list."
            )

        selection_labels = {
            self._normalize_cell_type(
                label
            ).casefold()
            for label in (
                candidate_selection
                .selected_candidate_cell_types
            )
        }

        if selection_labels != selected_labels:
            raise ValueError(
                "CandidateSelection labels do not "
                "match the selected candidate list."
            )

        flags: list[ReviewFlag] = []

        symbol_review_genes = sorted(
            interpretation.gene
            for interpretation in interpretations
            if (
                interpretation.evidence_category
                == "symbol_review_needed"
            )
        )

        if symbol_review_genes:
            flags.append(
                ReviewFlag(
                    code="symbol_review_needed",
                    severity="warning",
                    message=(
                        "Some genes were not found in HPA "
                        "using their exact symbols and require "
                        "authoritative symbol review."
                    ),
                    genes=symbol_review_genes,
                )
            )

        hpa_unqueried_genes = sorted(
            interpretation.gene
            for interpretation in interpretations
            if not interpretation.hpa.queried
        )

        if hpa_unqueried_genes:
            flags.append(
                ReviewFlag(
                    code="hpa_not_queried",
                    severity="warning",
                    message=(
                        "HPA was not queried for every selected "
                        "cluster gene. The professor-requested "
                        "panel catalog is incomplete."
                    ),
                    genes=hpa_unqueried_genes,
                )
            )

        supplemental_genes = sorted(
            interpretation.gene
            for interpretation in interpretations
            if not interpretation.direct_marker_supported
        )

        if supplemental_genes:
            flags.append(
                ReviewFlag(
                    code=(
                        "supplemental_evidence_"
                        "not_counted_as_marker"
                    ),
                    severity="info",
                    message=(
                        "Genes without direct breast-context "
                        "marker evidence were interpreted using "
                        "supplemental evidence only and were not "
                        "counted as candidate marker votes."
                    ),
                    genes=supplemental_genes,
                )
            )

        broad_selected_labels = sorted(
            candidate.cell_type
            for candidate in selected_candidates
            if self.config.is_broad_candidate_label(
                candidate.cell_type
            )
        )

        if broad_selected_labels:
            flags.append(
                ReviewFlag(
                    code="broad_candidate_label",
                    severity="warning",
                    message=(
                        "One or more selected candidates use "
                        "broad biological labels and may require "
                        "subtype or lineage review."
                    ),
                    cell_types=broad_selected_labels,
                )
            )

        single_gene_candidates = [
            candidate
            for candidate in selected_candidates
            if (
                candidate
                .unique_supporting_gene_count
                == 1
            )
        ]

        if single_gene_candidates:
            flags.append(
                ReviewFlag(
                    code="single_gene_support",
                    severity="warning",
                    message=(
                        "One or more selected candidates are "
                        "supported by only one direct-marker gene."
                    ),
                    genes=sorted(
                        {
                            gene
                            for candidate
                            in single_gene_candidates
                            for gene
                            in candidate.supporting_genes
                        }
                    ),
                    cell_types=sorted(
                        candidate.cell_type
                        for candidate
                        in single_gene_candidates
                    ),
                )
            )

        if self._top_candidates_tied(
            ranked_candidates
        ):
            first = ranked_candidates[0]
            second = ranked_candidates[1]

            flags.append(
                ReviewFlag(
                    code="candidate_tie",
                    severity="warning",
                    message=(
                        "The two leading candidates are tied "
                        "on unique supporting genes, multi-source "
                        "genes, and specificity-weighted support."
                    ),
                    genes=sorted(
                        set(first.supporting_genes)
                        | set(second.supporting_genes)
                    ),
                    cell_types=[
                        first.cell_type,
                        second.cell_type,
                    ],
                )
            )

        direct_marker_genes = sorted(
            interpretation.gene
            for interpretation in interpretations
            if interpretation.direct_marker_supported
        )

        direct_marker_threshold = (
            self.config
            .direct_marker_warning_threshold(
                len(interpretations)
            )
        )

        if (
            len(direct_marker_genes)
            < direct_marker_threshold
        ):
            flags.append(
                ReviewFlag(
                    code=(
                        "limited_direct_marker_coverage"
                    ),
                    severity="warning",
                    message=(
                        "Direct breast-context marker coverage "
                        "is below the configured review threshold: "
                        f"{len(direct_marker_genes)} supported "
                        "genes versus a threshold of "
                        f"{direct_marker_threshold}."
                    ),
                    genes=direct_marker_genes,
                )
            )

        if not ranked_candidates:
            flags.append(
                ReviewFlag(
                    code="no_candidate_evidence",
                    severity="warning",
                    message=(
                        "No candidate cell type could be "
                        "constructed from direct breast-context "
                        "marker evidence."
                    ),
                )
            )

        uncertain_expression_genes = sorted(
            interpretation.gene
            for interpretation in interpretations
            if interpretation.evidence_category
            in {
                "ambiguous_expression",
                "broad_expression",
                "trace_expression",
            }
        )

        if uncertain_expression_genes:
            flags.append(
                ReviewFlag(
                    code="hpa_expression_uncertainty",
                    severity="info",
                    message=(
                        "Some non-marker genes show ambiguous, "
                        "broad, or trace healthy-breast expression "
                        "patterns in HPA."
                    ),
                    genes=(
                        uncertain_expression_genes
                    ),
                )
            )

        return sorted(
            flags,
            key=lambda flag: (
                0
                if flag.severity == "warning"
                else 1,
                flag.code,
            ),
        )

    @staticmethod
    def human_review_required(
        flags: Sequence[ReviewFlag],
    ) -> bool:
        """Return whether at least one warning-level flag exists."""

        return any(
            flag.severity == "warning"
            for flag in flags
        )