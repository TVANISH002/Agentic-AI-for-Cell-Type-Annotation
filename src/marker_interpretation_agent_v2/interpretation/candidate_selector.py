from __future__ import annotations

from collections.abc import Sequence

from marker_interpretation_agent_v2.config import (
    Agent2Config,
    DEFAULT_CONFIG,
)
from marker_interpretation_agent_v2.schemas import (
    CandidateCellType,
    CandidateSelection,
)


class CandidateSelector:
    """Select candidates using support-drop and threshold methods."""

    def __init__(
        self,
        config: Agent2Config | None = None,
    ) -> None:
        """Initialize the selector."""

        self.config = config or DEFAULT_CONFIG

    @staticmethod
    def _validate_candidates(
        candidates: Sequence[CandidateCellType],
    ) -> list[CandidateCellType]:
        """Validate candidate ranks and support ordering."""

        ordered = list(candidates)

        expected_ranks = list(
            range(1, len(ordered) + 1)
        )

        actual_ranks = [
            candidate.candidate_rank
            for candidate in ordered
        ]

        if actual_ranks != expected_ranks:
            raise ValueError(
                "Candidate ranks must be consecutive "
                "and begin at 1."
            )

        support_counts = [
            candidate.unique_supporting_gene_count
            for candidate in ordered
        ]

        if any(
            following > current
            for current, following in zip(
                support_counts,
                support_counts[1:],
            )
        ):
            raise ValueError(
                "Candidate supporting-gene counts must "
                "be in non-increasing order."
            )

        return ordered

    @staticmethod
    def _build_selection(
        *,
        candidates: Sequence[CandidateCellType],
        cutoff_method: str,
        cutoff_index: int | None,
        support_drop_absolute: int = 0,
        support_drop_relative: float = 0.0,
    ) -> CandidateSelection:
        """Build candidate-selection metadata."""

        selected = list(candidates)

        return CandidateSelection(
            selected_candidate_count=len(
                selected
            ),
            selected_candidate_cell_types=[
                candidate.cell_type
                for candidate in selected
            ],
            cutoff_index=cutoff_index,
            cutoff_method=cutoff_method,
            support_drop_absolute=(
                support_drop_absolute
            ),
            support_drop_relative=round(
                support_drop_relative,
                6,
            ),
        )

    def select(
        self,
        candidates: Sequence[CandidateCellType],
    ) -> tuple[
        list[CandidateCellType],
        CandidateSelection,
    ]:
        """Select candidates using the existing support-drop rule."""

        ordered = self._validate_candidates(
            candidates
        )

        if not ordered:
            return [], self._build_selection(
                candidates=[],
                cutoff_method="no_candidates",
                cutoff_index=None,
            )

        if len(ordered) == 1:
            selected = list(ordered)

            return selected, self._build_selection(
                candidates=selected,
                cutoff_method="single_candidate",
                cutoff_index=1,
            )

        support_counts = [
            candidate.unique_supporting_gene_count
            for candidate in ordered
        ]

        search_limit = min(
            len(ordered),
            self.config.candidate_elbow_search_limit,
        )

        qualifying_drops: list[
            tuple[int, int, float]
        ] = []

        for index in range(
            search_limit - 1
        ):
            current_support = (
                support_counts[index]
            )

            next_support = (
                support_counts[index + 1]
            )

            absolute_drop = (
                current_support
                - next_support
            )

            relative_drop = (
                absolute_drop
                / current_support
                if current_support > 0
                else 0.0
            )

            if (
                absolute_drop
                >= self.config
                .candidate_min_absolute_drop
                and relative_drop
                >= self.config
                .candidate_min_relative_drop
            ):
                qualifying_drops.append(
                    (
                        index + 1,
                        absolute_drop,
                        relative_drop,
                    )
                )

        if qualifying_drops:
            (
                cutoff_index,
                absolute_drop,
                relative_drop,
            ) = max(
                qualifying_drops,
                key=lambda item: (
                    item[1],
                    item[2],
                    -item[0],
                ),
            )

            selected = ordered[
                :cutoff_index
            ]

            return selected, self._build_selection(
                candidates=selected,
                cutoff_method=(
                    "largest_qualifying_support_drop"
                ),
                cutoff_index=cutoff_index,
                support_drop_absolute=(
                    absolute_drop
                ),
                support_drop_relative=(
                    relative_drop
                ),
            )

        selected = list(ordered)

        return selected, self._build_selection(
            candidates=selected,
            cutoff_method=(
                "all_candidates_retained_no_clear_drop"
            ),
            cutoff_index=None,
        )

    def select_by_support_threshold(
        self,
        candidates: Sequence[CandidateCellType],
        *,
        minimum_supporting_genes: int | None = None,
    ) -> list[CandidateCellType]:
        """
        Return all candidates meeting the minimum support threshold.

        This is the additional professor-requested reporting method.
        The existing support-drop selection remains unchanged.
        """

        ordered = self._validate_candidates(
            candidates
        )

        threshold = (
            self.config
            .candidate_reporting_min_supporting_genes
            if minimum_supporting_genes is None
            else minimum_supporting_genes
        )

        if threshold < 0:
            raise ValueError(
                "minimum_supporting_genes "
                "cannot be negative."
            )

        return [
            candidate
            for candidate in ordered
            if (
                candidate
                .unique_supporting_gene_count
                >= threshold
            )
        ]