from __future__ import annotations

from collections.abc import Sequence

from marker_interpretation_agent_v2.interpretation.candidate_selector import (
    CandidateSelector,
)
from marker_interpretation_agent_v2.schemas import (
    CandidateCellType,
    CandidateSelection,
)


class MarkerOnlyCandidateSelector(CandidateSelector):
    """Bound the no-drop fallback without changing candidate ranking."""

    def select(
        self,
        candidates: Sequence[CandidateCellType],
    ) -> tuple[list[CandidateCellType], CandidateSelection]:
        """Keep multi-gene candidates when no clear support drop exists."""

        ordered = self._validate_candidates(candidates)
        selected, selection = super().select(ordered)

        if (
            selection.cutoff_method
            != "all_candidates_retained_no_clear_drop"
        ):
            return selected, selection

        eligible = [
            candidate
            for candidate in ordered
            if (
                candidate.unique_supporting_gene_count
                >= self.config.candidate_min_supporting_genes
            )
        ]

        bounded = eligible[
            : self.config.candidate_fallback_max_candidates
        ]

        if bounded:
            return bounded, self._build_selection(
                candidates=bounded,
                cutoff_method=(
                    "bounded_multi_gene_fallback_no_clear_drop"
                ),
                cutoff_index=len(bounded),
            )

        leading = ordered[:1]

        return leading, self._build_selection(
            candidates=leading,
            cutoff_method=(
                "leading_candidate_fallback_below_minimum_support"
            ),
            cutoff_index=1,
        )