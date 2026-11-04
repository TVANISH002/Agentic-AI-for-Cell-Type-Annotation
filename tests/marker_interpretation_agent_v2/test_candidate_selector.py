from __future__ import annotations

import pytest

from marker_interpretation_agent_v2.interpretation.candidate_selector import (
    CandidateSelector,
)
from marker_interpretation_agent_v2.schemas import CandidateCellType


def _candidate(
    rank: int,
    support_count: int,
    *,
    cell_type: str | None = None,
) -> CandidateCellType:
    """Create one ranked candidate for selector tests."""

    return CandidateCellType(
        candidate_rank=rank,
        cell_type=cell_type or f"Type {rank}",
        unique_supporting_gene_count=support_count,
        supporting_genes=[
            f"G{rank}_{index}"
            for index in range(1, support_count + 1)
        ],
        specificity_weighted_support=float(support_count),
        mean_supporting_gene_rank=float(rank),
    )


def test_selects_candidates_before_largest_qualifying_drop() -> None:
    """Select candidates before the strongest meaningful support drop."""

    candidates = [
        _candidate(1, 10),
        _candidate(2, 7),
        _candidate(3, 4),
        _candidate(4, 4),
    ]

    selected, metadata = CandidateSelector().select(candidates)

    assert [
        candidate.cell_type
        for candidate in selected
    ] == [
        "Type 1",
        "Type 2",
    ]

    assert metadata.selected_candidate_count == 2
    assert metadata.cutoff_index == 2

    assert (
        metadata.cutoff_method
        == "largest_qualifying_support_drop"
    )

    assert metadata.support_drop_absolute == 3
    assert metadata.support_drop_relative == pytest.approx(3 / 7)


def test_retains_all_candidates_without_clear_drop() -> None:
    """Retain all candidates when no meaningful support drop exists."""

    candidates = [
        _candidate(1, 15),
        _candidate(2, 14),
        _candidate(3, 13),
        _candidate(4, 12),
    ]

    selected, metadata = CandidateSelector().select(candidates)

    assert [
        candidate.candidate_rank
        for candidate in selected
    ] == [
        1,
        2,
        3,
        4,
    ]

    assert (
        metadata.cutoff_method
        == "all_candidates_retained_no_clear_drop"
    )

    assert metadata.cutoff_index is None
    assert metadata.selected_candidate_count == 4
    assert metadata.support_drop_absolute == 0
    assert metadata.support_drop_relative == 0.0


def test_retains_all_gradually_decreasing_candidates() -> None:
    """Retain gradual candidate distributions without forcing a cutoff."""

    candidates = [
        _candidate(1, 6),
        _candidate(2, 5),
        _candidate(3, 4),
        _candidate(4, 3),
    ]

    selected, metadata = CandidateSelector().select(candidates)

    assert [
        candidate.candidate_rank
        for candidate in selected
    ] == [
        1,
        2,
        3,
        4,
    ]

    assert (
        metadata.cutoff_method
        == "all_candidates_retained_no_clear_drop"
    )

    assert metadata.selected_candidate_count == 4
    assert metadata.cutoff_index is None


def test_retains_weak_candidates_when_no_drop_qualifies() -> None:
    """Retain weak candidates instead of forcing an artificial cutoff."""

    candidates = [
        _candidate(1, 2),
        _candidate(2, 1),
    ]

    selected, metadata = CandidateSelector().select(candidates)

    assert [
        candidate.candidate_rank
        for candidate in selected
    ] == [
        1,
        2,
    ]

    assert (
        metadata.cutoff_method
        == "all_candidates_retained_no_clear_drop"
    )

    assert metadata.selected_candidate_count == 2
    assert metadata.cutoff_index is None


def test_returns_only_candidate_for_single_candidate_input() -> None:
    """Handle a cluster with exactly one candidate."""

    candidate = _candidate(
        1,
        5,
        cell_type="Luminal epithelial cell",
    )

    selected, metadata = CandidateSelector().select([candidate])

    assert selected == [candidate]

    assert metadata.cutoff_method == "single_candidate"
    assert metadata.cutoff_index == 1
    assert metadata.selected_candidate_count == 1


def test_returns_empty_selection_for_empty_candidates() -> None:
    """Handle a cluster with no direct-marker-supported candidates."""

    selected, metadata = CandidateSelector().select([])

    assert selected == []
    assert metadata.selected_candidate_count == 0
    assert metadata.cutoff_method == "no_candidates"
    assert metadata.cutoff_index is None


def test_rejects_nonconsecutive_candidate_ranks() -> None:
    """Reject candidate lists with missing rank values."""

    candidates = [
        _candidate(1, 5),
        _candidate(3, 3),
    ]

    with pytest.raises(
        ValueError,
        match="(?i)consecutive",
    ):
        CandidateSelector().select(candidates)


def test_rejects_increasing_support_counts() -> None:
    """Require non-increasing supporting-gene counts."""

    candidates = [
        _candidate(1, 4),
        _candidate(2, 5),
    ]

    with pytest.raises(
        ValueError,
        match="(?i)non-increasing",
    ):
        CandidateSelector().select(candidates)