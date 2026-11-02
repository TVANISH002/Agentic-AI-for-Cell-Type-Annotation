from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from marker_interpretation_agent_v2.schemas import (
    CandidateCellType,
    CandidateSelection,
    GeneInterpretation,
    ReviewFlag,
)


class ExplanationBuilder:
    """
    Create a concise explanation of one cluster interpretation.

    The explanation describes the available biological evidence without
    claiming a final annotation or a confidence percentage.
    """

    @staticmethod
    def _normalize_cluster_id(cluster_id: str | int) -> str:
        """Normalize one cluster identifier."""

        normalized = str(cluster_id).strip()

        if not normalized:
            raise ValueError("cluster_id cannot be empty.")

        return normalized

    @staticmethod
    def _count_text(count: int, singular: str, plural: str | None = None) -> str:
        """Return a count with the correct singular or plural label."""

        label = singular if count == 1 else (plural or f"{singular}s")
        return f"{count} {label}"

    @staticmethod
    def _category_counts(
        interpretations: Sequence[GeneInterpretation],
    ) -> Counter[str]:
        """Count gene evidence categories."""

        return Counter(
            interpretation.evidence_category for interpretation in interpretations
        )

    @staticmethod
    def _human_review_required(review_flags: Sequence[ReviewFlag]) -> bool:
        """Return whether warning-level review flags exist."""

        return any(flag.severity == "warning" for flag in review_flags)

    @staticmethod
    def _warning_codes(review_flags: Sequence[ReviewFlag]) -> list[str]:
        """Return deterministic warning-level flag codes."""

        return sorted(
            {
                flag.code
                for flag in review_flags
                if flag.severity == "warning"
            }
        )

    def _selected_candidate_text(
        self,
        selected_candidates: Sequence[CandidateCellType],
    ) -> str:
        """Describe selected candidates and direct-marker support."""

        if not selected_candidates:
            return (
                "No candidate cell type was supported by direct "
                "breast-context marker evidence."
            )

        if len(selected_candidates) == 1:
            candidate = selected_candidates[0]
            support_text = self._count_text(
                candidate.unique_supporting_gene_count,
                "unique direct-marker gene",
            )

            return (
                f"The leading candidate is {candidate.cell_type}, supported by "
                f"{support_text}."
            )

        candidate_descriptions = [
            (
                f"{candidate.cell_type} "
                f"({self._count_text(candidate.unique_supporting_gene_count, 'gene')})"
            )
            for candidate in selected_candidates
        ]

        return (
            "The support-based cutoff retained these leading candidates: "
            + ", ".join(candidate_descriptions)
            + "."
        )

    def _hpa_summary(
        self,
        interpretations: Sequence[GeneInterpretation],
    ) -> str:
        """Describe HPA coverage without treating it as a marker vote."""

        total_genes = len(interpretations)
        queried_count = sum(
            1 for interpretation in interpretations if interpretation.hpa.queried
        )
        gene_match_count = sum(
            1
            for interpretation in interpretations
            if interpretation.hpa.gene_present_in_hpa
        )
        breast_expression_count = sum(
            1
            for interpretation in interpretations
            if interpretation.hpa.breast_expression_found
        )

        total_label = "gene" if total_genes == 1 else "genes"
        match_verb = "was" if gene_match_count == 1 else "were"

        return (
            f"HPA was queried for {queried_count} of {total_genes} selected "
            f"{total_label}; {gene_match_count} "
            f"{'gene' if gene_match_count == 1 else 'genes'} {match_verb} found "
            f"in HPA and {breast_expression_count} "
            f"{'gene' if breast_expression_count == 1 else 'genes'} "
            "had usable healthy-breast cell-type expression evidence."
        )

    @staticmethod
    def _category_summary(category_counts: Counter[str]) -> str:
        """Create a compact category-count statement."""

        if not category_counts:
            return "No gene interpretations were available."

        parts = [
            f"{category}={category_counts[category]}"
            for category in sorted(category_counts)
        ]

        return "Gene evidence categories were: " + ", ".join(parts) + "."

    def _selection_summary(self, candidate_selection: CandidateSelection) -> str:
        """Describe how the candidate cutoff was chosen."""

        method = candidate_selection.cutoff_method

        if method == "no_candidates":
            return (
                "No candidate-selection cutoff was applied because no "
                "direct-marker-supported candidates were available."
            )

        if method == "single_candidate":
            return "Candidate selection retained the only available candidate."

        if method == "all_candidates_retained_no_clear_drop":
            return (
                "No meaningful support drop met the configured thresholds, so "
                "all similarly supported candidates were retained for human or "
                "expert review."
            )

        if method == "largest_qualifying_support_drop":
            absolute_drop = candidate_selection.support_drop_absolute or 0
            relative_drop = candidate_selection.support_drop_relative or 0.0
            drop_text = self._count_text(absolute_drop, "gene")

            return (
                "Candidate selection used the largest qualifying support drop, "
                f"with an absolute drop of {drop_text} and a relative drop of "
                f"{relative_drop * 100.0:.1f}%."
            )

        return "Candidate selection used " + method.replace("_", " ") + "."

    def build(
        self,
        *,
        cluster_id: str | int,
        interpretations: Sequence[GeneInterpretation],
        ranked_candidates: Sequence[CandidateCellType],
        selected_candidates: Sequence[CandidateCellType],
        candidate_selection: CandidateSelection,
        review_flags: Sequence[ReviewFlag],
    ) -> str:
        """Build the complete cluster-level explanation."""

        normalized_cluster_id = self._normalize_cluster_id(cluster_id)
        total_genes = len(interpretations)
        direct_marker_count = sum(
            1
            for interpretation in interpretations
            if interpretation.direct_marker_supported
        )
        category_counts = self._category_counts(interpretations)
        human_review_required = self._human_review_required(review_flags)
        warning_codes = self._warning_codes(review_flags)

        ranked_candidate_count = len(ranked_candidates)
        ranked_candidate_verb = "was" if ranked_candidate_count == 1 else "were"

        sentences = [
            (
                f"Cluster {normalized_cluster_id} was interpreted using "
                f"{self._count_text(total_genes, 'selected marker gene')}."
            ),
            (
                f"{self._count_text(direct_marker_count, 'gene')} had direct "
                "breast-context marker evidence from CellMarker and/or PanglaoDB."
            ),
            self._hpa_summary(interpretations),
            self._category_summary(category_counts),
            self._selected_candidate_text(selected_candidates),
            (
                f"{self._count_text(ranked_candidate_count, 'candidate cell type')} "
                f"{ranked_candidate_verb} retained in the complete ranked audit table."
            ),
            self._selection_summary(candidate_selection),
        ]

        if human_review_required:
            sentences.append(
                "Human review is recommended because the following warning "
                f"flags were raised: {', '.join(warning_codes)}."
            )
        else:
            sentences.append("No warning-level review flag was raised.")

        sentences.extend(
            [
                (
                    "HPA evidence is complementary healthy-breast expression "
                    "evidence and does not independently create candidate marker votes."
                ),
                (
                    "This MarkerEvidence output provides ranked candidate evidence "
                    "for downstream agents; it is not yet a final cell-type "
                    "annotation or a percentage-confidence result."
                ),
            ]
        )

        return " ".join(
            sentence.strip() for sentence in sentences if sentence.strip()
        )