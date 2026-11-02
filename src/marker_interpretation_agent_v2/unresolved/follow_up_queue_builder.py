from __future__ import annotations

from collections.abc import Sequence

from marker_interpretation_agent_v2.schemas import (
    FollowUpGene,
    GeneInterpretation,
)


class FollowUpQueueBuilder:
    """
    Build a focused queue of genes requiring later investigation.

    The current queue includes only genes that remain unresolved after
    CellMarker, PanglaoDB, and HPA processing, or genes whose symbols
    require review.

    HPA expression-only, broad, ambiguous, and trace-expression genes
    remain in the Agent 2 evidence table but are not automatically sent
    to literature retrieval because evidence was already found for them.
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
    def _clean_strings(
        values: Sequence[str] | None,
    ) -> list[str]:
        """Return deterministic unique non-empty strings."""

        if not values:
            return []

        return sorted(
            {
                str(value).strip()
                for value in values
                if str(value).strip()
            },
            key=str.casefold,
        )

    @classmethod
    def _source_coverage(
        cls,
        interpretation: GeneInterpretation,
    ) -> list[str]:
        """Describe which evidence sources were checked or matched."""

        coverage: list[str] = []

        for source in interpretation.evidence_sources:
            cleaned_source = str(source).strip()

            if cleaned_source:
                coverage.append(
                    f"marker_source:{cleaned_source}"
                )

        if interpretation.hpa.queried:
            coverage.append("HPA:queried")
        else:
            coverage.append("HPA:not_queried")

        if interpretation.hpa.gene_present_in_hpa:
            coverage.append("HPA:gene_present")
        else:
            coverage.append("HPA:gene_not_found")

        if interpretation.hpa.breast_expression_found:
            coverage.append(
                "HPA:breast_expression_found"
            )
        else:
            coverage.append(
                "HPA:no_breast_expression"
            )

        return cls._clean_strings(
            coverage
        )

    @staticmethod
    def _symbol_review_reason(
        interpretation: GeneInterpretation,
    ) -> str:
        """Explain why a gene requires symbol review."""

        return (
            f"{interpretation.gene} was not found in HPA using "
            "the exact submitted gene symbol and has no direct "
            "breast-context marker evidence. Review the symbol "
            "against an authoritative gene-symbol source before "
            "performing literature retrieval."
        )

    @staticmethod
    def _unresolved_reason(
        interpretation: GeneInterpretation,
    ) -> str:
        """Explain why a gene requires targeted literature review."""

        if (
            interpretation.hpa.queried
            and interpretation.hpa.gene_present_in_hpa
            and not interpretation.hpa.breast_expression_found
        ):
            return (
                f"{interpretation.gene} has no direct breast-context "
                "marker support from CellMarker or PanglaoDB. The gene "
                "was found in HPA, but usable healthy-breast cell-type "
                "expression evidence was not found. It should enter the "
                "later targeted literature-review step."
            )

        if not interpretation.hpa.queried:
            return (
                f"{interpretation.gene} has no direct breast-context "
                "marker support and HPA was not queried. Complete the "
                "panel-wide HPA lookup before literature retrieval."
            )

        return (
            f"{interpretation.gene} has no direct breast-context marker "
            "support from CellMarker or PanglaoDB and no usable HPA "
            "healthy-breast expression evidence. It should enter the "
            "later targeted literature-review step."
        )

    @classmethod
    def _build_symbol_review_item(
        cls,
        interpretation: GeneInterpretation,
    ) -> FollowUpGene:
        """Build one authoritative symbol-review item."""

        warnings = cls._clean_strings(
            [
                *interpretation.warnings,
                (
                    "Do not perform automated literature retrieval "
                    "until the submitted gene symbol is reviewed."
                ),
            ]
        )

        return FollowUpGene(
            gene=interpretation.gene,
            reason=cls._symbol_review_reason(
                interpretation
            ),
            recommended_action="symbol_review",
            evidence_category=(
                interpretation.evidence_category
            ),
            source_coverage=cls._source_coverage(
                interpretation
            ),
            warnings=warnings,
        )

    @classmethod
    def _build_unresolved_item(
        cls,
        interpretation: GeneInterpretation,
    ) -> FollowUpGene:
        """Build one targeted literature-retrieval item."""

        warnings = cls._clean_strings(
            [
                *interpretation.warnings,
                (
                    "This queue entry is for a later narrow "
                    "literature-retrieval workflow, not for "
                    "automatic annotation inside Agent 2."
                ),
            ]
        )

        recommended_action = (
            "manual_review"
            if not interpretation.hpa.queried
            else "targeted_literature_retrieval"
        )

        return FollowUpGene(
            gene=interpretation.gene,
            reason=cls._unresolved_reason(
                interpretation
            ),
            recommended_action=recommended_action,
            evidence_category=(
                interpretation.evidence_category
            ),
            source_coverage=cls._source_coverage(
                interpretation
            ),
            warnings=warnings,
        )

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
                "Gene interpretations contain duplicate genes: "
                f"{duplicates}"
            )

    def build(
        self,
        interpretations: Sequence[
            GeneInterpretation
        ],
    ) -> list[FollowUpGene]:
        """
        Build the focused unresolved-gene queue.

        Included categories:
        - symbol_review_needed
        - unresolved

        Excluded categories:
        - direct_marker_supported
        - expression_only
        - ambiguous_expression
        - broad_expression
        - trace_expression
        """

        self._validate_interpretations(
            interpretations
        )

        queue: list[FollowUpGene] = []

        for interpretation in interpretations:
            if interpretation.direct_marker_supported:
                continue

            category = (
                interpretation.evidence_category
            )

            if category == "symbol_review_needed":
                queue.append(
                    self._build_symbol_review_item(
                        interpretation
                    )
                )

            elif category == "unresolved":
                queue.append(
                    self._build_unresolved_item(
                        interpretation
                    )
                )

        return sorted(
            queue,
            key=lambda item: (
                0
                if item.recommended_action
                == "symbol_review"
                else 1,
                item.gene,
            ),
        )

    @staticmethod
    def action_counts(
        queue: Sequence[FollowUpGene],
    ) -> dict[str, int]:
        """Return deterministic counts by recommended action."""

        counts: dict[str, int] = {}

        for item in queue:
            action = item.recommended_action

            counts[action] = (
                counts.get(action, 0) + 1
            )

        return {
            action: counts[action]
            for action in sorted(counts)
        }