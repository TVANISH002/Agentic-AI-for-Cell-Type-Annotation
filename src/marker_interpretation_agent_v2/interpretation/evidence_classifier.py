from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from marker_interpretation_agent_v2.config import (
    Agent2Config,
    DEFAULT_CONFIG,
)
from marker_interpretation_agent_v2.schemas import (
    EvidenceCategory,
    GeneInterpretation,
    GeneStatistics,
    HPAEvidenceSummary,
    InputGene,
    PanelEvidenceCatalog,
    PanelGeneEvidence,
)


class EvidenceClassifier:
    """
    Classify cluster marker genes using panel-wide reusable evidence.

    Evidence hierarchy:

    1. Breast-context CellMarker/PanglaoDB evidence creates direct
       marker support.
    2. HPA healthy-breast expression is complementary evidence.
    3. HPA alone never creates a candidate cell-type marker vote.
    """

    def __init__(
        self,
        config: Agent2Config | None = None,
    ) -> None:
        """Initialize the classifier with validated configuration."""

        self.config = config or DEFAULT_CONFIG

    @staticmethod
    def _normalize_gene(value: str) -> str:
        """Normalize one gene symbol."""

        gene = str(value).strip().upper()

        if not gene:
            raise ValueError("Gene symbol cannot be empty.")

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
    def _context_sources(
        cls,
        panel_gene: PanelGeneEvidence,
    ) -> list[str]:
        """Return sources supporting breast-context marker evidence."""

        sources: list[str] = []

        for record in panel_gene.context_matched_database_evidence:
            source = (
                record.get("source")
                or record.get("database")
            )

            if source is not None and str(source).strip():
                sources.append(str(source).strip())

        if not sources and panel_gene.direct_marker_supported:
            sources.extend(panel_gene.evidence_sources)

        return cls._clean_strings(sources)

    def _review_pattern(
        self,
        hpa: HPAEvidenceSummary,
    ) -> str:
        """Determine the deterministic HPA expression review pattern."""

        if not hpa.queried:
            return "not_queried"

        if not hpa.gene_present_in_hpa:
            return "symbol_not_found"

        if not hpa.breast_expression_found:
            return "no_breast_expression"

        top_ncpm = hpa.top_ncpm
        highest_read_count = hpa.highest_read_count

        if (
            top_ncpm is not None
            and highest_read_count is not None
            and top_ncpm <= self.config.trace_max_ncpm
            and highest_read_count
            <= self.config.trace_max_read_count
        ):
            return "trace_expression"

        ratio = hpa.top_to_second_ncpm_ratio

        if (
            ratio is not None
            and ratio <= self.config.near_tie_ratio
        ):
            return "near_tie"

        if (
            hpa.detected_cell_type_count
            >= self.config.broad_cell_type_count
            and ratio is not None
            and ratio <= self.config.broad_ratio
        ):
            return "broad_expression"

        return "expression_pattern_available"

    @staticmethod
    def _evidence_category(
        *,
        direct_marker_supported: bool,
        review_pattern: str,
        hpa: HPAEvidenceSummary,
    ) -> EvidenceCategory:
        """Assign the final gene-level evidence category."""

        if direct_marker_supported:
            return "direct_marker_supported"

        if review_pattern == "symbol_not_found":
            return "symbol_review_needed"

        if review_pattern == "trace_expression":
            return "trace_expression"

        if review_pattern == "near_tie":
            return "ambiguous_expression"

        if review_pattern == "broad_expression":
            return "broad_expression"

        if hpa.breast_expression_found:
            return "expression_only"

        return "unresolved"

    @staticmethod
    def _specificity_score(
        supported_cell_types: Sequence[str],
    ) -> float:
        """
        Calculate inverse cell-type specificity.

        A gene supporting one cell type receives 1.0.
        A gene supporting two cell types receives 0.5.
        """

        count = len(set(supported_cell_types))

        if count == 0:
            return 0.0

        return 1.0 / float(count)

    def _statistics(
        self,
        input_gene: InputGene,
        panel_gene: PanelGeneEvidence,
        sources: Sequence[str],
    ) -> GeneStatistics:
        """Build deterministic statistics used by later ranking."""

        supported_cell_types = (
            panel_gene.context_matched_cell_types
        )

        return GeneStatistics(
            source_agreement_count=len(set(sources)),
            marker_specificity_score=(
                self._specificity_score(
                    supported_cell_types
                )
            ),
            context_evidence_count=len(
                panel_gene.context_matched_database_evidence
            ),
            cluster_gene_rank=(
                input_gene.rank_within_cluster
            ),
            log_fold_change=input_gene.log_fold_change,
        )

    def _warnings(
        self,
        panel_gene: PanelGeneEvidence,
        *,
        category: EvidenceCategory,
        review_pattern: str,
    ) -> list[str]:
        """Build transparent gene-level warnings."""

        warnings = [
            *panel_gene.warnings,
            *panel_gene.hpa.warnings,
        ]

        if (
            not panel_gene.direct_marker_supported
            and panel_gene.hpa.breast_expression_found
        ):
            warnings.append(
                "HPA breast expression is supplemental evidence "
                "and is not counted as a direct marker vote."
            )

        if (
            not panel_gene.direct_marker_supported
            and panel_gene.global_database_evidence
        ):
            warnings.append(
                "Global CellMarker/PanglaoDB evidence exists, "
                "but no breast-context marker evidence was found."
            )

        if category == "symbol_review_needed":
            warnings.append(
                "The exact gene symbol was not found in HPA and "
                "should be reviewed against an authoritative "
                "gene-symbol source."
            )

        if review_pattern == "no_breast_expression":
            warnings.append(
                "The gene was found in HPA, but usable healthy-breast "
                "cell-type expression was not found."
            )

        if category == "trace_expression":
            warnings.append(
                "HPA healthy-breast expression is at or below the "
                "configured trace-expression thresholds."
            )

        if category == "ambiguous_expression":
            warnings.append(
                "The top two HPA breast cell types have similar "
                "expression and should not be treated as a specific "
                "cell-type signal."
            )

        if category == "broad_expression":
            warnings.append(
                "HPA expression is distributed across many breast "
                "cell types and is therefore non-specific."
            )

        if review_pattern == "not_queried":
            warnings.append(
                "HPA was not queried for this gene. The panel catalog "
                "must be completed before all-cluster interpretation."
            )

        if category == "unresolved":
            warnings.append(
                "No direct breast-context marker evidence or usable "
                "HPA breast-expression evidence was found."
            )

        return self._clean_strings(warnings)

    def classify_gene(
        self,
        input_gene: InputGene,
        panel_gene: PanelGeneEvidence,
    ) -> GeneInterpretation:
        """Classify one cluster marker gene."""

        input_symbol = self._normalize_gene(input_gene.gene)
        panel_symbol = self._normalize_gene(panel_gene.gene)

        if input_symbol != panel_symbol:
            raise ValueError(
                "Input gene and panel evidence gene do not match: "
                f"{input_symbol!r} != {panel_symbol!r}"
            )

        supported_cell_types = self._clean_strings(
            panel_gene.context_matched_cell_types
        )

        direct_marker_supported = bool(
            supported_cell_types
        )

        sources = self._context_sources(panel_gene)

        review_pattern = self._review_pattern(
            panel_gene.hpa
        )

        category = self._evidence_category(
            direct_marker_supported=direct_marker_supported,
            review_pattern=review_pattern,
            hpa=panel_gene.hpa,
        )

        hpa_summary = panel_gene.hpa.model_copy(
            update={
                "review_pattern": review_pattern,
                "warnings": self._clean_strings(
                    panel_gene.hpa.warnings
                ),
            }
        )

        return GeneInterpretation(
            gene=input_symbol,
            direct_marker_supported=(
                direct_marker_supported
            ),
            evidence_category=category,
            supported_cell_types=(
                supported_cell_types
            ),
            context_evidence=[
                dict(record)
                for record
                in panel_gene.context_matched_database_evidence
            ],
            evidence_sources=sources,
            statistics=self._statistics(
                input_gene,
                panel_gene,
                sources,
            ),
            hpa=hpa_summary,
            warnings=self._warnings(
                panel_gene,
                category=category,
                review_pattern=review_pattern,
            ),
        )

    def classify(
        self,
        input_genes: Sequence[InputGene],
        catalog: PanelEvidenceCatalog,
    ) -> list[GeneInterpretation]:
        """Classify all selected genes for one cluster."""

        if not input_genes:
            return []

        input_symbols = [
            self._normalize_gene(gene.gene)
            for gene in input_genes
        ]

        duplicate_input_genes = sorted(
            {
                gene
                for gene in input_symbols
                if input_symbols.count(gene) > 1
            }
        )

        if duplicate_input_genes:
            raise ValueError(
                "Cluster input contains duplicate genes: "
                f"{duplicate_input_genes}"
            )

        catalog_map: dict[str, PanelGeneEvidence] = {
            self._normalize_gene(record.gene): record
            for record in catalog.genes
        }

        missing_genes = sorted(
            set(input_symbols) - set(catalog_map)
        )

        if missing_genes:
            raise ValueError(
                "Selected cluster genes are missing from the "
                "panel evidence catalog: "
                f"{missing_genes}"
            )

        interpretations: list[GeneInterpretation] = []

        for input_gene in input_genes:
            gene = self._normalize_gene(
                input_gene.gene
            )

            interpretations.append(
                self.classify_gene(
                    input_gene,
                    catalog_map[gene],
                )
            )

        return interpretations

    @staticmethod
    def category_counts(
        interpretations: Sequence[
            GeneInterpretation
        ],
    ) -> dict[str, int]:
        """Return deterministic category counts."""

        counts: dict[str, int] = {}

        for interpretation in interpretations:
            category = interpretation.evidence_category

            counts[category] = (
                counts.get(category, 0) + 1
            )

        return {
            category: counts[category]
            for category in sorted(counts)
        }