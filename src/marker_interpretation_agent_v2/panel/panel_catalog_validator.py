from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from marker_interpretation_agent_v2.schemas import (
    PanelEvidenceCatalog,
    PanelGeneEvidence,
)


@dataclass(frozen=True, slots=True)
class CatalogValidationResult:
    """Summary returned after validating a panel evidence catalog."""

    is_valid: bool
    panel_gene_count: int
    expected_gene_count: int | None
    direct_marker_gene_count: int
    hpa_queried_gene_count: int
    hpa_gene_match_count: int
    hpa_breast_expression_gene_count: int
    unresolved_gene_count: int
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable validation summary."""

        payload = asdict(self)
        payload["warnings"] = list(self.warnings)

        return payload


class PanelCatalogValidator:
    """
    Validate the reusable panel-wide evidence catalog.

    The validator enforces the professor-requested workflow:
    every panel gene appears once, HPA is queried for every gene,
    breast-context marker evidence is internally consistent, and
    the unresolved-gene list matches the available evidence.
    """

    @staticmethod
    def _normalize_gene(value: str) -> str:
        """Normalize one gene symbol."""

        gene = str(value).strip().upper()

        if not gene:
            raise ValueError("Gene symbol cannot be empty.")

        return gene

    @classmethod
    def _normalize_expected_genes(
        cls,
        expected_genes: Sequence[str],
    ) -> list[str]:
        """Normalize and validate an expected panel gene list."""

        normalized = [
            cls._normalize_gene(gene)
            for gene in expected_genes
        ]

        counts = Counter(normalized)

        duplicates = sorted(
            gene
            for gene, count in counts.items()
            if count > 1
        )

        if duplicates:
            duplicate_text = cls._format_gene_list(
                duplicates
            )

            raise ValueError(
                "Expected panel gene list contains duplicate genes: "
                f"{duplicate_text}"
            )

        return normalized

    @staticmethod
    def _format_gene_list(
        genes: Sequence[str],
        *,
        limit: int = 20,
    ) -> str:
        """Format a gene list without producing an overly long message."""

        sorted_genes = sorted(set(genes))

        if len(sorted_genes) <= limit:
            return ", ".join(sorted_genes)

        visible = ", ".join(
            sorted_genes[:limit]
        )

        remaining = len(sorted_genes) - limit

        return f"{visible}, ... and {remaining} more"

    @staticmethod
    def _computed_unresolved_genes(
        genes: Sequence[PanelGeneEvidence],
    ) -> set[str]:
        """
        Compute genes unresolved for breast-context interpretation.

        A gene is resolved when it has direct breast-context marker
        evidence or healthy-breast expression evidence from HPA.
        """

        return {
            gene.gene
            for gene in genes
            if (
                not gene.direct_marker_supported
                and not gene.hpa.breast_expression_found
            )
        }

    @staticmethod
    def _validate_gene_evidence_consistency(
        gene: PanelGeneEvidence,
    ) -> list[str]:
        """Return structural consistency errors for one gene."""

        errors: list[str] = []

        has_context_cell_types = bool(
            gene.context_matched_cell_types
        )

        has_context_records = bool(
            gene.context_matched_database_evidence
        )

        if (
            has_context_cell_types
            and not has_context_records
        ):
            errors.append(
                f"{gene.gene}: context-matched cell types exist, "
                "but context-matched database evidence is empty."
            )

        if (
            has_context_records
            and not has_context_cell_types
        ):
            errors.append(
                f"{gene.gene}: context-matched database evidence exists, "
                "but context-matched cell types are empty."
            )

        if (
            gene.direct_marker_supported
            and not gene.evidence_sources
        ):
            errors.append(
                f"{gene.gene}: direct marker support exists, "
                "but evidence_sources is empty."
            )

        if (
            gene.hpa.breast_expression_found
            and gene.hpa.top_cell_type is None
        ):
            errors.append(
                f"{gene.gene}: HPA breast expression was found, "
                "but top_cell_type is missing."
            )

        if (
            gene.hpa.breast_expression_found
            and gene.hpa.top_ncpm is None
        ):
            errors.append(
                f"{gene.gene}: HPA breast expression was found, "
                "but top_ncpm is missing."
            )

        return errors

    def validate(
        self,
        catalog: PanelEvidenceCatalog | Mapping[str, Any],
        *,
        expected_genes: Sequence[str] | None = None,
        require_hpa_for_all_genes: bool = True,
    ) -> CatalogValidationResult:
        """Validate one panel catalog and return its summary."""

        validated_catalog = (
            catalog
            if isinstance(
                catalog,
                PanelEvidenceCatalog,
            )
            else PanelEvidenceCatalog.model_validate(
                catalog
            )
        )

        errors: list[str] = []
        warnings: list[str] = []

        actual_genes = [
            gene.gene
            for gene in validated_catalog.genes
        ]

        actual_gene_set = set(
            actual_genes
        )

        expected_gene_count: int | None = None

        if expected_genes is not None:
            normalized_expected_genes = (
                self._normalize_expected_genes(
                    expected_genes
                )
            )

            expected_gene_set = set(
                normalized_expected_genes
            )

            expected_gene_count = len(
                normalized_expected_genes
            )

            missing_genes = sorted(
                expected_gene_set
                - actual_gene_set
            )

            extra_genes = sorted(
                actual_gene_set
                - expected_gene_set
            )

            if missing_genes:
                missing_text = (
                    self._format_gene_list(
                        missing_genes
                    )
                )

                errors.append(
                    "Catalog is missing expected panel genes: "
                    f"{missing_text}"
                )

            if extra_genes:
                extra_text = (
                    self._format_gene_list(
                        extra_genes
                    )
                )

                errors.append(
                    "Catalog contains genes not present "
                    "in the panel: "
                    f"{extra_text}"
                )

        for gene in validated_catalog.genes:
            errors.extend(
                self._validate_gene_evidence_consistency(
                    gene
                )
            )

        unqueried_hpa_genes = [
            gene.gene
            for gene in validated_catalog.genes
            if not gene.hpa.queried
        ]

        if (
            require_hpa_for_all_genes
            and unqueried_hpa_genes
        ):
            unqueried_text = (
                self._format_gene_list(
                    unqueried_hpa_genes
                )
            )

            errors.append(
                "HPA was not queried for every panel gene. "
                f"Unqueried genes: {unqueried_text}"
            )

        computed_unresolved = (
            self._computed_unresolved_genes(
                validated_catalog.genes
            )
        )

        declared_unresolved = set(
            validated_catalog.unresolved_genes
        )

        missing_from_unresolved = sorted(
            computed_unresolved
            - declared_unresolved
        )

        incorrectly_unresolved = sorted(
            declared_unresolved
            - computed_unresolved
        )

        if missing_from_unresolved:
            missing_unresolved_text = (
                self._format_gene_list(
                    missing_from_unresolved
                )
            )

            errors.append(
                "Catalog unresolved_genes is missing genes "
                "that have neither direct breast-marker evidence "
                "nor HPA breast expression: "
                f"{missing_unresolved_text}"
            )

        if incorrectly_unresolved:
            incorrectly_unresolved_text = (
                self._format_gene_list(
                    incorrectly_unresolved
                )
            )

            errors.append(
                "Catalog unresolved_genes contains genes "
                "that already have usable breast-context evidence: "
                f"{incorrectly_unresolved_text}"
            )

        direct_marker_gene_count = sum(
            1
            for gene in validated_catalog.genes
            if gene.direct_marker_supported
        )

        hpa_queried_gene_count = sum(
            1
            for gene in validated_catalog.genes
            if gene.hpa.queried
        )

        hpa_gene_match_count = sum(
            1
            for gene in validated_catalog.genes
            if gene.hpa.gene_present_in_hpa
        )

        hpa_breast_expression_gene_count = sum(
            1
            for gene in validated_catalog.genes
            if gene.hpa.breast_expression_found
        )

        direct_markers_without_hpa_expression = [
            gene.gene
            for gene in validated_catalog.genes
            if (
                gene.direct_marker_supported
                and gene.hpa.queried
                and not gene.hpa.breast_expression_found
            )
        ]

        if direct_markers_without_hpa_expression:
            direct_marker_text = (
                self._format_gene_list(
                    direct_markers_without_hpa_expression
                )
            )

            warnings.append(
                "Some direct marker genes do not have "
                "healthy-breast expression evidence in HPA. "
                "This is not automatically an error because "
                "HPA is complementary evidence. "
                f"Genes: {direct_marker_text}"
            )

        if errors:
            formatted_errors = "\n".join(
                f"  {index}. {message}"
                for index, message in enumerate(
                    errors,
                    start=1,
                )
            )

            raise ValueError(
                "Panel evidence catalog validation failed:\n"
                f"{formatted_errors}"
            )

        return CatalogValidationResult(
            is_valid=True,
            panel_gene_count=(
                validated_catalog.panel_gene_count
            ),
            expected_gene_count=expected_gene_count,
            direct_marker_gene_count=(
                direct_marker_gene_count
            ),
            hpa_queried_gene_count=(
                hpa_queried_gene_count
            ),
            hpa_gene_match_count=(
                hpa_gene_match_count
            ),
            hpa_breast_expression_gene_count=(
                hpa_breast_expression_gene_count
            ),
            unresolved_gene_count=len(
                validated_catalog.unresolved_genes
            ),
            warnings=tuple(
                sorted(set(warnings))
            ),
        )

    def validate_file(
        self,
        catalog_path: str | Path,
        *,
        expected_genes: Sequence[str] | None = None,
        require_hpa_for_all_genes: bool = True,
    ) -> CatalogValidationResult:
        """Load and validate a panel catalog JSON file."""

        resolved_path = Path(
            catalog_path
        ).expanduser().resolve()

        if not resolved_path.exists():
            raise FileNotFoundError(
                "Panel catalog file does not exist: "
                f"{resolved_path}"
            )

        try:
            with resolved_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                payload: Any = json.load(
                    file
                )
        except json.JSONDecodeError as error:
            raise ValueError(
                "Panel catalog contains invalid JSON: "
                f"{resolved_path}"
            ) from error

        if not isinstance(
            payload,
            dict,
        ):
            raise ValueError(
                "Panel catalog JSON must contain "
                "one top-level object."
            )

        return self.validate(
            payload,
            expected_genes=expected_genes,
            require_hpa_for_all_genes=(
                require_hpa_for_all_genes
            ),
        )