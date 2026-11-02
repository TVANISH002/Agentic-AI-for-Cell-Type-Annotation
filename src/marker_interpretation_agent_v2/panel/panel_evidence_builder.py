from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from marker_interpretation_agent_v2.panel.panel_catalog_validator import (
    CatalogValidationResult,
    PanelCatalogValidator,
)
from marker_interpretation_agent_v2.panel.panel_evidence_loader import (
    PanelEvidenceLoader,
)
from marker_interpretation_agent_v2.paths import (
    PANEL_CATALOG_CSV,
    PANEL_CATALOG_JSON,
    PANEL_UNRESOLVED_GENES_CSV,
    normalize_path,
)
from marker_interpretation_agent_v2.schemas import (
    HPAEvidenceSummary,
    PanelEvidenceCatalog,
    PanelGeneEvidence,
)


class PanelEvidenceBuilder:
    """
    Merge panel-wide Agent 1 and HPA outputs into one reusable catalog.

    CellMarker and PanglaoDB provide direct marker evidence.
    HPA provides complementary healthy-breast expression evidence.
    """

    def __init__(
        self,
        loader: PanelEvidenceLoader | None = None,
        validator: PanelCatalogValidator | None = None,
    ) -> None:
        """Initialize the panel evidence builder."""

        self.loader = loader or PanelEvidenceLoader()
        self.validator = validator or PanelCatalogValidator()

    @staticmethod
    def _normalize_gene(value: Any) -> str:
        """Normalize one gene symbol."""

        gene = str(value).strip().upper()

        if not gene:
            raise ValueError("Gene symbol cannot be empty.")

        return gene

    @staticmethod
    def _clean_strings(values: Any) -> list[str]:
        """Create a deterministic list of unique non-empty strings."""

        if values is None:
            return []

        if isinstance(values, str):
            normalized_values: Sequence[Any] = (
                values.replace(",", "|").split("|")
            )
        elif (
            isinstance(values, Sequence)
            and not isinstance(
                values,
                (
                    bytes,
                    bytearray,
                ),
            )
        ):
            normalized_values = values
        else:
            normalized_values = [values]

        return sorted(
            {
                str(value).strip()
                for value in normalized_values
                if value is not None
                and str(value).strip()
            },
            key=str.casefold,
        )

    @staticmethod
    def _records(
        value: Any,
        *,
        field_name: str,
    ) -> list[dict[str, Any]]:
        """Validate and normalize a list of JSON object records."""

        if value is None:
            return []

        if not isinstance(value, list):
            raise ValueError(
                f"{field_name} must be a list."
            )

        records: list[dict[str, Any]] = []

        for index, item in enumerate(
            value,
            start=1,
        ):
            if not isinstance(
                item,
                Mapping,
            ):
                raise ValueError(
                    f"{field_name} item {index} "
                    "must be a JSON object."
                )

            records.append(
                dict(item)
            )

        return records

    @staticmethod
    def _optional_text(
        value: Any,
    ) -> str | None:
        """Return stripped text or None."""

        if value is None:
            return None

        text = str(value).strip()

        return text or None

    @staticmethod
    def _optional_float(
        value: Any,
    ) -> float | None:
        """Return a non-negative float or None."""

        if value is None or value == "":
            return None

        number = float(value)

        if number < 0:
            raise ValueError(
                "Numeric HPA values cannot be negative. "
                f"Received: {value!r}"
            )

        return number

    @staticmethod
    def _non_negative_int(
        value: Any,
        *,
        default: int = 0,
    ) -> int:
        """Return a non-negative whole number."""

        if value is None or value == "":
            return default

        numeric_value = float(value)

        if (
            numeric_value < 0
            or not numeric_value.is_integer()
        ):
            raise ValueError(
                "Count values must be non-negative "
                f"whole numbers. Received: {value!r}"
            )

        return int(numeric_value)

    @staticmethod
    def _boolean(
        value: Any,
        *,
        default: bool = False,
    ) -> bool:
        """Convert common JSON boolean representations safely."""

        if value is None:
            return default

        if isinstance(value, bool):
            return value

        if isinstance(
            value,
            (
                int,
                float,
            ),
        ):
            return bool(value)

        text = str(value).strip().casefold()

        if text in {
            "true",
            "1",
            "yes",
            "y",
        }:
            return True

        if text in {
            "false",
            "0",
            "no",
            "n",
            "",
        }:
            return False

        raise ValueError(
            f"Cannot interpret {value!r} as a boolean."
        )

    @classmethod
    def _cell_types_from_records(
        cls,
        records: Sequence[
            Mapping[str, Any]
        ],
    ) -> list[str]:
        """Derive cell-type labels from database records."""

        values: list[str] = []

        for record in records:
            value = (
                record.get(
                    "normalized_cell_type"
                )
                or record.get("cell_type")
                or record.get("cell_name")
            )

            if (
                value is not None
                and str(value).strip()
            ):
                values.append(
                    str(value).strip()
                )

        return cls._clean_strings(
            values
        )

    @classmethod
    def _record_values(
        cls,
        records: Sequence[
            Mapping[str, Any]
        ],
        keys: Sequence[str],
    ) -> list[str]:
        """Collect values from several possible record fields."""

        values: list[str] = []

        for record in records:
            for key in keys:
                value = record.get(key)

                if value is not None:
                    values.extend(
                        cls._clean_strings(
                            value
                        )
                    )

        return cls._clean_strings(
            values
        )

    @classmethod
    def _build_hpa_summary(
        cls,
        hpa_record: Mapping[str, Any],
        *,
        source: str,
        source_url: str | None,
    ) -> HPAEvidenceSummary:
        """Convert one HPA summary to the V2 schema."""

        queried = cls._boolean(
            hpa_record.get("queried"),
            default=True,
        )

        gene_present = cls._boolean(
            hpa_record.get(
                "gene_present_in_hpa"
            ),
            default=False,
        )

        breast_found = cls._boolean(
            hpa_record.get(
                "breast_expression_found"
            ),
            default=False,
        )

        top_cell_type = cls._optional_text(
            hpa_record.get("top_cell_type")
            or hpa_record.get(
                "top_breast_cell_type"
            )
        )

        top_ncpm_value = (
            hpa_record.get("top_ncpm")
            if hpa_record.get(
                "top_ncpm"
            )
            is not None
            else hpa_record.get(
                "highest_ncpm"
            )
        )

        top_ncpm = cls._optional_float(
            top_ncpm_value
        )

        second_cell_type = cls._optional_text(
            hpa_record.get(
                "second_cell_type"
            )
            or hpa_record.get(
                "second_breast_cell_type"
            )
        )

        second_ncpm_value = (
            hpa_record.get(
                "second_ncpm"
            )
            if hpa_record.get(
                "second_ncpm"
            )
            is not None
            else hpa_record.get(
                "second_highest_ncpm"
            )
        )

        second_ncpm = cls._optional_float(
            second_ncpm_value
        )

        ratio = cls._optional_float(
            hpa_record.get(
                "top_to_second_ncpm_ratio"
            )
        )

        highest_read_count_value = (
            hpa_record.get(
                "highest_read_count"
            )
            if hpa_record.get(
                "highest_read_count"
            )
            is not None
            else hpa_record.get(
                "top_cell_type_max_read_count"
            )
        )

        highest_read_count = (
            cls._optional_float(
                highest_read_count_value
            )
        )

        expression_records = cls._records(
            hpa_record.get(
                "cell_type_expression"
            )
            or hpa_record.get(
                "breast_cell_type_expression"
            )
            or hpa_record.get(
                "expression_records"
            )
            or [],
            field_name=(
                "HPA cell-type expression"
            ),
        )

        warnings = cls._clean_strings(
            hpa_record.get("warnings")
            or hpa_record.get(
                "quality_warnings"
            )
            or []
        )

        if queried:
            warnings.append(
                "HPA is complementary healthy-breast "
                "expression evidence and is not an "
                "independent marker vote."
            )

        return HPAEvidenceSummary(
            queried=queried,
            gene_present_in_hpa=(
                gene_present
            ),
            breast_expression_found=(
                breast_found
            ),
            top_cell_type=top_cell_type,
            top_ncpm=top_ncpm,
            second_cell_type=(
                second_cell_type
            ),
            second_ncpm=second_ncpm,
            top_to_second_ncpm_ratio=(
                ratio
            ),
            detected_cell_type_count=(
                cls._non_negative_int(
                    hpa_record.get(
                        "detected_cell_type_count"
                    )
                )
            ),
            highest_read_count=(
                highest_read_count
            ),
            review_pattern=(
                cls._optional_text(
                    hpa_record.get(
                        "review_pattern"
                    )
                )
            ),
            source=source,
            source_url=source_url,
            cell_type_expression=(
                expression_records
            ),
            warnings=cls._clean_strings(
                warnings
            ),
        )

    @classmethod
    def _build_gene_record(
        cls,
        gene: str,
        agent1_record: Mapping[str, Any],
        hpa_record: Mapping[str, Any],
        *,
        hpa_source: str,
        hpa_source_url: str | None,
    ) -> PanelGeneEvidence:
        """Merge Agent 1 and HPA evidence for one gene."""

        context_records = cls._records(
            agent1_record.get(
                "context_matched_database_evidence"
            )
            or [],
            field_name=(
                f"{gene} context-matched "
                "database evidence"
            ),
        )

        global_records = cls._records(
            agent1_record.get(
                "global_database_evidence"
            )
            or agent1_record.get(
                "database_evidence"
            )
            or [],
            field_name=(
                f"{gene} global database evidence"
            ),
        )

        context_cell_types = (
            cls._clean_strings(
                agent1_record.get(
                    "context_matched_cell_types"
                )
                or []
            )
        )

        if (
            not context_cell_types
            and context_records
        ):
            context_cell_types = (
                cls._cell_types_from_records(
                    context_records
                )
            )

        evidence_sources = (
            cls._clean_strings(
                agent1_record.get(
                    "evidence_sources"
                )
                or []
            )
        )

        if not evidence_sources:
            evidence_sources = (
                cls._record_values(
                    [
                        *global_records,
                        *context_records,
                    ],
                    (
                        "source",
                        "database",
                    ),
                )
            )

        tissue_contexts = (
            cls._clean_strings(
                agent1_record.get(
                    "tissue_contexts"
                )
                or agent1_record.get(
                    "tissue_specificity"
                )
                or []
            )
        )

        tissue_contexts = cls._clean_strings(
            [
                *tissue_contexts,
                *cls._record_values(
                    [
                        *global_records,
                        *context_records,
                    ],
                    (
                        "tissue",
                        "tissue_type",
                        "tissue_context",
                    ),
                ),
            ]
        )

        disease_contexts = (
            cls._clean_strings(
                agent1_record.get(
                    "disease_contexts"
                )
                or []
            )
        )

        disease_contexts = cls._clean_strings(
            [
                *disease_contexts,
                *cls._record_values(
                    [
                        *global_records,
                        *context_records,
                    ],
                    (
                        "disease",
                        "disease_context",
                        "cancer_type",
                    ),
                ),
            ]
        )

        cell_ontology_ids = (
            cls._clean_strings(
                agent1_record.get(
                    "cell_ontology_ids"
                )
                or []
            )
        )

        cell_ontology_ids = cls._clean_strings(
            [
                *cell_ontology_ids,
                *cls._record_values(
                    [
                        *global_records,
                        *context_records,
                    ],
                    (
                        "cell_ontology_id",
                        "cell_ontology_ids",
                        "cl_id",
                    ),
                ),
            ]
        )

        warnings = cls._clean_strings(
            agent1_record.get("warnings")
            or agent1_record.get(
                "quality_warnings"
            )
            or []
        )

        hpa_summary = (
            cls._build_hpa_summary(
                hpa_record,
                source=hpa_source,
                source_url=hpa_source_url,
            )
        )

        official_symbol = (
            cls._optional_text(
                agent1_record.get(
                    "official_symbol"
                )
            )
        )

        if official_symbol is not None:
            official_symbol = (
                cls._normalize_gene(
                    official_symbol
                )
            )

        marker_status = cls._optional_text(
            agent1_record.get(
                "marker_status"
            )
        )

        if marker_status is None:
            if context_cell_types:
                marker_status = (
                    "context_marker_supported"
                )
            elif global_records:
                marker_status = (
                    "global_marker_evidence_only"
                )
            else:
                marker_status = "unresolved"

        return PanelGeneEvidence(
            gene=gene,
            official_symbol=(
                official_symbol or gene
            ),
            aliases=cls._clean_strings(
                agent1_record.get("aliases")
                or []
            ),
            marker_status=marker_status,
            context_matched_cell_types=(
                context_cell_types
            ),
            context_matched_database_evidence=(
                context_records
            ),
            global_database_evidence=(
                global_records
            ),
            evidence_sources=(
                evidence_sources
            ),
            cell_ontology_ids=(
                cell_ontology_ids
            ),
            tissue_contexts=tissue_contexts,
            disease_contexts=(
                disease_contexts
            ),
            hpa=hpa_summary,
            warnings=warnings,
        )

    def build(
        self,
        agent1: Mapping[str, Any],
        hpa: Mapping[str, Any],
        *,
        expected_genes: Sequence[str] | None = None,
        source_files: Mapping[str, str] | None = None,
        require_hpa_for_all_genes: bool = True,
    ) -> PanelEvidenceCatalog:
        """Build and validate one reusable panel catalog."""

        agent1_payload = dict(agent1)
        hpa_payload = dict(hpa)

        self.loader.validate_pair(
            agent1_payload,
            hpa_payload,
        )

        input_genes_value = (
            agent1_payload.get(
                "input_genes"
            )
        )

        gene_evidence_value = (
            agent1_payload.get(
                "gene_evidence"
            )
        )

        hpa_summaries_value = (
            hpa_payload.get(
                "gene_summaries"
            )
        )

        if not isinstance(
            input_genes_value,
            list,
        ):
            raise ValueError(
                "Agent 1 input_genes must be a list."
            )

        if not isinstance(
            gene_evidence_value,
            list,
        ):
            raise ValueError(
                "Agent 1 gene_evidence must be a list."
            )

        if not isinstance(
            hpa_summaries_value,
            list,
        ):
            raise ValueError(
                "HPA gene_summaries must be a list."
            )

        input_genes = [
            self._normalize_gene(
                item.get("gene")
            )
            for item in input_genes_value
            if isinstance(
                item,
                Mapping,
            )
        ]

        if len(input_genes) != len(
            input_genes_value
        ):
            raise ValueError(
                "Every Agent 1 input gene must "
                "be a JSON object."
            )

        evidence_map = {
            self._normalize_gene(
                item.get("gene")
            ): dict(item)
            for item in gene_evidence_value
            if isinstance(
                item,
                Mapping,
            )
        }

        hpa_map = {
            self._normalize_gene(
                item.get("gene")
            ): dict(item)
            for item in hpa_summaries_value
            if isinstance(
                item,
                Mapping,
            )
        }

        if len(evidence_map) != len(
            gene_evidence_value
        ):
            raise ValueError(
                "Every Agent 1 gene evidence item "
                "must be a unique JSON object."
            )

        if len(hpa_map) != len(
            hpa_summaries_value
        ):
            raise ValueError(
                "Every HPA gene summary must be "
                "a unique JSON object."
            )

        hpa_source = str(
            hpa_payload.get(
                "source",
                "Human Protein Atlas",
            )
        ).strip()

        hpa_source_url = (
            self._optional_text(
                hpa_payload.get(
                    "source_url"
                )
            )
        )

        records = [
            self._build_gene_record(
                gene,
                evidence_map[gene],
                hpa_map[gene],
                hpa_source=(
                    hpa_source
                    or "Human Protein Atlas"
                ),
                hpa_source_url=(
                    hpa_source_url
                ),
            )
            for gene in input_genes
        ]

        unresolved_genes = sorted(
            record.gene
            for record in records
            if (
                not record.direct_marker_supported
                and not record.hpa.breast_expression_found
            )
        )

        catalog = PanelEvidenceCatalog(
            species=str(
                agent1_payload.get(
                    "species",
                    "",
                )
            ).strip(),
            tissue=str(
                agent1_payload.get(
                    "tissue",
                    "",
                )
            ).strip(),
            platform=str(
                agent1_payload.get(
                    "platform",
                    "",
                )
            ).strip(),
            panel_gene_count=len(records),
            genes=records,
            unresolved_genes=(
                unresolved_genes
            ),
            source_files=dict(
                source_files or {}
            ),
            provenance={
                "agent1_scope": str(
                    agent1_payload.get(
                        "cluster_id",
                        "panel",
                    )
                ),
                "hpa_scope": str(
                    hpa_payload.get(
                        "scope",
                        "panel",
                    )
                ),
                "hpa_source": (
                    hpa_source
                    or "Human Protein Atlas"
                ),
                "hpa_source_url": (
                    hpa_source_url
                ),
                "hpa_source_sha256": (
                    self._optional_text(
                        hpa_payload.get(
                            "source_sha256"
                        )
                    )
                ),
                "catalog_policy": (
                    "CellMarker and PanglaoDB "
                    "breast-context marker evidence "
                    "plus HPA healthy-breast expression "
                    "are precomputed once for every "
                    "unique panel gene."
                ),
            },
            warnings=[
                (
                    "HPA evidence is complementary "
                    "healthy-breast expression evidence "
                    "and has zero independent "
                    "candidate-voting weight in the "
                    "current baseline."
                ),
                (
                    "Cluster-specific DEG statistics "
                    "are loaded from the original marker "
                    "CSV and are not taken from panel "
                    "lookup transport records."
                ),
            ],
        )

        self.validator.validate(
            catalog,
            expected_genes=(
                expected_genes
                or input_genes
            ),
            require_hpa_for_all_genes=require_hpa_for_all_genes,
        )

        return catalog

    def build_from_files(
        self,
        agent1_path: str | Path,
        hpa_path: str | Path | None,
        *,
        expected_genes: Sequence[str] | None = None,
        require_hpa_for_all_genes: bool = True,
    ) -> PanelEvidenceCatalog:
        """Load source files and build the catalog."""

        agent1_source = normalize_path(
            agent1_path
        )

        expected_gene_list = (
            list(expected_genes)
            if expected_genes is not None
            else None
        )

        agent1 = self.loader.load_agent1(
            agent1_source,
            expected_genes=(
                expected_gene_list
            ),
        )

        if hpa_path is None:
            input_genes_value = agent1.get(
                "input_genes"
            )

            if not isinstance(
                input_genes_value,
                list,
            ):
                raise ValueError(
                    "Agent 1 input_genes must be a list."
                )

            genes = [
                self._normalize_gene(
                    item.get("gene")
                )
                for item in input_genes_value
                if isinstance(
                    item,
                    Mapping,
                )
            ]

            hpa = {
                "cluster_id": "panel",
                "run_mode": "marker_only",
                "query_tissue": str(
                    agent1.get("tissue", "")
                ).strip().casefold(),
                "source": "Human Protein Atlas",
                "source_release": "",
                "source_url": None,
                "source_sha256": None,
                "selected_genes": genes,
                "summary": {
                    "requested_gene_count": len(genes),
                    "hpa_enabled": False,
                    "note": (
                        "HPA was intentionally not queried "
                        "for this marker-only catalog."
                    ),
                },
                "gene_summaries": [
                    {
                        "gene": gene,
                        "queried": False,
                        "gene_present_in_hpa": False,
                        "breast_expression_found": False,
                        "source_status": "not_queried",
                        "warnings": [
                            (
                                "HPA was intentionally not queried "
                                "for this marker-only catalog."
                            )
                        ],
                    }
                    for gene in genes
                ],
                "expression_records": [],
                "warnings": [
                    (
                        "Marker-only catalog: HPA evidence "
                        "is handled by a separate validation workflow."
                    )
                ],
            }

            source_file_map = {
                "agent1_panel_json": str(
                    agent1_source
                ),
                "hpa_panel_json": "not_queried_marker_only",
            }

        else:
            hpa_source = normalize_path(
                hpa_path
            )

            hpa = self.loader.load_hpa(
                hpa_source,
                expected_genes=(
                    expected_gene_list
                ),
            )

            source_file_map = {
                "agent1_panel_json": str(
                    agent1_source
                ),
                "hpa_panel_json": str(
                    hpa_source
                ),
            }

        return self.build(
            agent1,
            hpa,
            expected_genes=expected_genes,
            source_files=source_file_map,
            require_hpa_for_all_genes=(
                require_hpa_for_all_genes
            ),
        )

    @staticmethod
    def _summary_rows(
        catalog: PanelEvidenceCatalog,
    ) -> list[dict[str, Any]]:
        """Create a flat one-row-per-gene table."""

        unresolved_set = set(
            catalog.unresolved_genes
        )

        rows: list[dict[str, Any]] = []

        for record in catalog.genes:
            combined_warnings = sorted(
                set(
                    [
                        *record.warnings,
                        *record.hpa.warnings,
                    ]
                ),
                key=str.casefold,
            )

            rows.append(
                {
                    "gene": record.gene,
                    "official_symbol": (
                        record.official_symbol
                        or ""
                    ),
                    "aliases": " | ".join(
                        record.aliases
                    ),
                    "marker_status": (
                        record.marker_status
                    ),
                    "direct_marker_supported": (
                        record.direct_marker_supported
                    ),
                    "context_matched_cell_types": (
                        " | ".join(
                            record.context_matched_cell_types
                        )
                    ),
                    "context_evidence_count": (
                        len(
                            record.context_matched_database_evidence
                        )
                    ),
                    "global_evidence_count": (
                        len(
                            record.global_database_evidence
                        )
                    ),
                    "evidence_sources": (
                        " | ".join(
                            record.evidence_sources
                        )
                    ),
                    "tissue_contexts": (
                        " | ".join(
                            record.tissue_contexts
                        )
                    ),
                    "disease_contexts": (
                        " | ".join(
                            record.disease_contexts
                        )
                    ),
                    "cell_ontology_ids": (
                        " | ".join(
                            record.cell_ontology_ids
                        )
                    ),
                    "hpa_queried": (
                        record.hpa.queried
                    ),
                    "hpa_gene_present": (
                        record.hpa.gene_present_in_hpa
                    ),
                    "hpa_breast_expression_found": (
                        record.hpa.breast_expression_found
                    ),
                    "hpa_top_cell_type": (
                        record.hpa.top_cell_type
                        or ""
                    ),
                    "hpa_top_ncpm": (
                        record.hpa.top_ncpm
                    ),
                    "hpa_second_cell_type": (
                        record.hpa.second_cell_type
                        or ""
                    ),
                    "hpa_second_ncpm": (
                        record.hpa.second_ncpm
                    ),
                    "hpa_top_to_second_ratio": (
                        record.hpa.top_to_second_ncpm_ratio
                    ),
                    "hpa_detected_cell_type_count": (
                        record.hpa.detected_cell_type_count
                    ),
                    "hpa_highest_read_count": (
                        record.hpa.highest_read_count
                    ),
                    "unresolved_for_current_sources": (
                        record.gene
                        in unresolved_set
                    ),
                    "warnings": " | ".join(
                        combined_warnings
                    ),
                }
            )

        return rows

    @staticmethod
    def _write_csv(
        path: Path,
        rows: list[dict[str, Any]],
    ) -> None:
        """Write dictionaries to CSV safely."""

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = (
            list(rows[0])
            if rows
            else ["gene"]
        )

        with path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

    def write(
        self,
        catalog: PanelEvidenceCatalog,
        *,
        json_path: str | Path | None = None,
        csv_path: str | Path | None = None,
        unresolved_path: str | Path | None = None,
        require_hpa_for_all_genes: bool = True,
    ) -> dict[str, Path]:
        """Write catalog JSON, CSV, unresolved genes, and validation."""

        validation: CatalogValidationResult = (
            self.validator.validate(
                catalog,
                expected_genes=[
                    record.gene
                    for record in catalog.genes
                ],
                require_hpa_for_all_genes=(
                    require_hpa_for_all_genes
                ),
            )
        )

        resolved_json = (
            normalize_path(json_path)
            if json_path is not None
            else PANEL_CATALOG_JSON
        )

        resolved_csv = (
            normalize_path(csv_path)
            if csv_path is not None
            else PANEL_CATALOG_CSV
        )

        resolved_unresolved = (
            normalize_path(
                unresolved_path
            )
            if unresolved_path is not None
            else PANEL_UNRESOLVED_GENES_CSV
        )

        resolved_json.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        resolved_json.write_text(
            catalog.model_dump_json(
                indent=2
            ),
            encoding="utf-8",
        )

        self._write_csv(
            resolved_csv,
            self._summary_rows(
                catalog
            ),
        )

        unresolved_set = set(
            catalog.unresolved_genes
        )

        unresolved_rows = [
            {
                "gene": record.gene,
                "marker_status": (
                    record.marker_status
                ),
                "hpa_gene_present": (
                    record.hpa.gene_present_in_hpa
                ),
                "hpa_breast_expression_found": (
                    record.hpa.breast_expression_found
                ),
                "recommended_next_step": (
                    "authoritative_symbol_review"
                    if (
                        record.hpa.queried
                        and not record.hpa.gene_present_in_hpa
                    )
                    else "targeted_manual_source_review"
                ),
            }
            for record in catalog.genes
            if record.gene in unresolved_set
        ]

        self._write_csv(
            resolved_unresolved,
            unresolved_rows,
        )

        validation_path = (
            resolved_json.with_name(
                "panel_evidence_catalog_validation.json"
            )
        )

        validation_path.write_text(
            json.dumps(
                validation.to_dict(),
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "panel_catalog_json": (
                resolved_json
            ),
            "panel_catalog_csv": (
                resolved_csv
            ),
            "unresolved_panel_genes_csv": (
                resolved_unresolved
            ),
            "panel_catalog_validation_json": (
                validation_path
            ),
        }