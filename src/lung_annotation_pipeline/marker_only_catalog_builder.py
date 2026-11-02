from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from marker_interpretation_agent_v2.panel.panel_catalog_validator import (
    PanelCatalogValidator,
)
from marker_interpretation_agent_v2.panel.panel_evidence_builder import (
    PanelEvidenceBuilder,
)
from marker_interpretation_agent_v2.paths import normalize_path
from marker_interpretation_agent_v2.schemas import PanelEvidenceCatalog


class MarkerOnlyPanelCatalogBuilder(PanelEvidenceBuilder):
    """Build an Agent 2 catalog from Agent 1 without querying HPA."""

    def build_from_agent1_file(
        self,
        agent1_path: str | Path,
        *,
        expected_genes: Sequence[str],
    ) -> PanelEvidenceCatalog:
        """Load genuine Agent 1 evidence and add unqueried HPA fields."""

        agent1_source = normalize_path(agent1_path)

        expected_gene_list = [
            self._normalize_gene(gene)
            for gene in expected_genes
        ]

        payload = self.loader.load_agent1(
            agent1_source,
            expected_genes=expected_gene_list,
        )

        input_records = payload.get("input_genes")
        evidence_records = payload.get("gene_evidence")

        if not isinstance(input_records, list):
            raise ValueError(
                "Agent 1 input_genes must be a list."
            )

        if not isinstance(evidence_records, list):
            raise ValueError(
                "Agent 1 gene_evidence must be a list."
            )

        input_genes: list[str] = []

        for index, item in enumerate(
            input_records,
            start=1,
        ):
            if not isinstance(item, Mapping):
                raise ValueError(
                    "Agent 1 input_genes item "
                    f"{index} must be a JSON object."
                )

            input_genes.append(
                self._normalize_gene(
                    item.get("gene")
                )
            )

        evidence_map: dict[
            str,
            dict[str, Any],
        ] = {}

        for index, item in enumerate(
            evidence_records,
            start=1,
        ):
            if not isinstance(item, Mapping):
                raise ValueError(
                    "Agent 1 gene_evidence item "
                    f"{index} must be a JSON object."
                )

            gene = self._normalize_gene(
                item.get("gene")
            )

            if gene in evidence_map:
                raise ValueError(
                    "Agent 1 gene_evidence contains "
                    f"duplicate gene {gene!r}."
                )

            evidence_map[gene] = dict(item)

        if input_genes != expected_gene_list:
            raise ValueError(
                "Agent 1 input gene order does not match "
                "the expected panel order."
            )

        if set(evidence_map) != set(input_genes):
            missing = sorted(
                set(input_genes) - set(evidence_map)
            )

            extra = sorted(
                set(evidence_map) - set(input_genes)
            )

            raise ValueError(
                "Agent 1 gene evidence does not match "
                "its input genes. "
                f"Missing: {missing}; extra: {extra}."
            )

        records = [
            self._build_gene_record(
                gene,
                evidence_map[gene],
                {
                    "gene": gene,
                    "queried": False,
                },
                hpa_source="Human Protein Atlas",
                hpa_source_url=None,
            )
            for gene in input_genes
        ]

        catalog = PanelEvidenceCatalog(
            species=str(
                payload.get(
                    "species",
                    "",
                )
            ).strip(),
            tissue=str(
                payload.get(
                    "tissue",
                    "",
                )
            ).strip(),
            platform=str(
                payload.get(
                    "platform",
                    "",
                )
            ).strip(),
            panel_gene_count=len(records),
            genes=records,
            unresolved_genes=sorted(
                record.gene
                for record in records
                if not record.direct_marker_supported
            ),
            source_files={
                "agent1_panel_json": str(
                    agent1_source
                ),
            },
            provenance={
                "agent1_scope": str(
                    payload.get(
                        "cluster_id",
                        "panel",
                    )
                ),
                "evidence_mode": (
                    "cellmarker_panglaodb_only"
                ),
                "hpa_enabled": False,
                "catalog_policy": (
                    "CellMarker and PanglaoDB "
                    "context-matched marker evidence "
                    "is reused across clusters. "
                    "HPA was not queried."
                ),
            },
            warnings=[
                (
                    "HPA was intentionally disabled "
                    "for this lung dataset run."
                ),
                (
                    "Cluster-specific DEG statistics "
                    "come from the processed lung marker "
                    "CSV, not from Agent 1 panel "
                    "transport values."
                ),
            ],
        )

        self.validator.validate(
            catalog,
            expected_genes=expected_gene_list,
            require_hpa_for_all_genes=False,
        )

        return catalog

    def write_marker_only(
        self,
        catalog: PanelEvidenceCatalog,
        *,
        json_path: str | Path,
        csv_path: str | Path,
        unresolved_path: str | Path,
    ) -> dict[str, Path]:
        """Write the marker-only catalog and validation outputs."""

        expected_genes = [
            record.gene
            for record in catalog.genes
        ]

        validation = PanelCatalogValidator().validate(
            catalog,
            expected_genes=expected_genes,
            require_hpa_for_all_genes=False,
        )

        resolved_json = normalize_path(
            json_path
        )

        resolved_csv = normalize_path(
            csv_path
        )

        resolved_unresolved = normalize_path(
            unresolved_path
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
            self._summary_rows(catalog),
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
                "hpa_queried": (
                    record.hpa.queried
                ),
                "recommended_next_step": (
                    "targeted_manual_source_review"
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
            "panel_catalog_json": resolved_json,
            "panel_catalog_csv": resolved_csv,
            "unresolved_panel_genes_csv": (
                resolved_unresolved
            ),
            "panel_catalog_validation_json": (
                validation_path
            ),
        }