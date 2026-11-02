from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from marker_interpretation_agent_v2.schemas import PanelEvidenceCatalog


class PanelEvidenceLoader:
    """Load and validate panel-wide Agent 1, HPA, and catalog JSON files."""

    AGENT1_REQUIRED_KEYS = {
        "cluster_id",
        "species",
        "tissue",
        "platform",
        "input_genes",
        "gene_evidence",
    }

    HPA_REQUIRED_KEYS = {
        "source",
        "selected_genes",
        "gene_summaries",
    }

    @staticmethod
    def _normalize_gene(value: Any) -> str:
        """Normalize one gene symbol."""

        gene = str(value).strip().upper()

        if not gene:
            raise ValueError("Gene symbol cannot be empty.")

        return gene

    @staticmethod
    def _read_json(path: str | Path) -> dict[str, Any]:
        """Read one top-level JSON object."""

        source = Path(path).expanduser().resolve()

        if not source.exists():
            raise FileNotFoundError(
                f"JSON file does not exist: {source}"
            )

        try:
            with source.open(
                "r",
                encoding="utf-8",
            ) as file:
                payload: Any = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON file: {source}"
            ) from error

        if not isinstance(payload, dict):
            raise ValueError(
                f"Expected one top-level JSON object in {source}."
            )

        return payload

    @staticmethod
    def _require_keys(
        payload: dict[str, Any],
        required: set[str],
        *,
        source_name: str,
    ) -> None:
        """Require top-level JSON keys."""

        missing = sorted(
            required - set(payload)
        )

        if missing:
            raise ValueError(
                f"{source_name} is missing required keys: "
                f"{missing}"
            )

    @classmethod
    def _genes_from_records(
        cls,
        records: list[Any],
        *,
        source_name: str,
    ) -> list[str]:
        """Extract unique normalized genes from JSON records."""

        genes: list[str] = []

        for index, record in enumerate(
            records,
            start=1,
        ):
            if not isinstance(record, dict):
                raise ValueError(
                    f"{source_name} item {index} must "
                    "be a JSON object."
                )

            genes.append(
                cls._normalize_gene(
                    record.get("gene", "")
                )
            )

        duplicates = sorted(
            gene
            for gene, count in Counter(
                genes
            ).items()
            if count > 1
        )

        if duplicates:
            raise ValueError(
                f"{source_name} contains duplicate genes: "
                f"{duplicates}"
            )

        return genes

    @classmethod
    def _validate_expected_genes(
        cls,
        actual_genes: list[str],
        expected_genes: list[str] | None,
        *,
        source_name: str,
    ) -> None:
        """Require an exact match with the panel gene list."""

        if expected_genes is None:
            return

        expected = [
            cls._normalize_gene(gene)
            for gene in expected_genes
        ]

        duplicate_expected = sorted(
            gene
            for gene, count in Counter(
                expected
            ).items()
            if count > 1
        )

        if duplicate_expected:
            raise ValueError(
                "Expected panel gene list contains "
                "duplicates: "
                f"{duplicate_expected}"
            )

        missing = sorted(
            set(expected) - set(actual_genes)
        )

        extra = sorted(
            set(actual_genes) - set(expected)
        )

        if missing or extra:
            raise ValueError(
                f"{source_name} does not match the "
                "panel genes. "
                f"Missing: {missing}; extra: {extra}."
            )

    def load_agent1(
        self,
        path: str | Path,
        *,
        expected_genes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Load and validate panel-wide Agent 1 evidence."""

        payload = self._read_json(path)

        self._require_keys(
            payload,
            self.AGENT1_REQUIRED_KEYS,
            source_name="Agent 1 JSON",
        )

        input_genes_value = payload[
            "input_genes"
        ]

        gene_evidence_value = payload[
            "gene_evidence"
        ]

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

        if not input_genes_value:
            raise ValueError(
                "Agent 1 JSON contains no input genes."
            )

        input_genes = self._genes_from_records(
            input_genes_value,
            source_name="Agent 1 input_genes",
        )

        evidence_genes = self._genes_from_records(
            gene_evidence_value,
            source_name="Agent 1 gene_evidence",
        )

        if input_genes != evidence_genes:
            raise ValueError(
                "Agent 1 input_genes and "
                "gene_evidence must have the same "
                "normalized order."
            )

        self._validate_expected_genes(
            input_genes,
            expected_genes,
            source_name="Agent 1 JSON",
        )

        return payload

    def load_hpa(
        self,
        path: str | Path,
        *,
        expected_genes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Load and validate panel-wide HPA evidence."""

        payload = self._read_json(path)

        self._require_keys(
            payload,
            self.HPA_REQUIRED_KEYS,
            source_name="HPA JSON",
        )

        selected_value = payload[
            "selected_genes"
        ]

        summaries_value = payload[
            "gene_summaries"
        ]

        if not isinstance(
            selected_value,
            list,
        ):
            raise ValueError(
                "HPA selected_genes must be a list."
            )

        if not isinstance(
            summaries_value,
            list,
        ):
            raise ValueError(
                "HPA gene_summaries must be a list."
            )

        selected_genes = [
            self._normalize_gene(gene)
            for gene in selected_value
        ]

        duplicate_selected = sorted(
            gene
            for gene, count in Counter(
                selected_genes
            ).items()
            if count > 1
        )

        if duplicate_selected:
            raise ValueError(
                "HPA selected_genes contains "
                "duplicate genes: "
                f"{duplicate_selected}"
            )

        summary_genes = self._genes_from_records(
            summaries_value,
            source_name="HPA gene_summaries",
        )

        if set(selected_genes) != set(
            summary_genes
        ):
            missing = sorted(
                set(selected_genes)
                - set(summary_genes)
            )

            extra = sorted(
                set(summary_genes)
                - set(selected_genes)
            )

            raise ValueError(
                "HPA selected_genes and "
                "gene_summaries disagree. "
                f"Missing summaries: {missing}; "
                f"extra summaries: {extra}."
            )

        self._validate_expected_genes(
            summary_genes,
            expected_genes,
            source_name="HPA JSON",
        )

        return payload

    def validate_pair(
        self,
        agent1: dict[str, Any],
        hpa: dict[str, Any],
    ) -> None:
        """Require Agent 1 and HPA to cover the same genes."""

        input_genes_value = agent1.get(
            "input_genes"
        )

        summaries_value = hpa.get(
            "gene_summaries"
        )

        if not isinstance(
            input_genes_value,
            list,
        ):
            raise ValueError(
                "Agent 1 input_genes must be a list."
            )

        if not isinstance(
            summaries_value,
            list,
        ):
            raise ValueError(
                "HPA gene_summaries must be a list."
            )

        agent1_genes = self._genes_from_records(
            input_genes_value,
            source_name="Agent 1 input_genes",
        )

        hpa_genes = self._genes_from_records(
            summaries_value,
            source_name="HPA gene_summaries",
        )

        missing_hpa = sorted(
            set(agent1_genes)
            - set(hpa_genes)
        )

        extra_hpa = sorted(
            set(hpa_genes)
            - set(agent1_genes)
        )

        if missing_hpa or extra_hpa:
            raise ValueError(
                "Agent 1 and HPA panel genes "
                "do not match. "
                f"Missing HPA genes: {missing_hpa}; "
                f"extra HPA genes: {extra_hpa}."
            )

    def load_catalog(
        self,
        path: str | Path,
    ) -> PanelEvidenceCatalog:
        """Load a completed reusable panel evidence catalog."""

        payload = self._read_json(path)

        return PanelEvidenceCatalog.model_validate(
            payload
        )