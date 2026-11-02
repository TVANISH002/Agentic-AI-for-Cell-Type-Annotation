from __future__ import annotations

import csv
from pathlib import Path

from hpa_evidence_agent.config import (
    DEFAULT_CONFIG,
    HPAEvidenceConfig,
)


class PanelGeneLoader:
    """Load the canonical cleaned panel-gene list."""

    def __init__(
        self,
        config: HPAEvidenceConfig = DEFAULT_CONFIG,
    ) -> None:
        """Store and validate the HPA workflow configuration."""

        config.validate()
        self.config = config

    @staticmethod
    def _normalize_gene(
        value: str,
        *,
        row_number: int,
    ) -> str:
        """Normalize one gene symbol and reject empty values."""

        normalized = value.strip().upper()

        if not normalized:
            raise ValueError(
                "Panel gene is empty at CSV row "
                f"{row_number}."
            )

        return normalized

    def load(
        self,
        path: str | Path | None = None,
    ) -> list[str]:
        """Read unique panel genes while preserving CSV order."""

        resolved_path = (
            Path(path)
            if path is not None
            else self.config.panel_gene_list_path
        )

        if not resolved_path.exists():
            raise FileNotFoundError(
                "Panel gene-list file was not found: "
                f"{resolved_path.resolve()}"
            )

        if not resolved_path.is_file():
            raise ValueError(
                "Panel gene-list path is not a file: "
                f"{resolved_path.resolve()}"
            )

        gene_column = self.config.panel_gene_column
        genes: list[str] = []
        first_row_by_gene: dict[str, int] = {}

        with resolved_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                raise ValueError(
                    "Panel gene-list CSV does not contain "
                    "a header row."
                )

            normalized_headers = {
                header.strip(): header
                for header in reader.fieldnames
                if header is not None
            }

            if gene_column not in normalized_headers:
                available_columns = ", ".join(
                    sorted(normalized_headers)
                )

                raise ValueError(
                    "Panel gene-list CSV is missing the "
                    f"required '{gene_column}' column. "
                    "Available columns: "
                    f"{available_columns or 'none'}."
                )

            source_gene_column = normalized_headers[
                gene_column
            ]

            for row_number, row in enumerate(
                reader,
                start=2,
            ):
                raw_gene = row.get(
                    source_gene_column
                )

                if raw_gene is None:
                    raise ValueError(
                        "Panel gene column is missing "
                        f"at CSV row {row_number}."
                    )

                gene = self._normalize_gene(
                    raw_gene,
                    row_number=row_number,
                )

                if gene in first_row_by_gene:
                    raise ValueError(
                        "Duplicate panel gene after "
                        f"normalization: {gene}. "
                        "First seen at CSV row "
                        f"{first_row_by_gene[gene]}, "
                        "repeated at CSV row "
                        f"{row_number}."
                    )

                first_row_by_gene[gene] = (
                    row_number
                )

                genes.append(gene)

        if not genes:
            raise ValueError(
                "Panel gene-list CSV does not "
                "contain any genes."
            )

        return genes


def load_panel_genes(
    config: HPAEvidenceConfig = DEFAULT_CONFIG,
    *,
    path: str | Path | None = None,
) -> list[str]:
    """Convenience function for loading the canonical panel genes."""

    return PanelGeneLoader(
        config
    ).load(
        path
    )