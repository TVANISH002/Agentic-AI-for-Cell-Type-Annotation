from __future__ import annotations

import csv
import hashlib
import io
import math
import zipfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from hpa_evidence_agent.config import (
    DEFAULT_CONFIG,
    HPAEvidenceConfig,
)


@dataclass(frozen=True, slots=True)
class HPASourceRecord:
    """One selected positive HPA breast-expression source row."""

    gene: str
    ensembl_gene_id: str
    tissue: str
    cell_type: str
    cluster: str
    read_count: float
    ncpm: float


@dataclass(frozen=True, slots=True)
class HPASourceLoadResult:
    """Filtered HPA source data and source-file provenance."""

    selected_genes: tuple[str, ...]
    genes_present_in_hpa: frozenset[str]
    breast_expression_records: tuple[HPASourceRecord, ...]
    source_path: Path
    source_url: str
    source_sha256: str
    source_file_size_bytes: int
    archive_member_name: str
    scanned_row_count: int
    matched_panel_row_count: int


class HPASourceLoader:
    """Stream the HPA source and retain panel-gene breast evidence."""

    SOURCE_URL = (
        "https://www.proteinatlas.org/download/tsv/"
        "rna_single_cell_cluster.tsv.zip"
    )

    def __init__(
        self,
        config: HPAEvidenceConfig = DEFAULT_CONFIG,
    ) -> None:
        """Store and validate the HPA workflow configuration."""

        config.validate()
        self.config = config

    @staticmethod
    def _normalize_selected_genes(
        selected_genes: Sequence[str],
    ) -> tuple[str, ...]:
        """Normalize panel symbols and reject empty or duplicate genes."""

        normalized: list[str] = []
        first_position: dict[str, int] = {}

        for position, value in enumerate(
            selected_genes,
            start=1,
        ):
            gene = str(value).strip().upper()

            if not gene:
                raise ValueError(
                    "selected_genes contains an empty value "
                    f"at position {position}."
                )

            if gene in first_position:
                raise ValueError(
                    "selected_genes contains a duplicate gene "
                    f"after normalization: {gene}. "
                    "First position "
                    f"{first_position[gene]}, repeated at "
                    f"position {position}."
                )

            first_position[gene] = position
            normalized.append(gene)

        if not normalized:
            raise ValueError(
                "selected_genes cannot be empty."
            )

        return tuple(normalized)

    def _required_columns(self) -> tuple[str, ...]:
        """Return the HPA columns required by this workflow."""

        return (
            self.config.hpa_gene_column,
            self.config.hpa_ensembl_column,
            self.config.hpa_tissue_column,
            self.config.hpa_cell_type_column,
            self.config.hpa_cluster_column,
            self.config.hpa_read_count_column,
            self.config.hpa_ncpm_column,
        )

    @staticmethod
    def _compute_sha256(
        path: Path,
    ) -> str:
        """Compute the source-file checksum without loading it into memory."""

        digest = hashlib.sha256()

        with path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)

        return digest.hexdigest()

    @staticmethod
    def _select_archive_member(
        archive: zipfile.ZipFile,
    ) -> str:
        """Select the single TSV file stored inside the HPA archive."""

        members = [
            info.filename
            for info in archive.infolist()
            if (
                not info.is_dir()
                and info.filename.lower().endswith(
                    (".tsv", ".txt")
                )
            )
        ]

        if not members:
            raise ValueError(
                "The HPA ZIP archive does not contain "
                "a TSV or TXT data file."
            )

        if len(members) != 1:
            member_text = ", ".join(
                sorted(members)
            )

            raise ValueError(
                "The HPA ZIP archive must contain exactly "
                "one TSV/TXT data file. Found: "
                f"{member_text}."
            )

        return members[0]

    @contextmanager
    def _open_source_text(
        self,
        path: Path,
    ) -> Iterator[tuple[TextIO, str]]:
        """Open either a zipped or plain HPA TSV as UTF-8 text."""

        if path.suffix.casefold() == ".zip":
            with zipfile.ZipFile(
                path,
                "r",
            ) as archive:
                member_name = (
                    self._select_archive_member(
                        archive
                    )
                )

                with archive.open(
                    member_name,
                    "r",
                ) as binary_file:
                    with io.TextIOWrapper(
                        binary_file,
                        encoding="utf-8-sig",
                        newline="",
                    ) as text_file:
                        yield (
                            text_file,
                            member_name,
                        )

            return

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as text_file:
            yield (
                text_file,
                path.name,
            )

    @staticmethod
    def _parse_nonnegative_float(
        value: str | None,
        *,
        field_name: str,
        row_number: int,
    ) -> float:
        """Parse one finite, non-negative numeric HPA value."""

        raw_value = (
            ""
            if value is None
            else value.strip()
        )

        if not raw_value:
            raise ValueError(
                f"{field_name} is empty at HPA row "
                f"{row_number}."
            )

        try:
            parsed = float(raw_value)
        except ValueError as error:
            raise ValueError(
                f"{field_name} is not numeric at HPA row "
                f"{row_number}: {raw_value!r}."
            ) from error

        if not math.isfinite(parsed):
            raise ValueError(
                f"{field_name} must be finite at HPA row "
                f"{row_number}."
            )

        if parsed < 0:
            raise ValueError(
                f"{field_name} cannot be negative at HPA row "
                f"{row_number}."
            )

        return parsed

    @staticmethod
    def _required_text(
        value: str | None,
        *,
        field_name: str,
        row_number: int,
    ) -> str:
        """Validate one required text value from a retained HPA row."""

        cleaned = (
            ""
            if value is None
            else value.strip()
        )

        if not cleaned:
            raise ValueError(
                f"{field_name} is empty at HPA row "
                f"{row_number}."
            )

        return cleaned

    def load(
        self,
        selected_genes: Sequence[str],
        *,
        path: str | Path | None = None,
    ) -> HPASourceLoadResult:
        """Scan HPA once and retain evidence for every selected gene."""

        normalized_genes = (
            self._normalize_selected_genes(
                selected_genes
            )
        )

        selected_gene_set = set(
            normalized_genes
        )

        resolved_path = (
            Path(path)
            if path is not None
            else self.config.hpa_source_path
        )

        if not resolved_path.exists():
            raise FileNotFoundError(
                "HPA source file was not found: "
                f"{resolved_path.resolve()}"
            )

        if not resolved_path.is_file():
            raise ValueError(
                "HPA source path is not a file: "
                f"{resolved_path.resolve()}"
            )

        source_size = (
            resolved_path.stat().st_size
        )

        if source_size <= 0:
            raise ValueError(
                "HPA source file is empty: "
                f"{resolved_path.resolve()}"
            )

        query_tissue = (
            self.config.query_tissue
            .strip()
            .casefold()
        )

        minimum_ncpm = (
            self.config.minimum_positive_ncpm
        )

        genes_present: set[str] = set()

        breast_records: list[
            HPASourceRecord
        ] = []

        scanned_row_count = 0
        matched_panel_row_count = 0
        archive_member_name = (
            resolved_path.name
        )

        with self._open_source_text(
            resolved_path
        ) as (
            text_file,
            archive_member_name,
        ):
            reader = csv.DictReader(
                text_file,
                delimiter="\t",
            )

            if reader.fieldnames is None:
                raise ValueError(
                    "HPA source does not contain "
                    "a header row."
                )

            normalized_headers = {
                header.strip(): header
                for header in reader.fieldnames
                if header is not None
            }

            missing_columns = [
                column
                for column
                in self._required_columns()
                if column
                not in normalized_headers
            ]

            if missing_columns:
                available_columns = ", ".join(
                    sorted(
                        normalized_headers
                    )
                )

                missing_text = ", ".join(
                    missing_columns
                )

                raise ValueError(
                    "HPA source is missing required "
                    f"columns: {missing_text}. "
                    "Available columns: "
                    f"{available_columns or 'none'}."
                )

            source_columns = {
                column: normalized_headers[
                    column
                ]
                for column
                in self._required_columns()
            }

            for row_number, row in enumerate(
                reader,
                start=2,
            ):
                scanned_row_count += 1

                raw_gene = row.get(
                    source_columns[
                        self.config.hpa_gene_column
                    ]
                )

                gene = (
                    ""
                    if raw_gene is None
                    else raw_gene.strip().upper()
                )

                if gene not in selected_gene_set:
                    continue

                matched_panel_row_count += 1
                genes_present.add(gene)

                raw_tissue = row.get(
                    source_columns[
                        self.config.hpa_tissue_column
                    ]
                )

                tissue = (
                    ""
                    if raw_tissue is None
                    else raw_tissue.strip()
                )

                if (
                    tissue.casefold()
                    != query_tissue
                ):
                    continue

                ncpm = (
                    self._parse_nonnegative_float(
                        row.get(
                            source_columns[
                                self.config
                                .hpa_ncpm_column
                            ]
                        ),
                        field_name=(
                            self.config
                            .hpa_ncpm_column
                        ),
                        row_number=row_number,
                    )
                )

                if ncpm <= minimum_ncpm:
                    continue

                read_count = (
                    self._parse_nonnegative_float(
                        row.get(
                            source_columns[
                                self.config
                                .hpa_read_count_column
                            ]
                        ),
                        field_name=(
                            self.config
                            .hpa_read_count_column
                        ),
                        row_number=row_number,
                    )
                )

                breast_records.append(
                    HPASourceRecord(
                        gene=gene,
                        ensembl_gene_id=(
                            self._required_text(
                                row.get(
                                    source_columns[
                                        self.config
                                        .hpa_ensembl_column
                                    ]
                                ),
                                field_name=(
                                    self.config
                                    .hpa_ensembl_column
                                ),
                                row_number=(
                                    row_number
                                ),
                            )
                        ),
                        tissue=tissue,
                        cell_type=(
                            self._required_text(
                                row.get(
                                    source_columns[
                                        self.config
                                        .hpa_cell_type_column
                                    ]
                                ),
                                field_name=(
                                    self.config
                                    .hpa_cell_type_column
                                ),
                                row_number=(
                                    row_number
                                ),
                            )
                        ),
                        cluster=(
                            self._required_text(
                                row.get(
                                    source_columns[
                                        self.config
                                        .hpa_cluster_column
                                    ]
                                ),
                                field_name=(
                                    self.config
                                    .hpa_cluster_column
                                ),
                                row_number=(
                                    row_number
                                ),
                            )
                        ),
                        read_count=read_count,
                        ncpm=ncpm,
                    )
                )

        return HPASourceLoadResult(
            selected_genes=normalized_genes,
            genes_present_in_hpa=frozenset(
                genes_present
            ),
            breast_expression_records=tuple(
                breast_records
            ),
            source_path=(
                resolved_path.resolve()
            ),
            source_url=self.SOURCE_URL,
            source_sha256=(
                self._compute_sha256(
                    resolved_path
                )
            ),
            source_file_size_bytes=(
                source_size
            ),
            archive_member_name=(
                archive_member_name
            ),
            scanned_row_count=(
                scanned_row_count
            ),
            matched_panel_row_count=(
                matched_panel_row_count
            ),
        )


def load_hpa_source(
    selected_genes: Sequence[str],
    config: HPAEvidenceConfig = DEFAULT_CONFIG,
    *,
    path: str | Path | None = None,
) -> HPASourceLoadResult:
    """Convenience function for loading independent HPA evidence."""

    return HPASourceLoader(
        config
    ).load(
        selected_genes,
        path=path,
    )