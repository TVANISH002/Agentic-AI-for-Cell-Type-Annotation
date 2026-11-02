from __future__ import annotations

import csv
import hashlib
import zipfile
from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from datetime import datetime, timezone
from io import TextIOWrapper
from pathlib import Path
from typing import Literal, TypeAlias

from .hpa_schemas import (
    HPACellTypeExpressionSummary,
    HPAClusterExpressionRecord,
    HPAGeneExpressionSummary,
    HPARetrievalResult,
    HPARunSummary,
)


HPARunMode: TypeAlias = Literal[
    "pilot",
    "all",
]

HPAStatus: TypeAlias = Literal[
    "breast_expression_found",
    "gene_found_without_breast_expression",
    "gene_not_found_in_hpa",
]


class HPASingleCellRetriever:
    REQUIRED_COLUMNS = {
        "Gene",
        "Gene name",
        "Tissue",
        "Cell type",
        "Cluster",
        "Read count",
        "nCPM",
    }

    def __init__(
        self,
        source_url: str,
        source_release: str,
    ) -> None:
        """Initializes the HPA retriever."""

        if not source_url.strip():
            raise ValueError(
                "source_url must not be empty."
            )

        if not source_release.strip():
            raise ValueError(
                "source_release must not be empty."
            )

        self.source_url = source_url
        self.source_release = source_release

    def retrieve(
        self,
        reference_path: Path,
        genes: list[str],
        tissue: str,
        cluster_id: int | str,
        run_mode: HPARunMode,
        progress_callback: Callable[
            [str],
            None,
        ]
        | None = None,
    ) -> HPARetrievalResult:
        """Retrieves HPA breast-expression evidence."""

        selected_genes = self._clean_genes(
            genes
        )

        cleaned_tissue = tissue.strip()

        if not cleaned_tissue:
            raise ValueError(
                "tissue must not be empty."
            )

        if not reference_path.exists():
            raise FileNotFoundError(
                f"HPA reference file not found: "
                f"{reference_path}"
            )

        if not reference_path.is_file():
            raise ValueError(
                f"HPA reference path is not a file: "
                f"{reference_path}"
            )

        if not zipfile.is_zipfile(
            reference_path
        ):
            raise ValueError(
                f"HPA reference is not a valid ZIP: "
                f"{reference_path}"
            )

        target_by_upper = {
            gene.upper(): gene
            for gene in selected_genes
        }

        gene_present_anywhere = {
            gene: False
            for gene in selected_genes
        }

        records_by_gene: dict[
            str,
            list[HPAClusterExpressionRecord],
        ] = {
            gene: []
            for gene in selected_genes
        }

        self._notify(
            progress_callback,
            "Opening the local HPA ZIP file...",
        )

        row_count = 0

        with zipfile.ZipFile(
            reference_path
        ) as archive:
            tsv_member = self._find_tsv_member(
                archive
            )

            self._notify(
                progress_callback,
                f"Reading HPA table: {tsv_member}",
            )

            with archive.open(
                tsv_member
            ) as raw_handle:
                with TextIOWrapper(
                    raw_handle,
                    encoding="utf-8",
                    newline="",
                ) as text_handle:
                    reader = csv.DictReader(
                        text_handle,
                        delimiter="\t",
                    )

                    self._validate_header(
                        reader.fieldnames
                    )

                    for row in reader:
                        row_count += 1

                        if row_count % 1_000_000 == 0:
                            self._notify(
                                progress_callback,
                                (
                                    "Processed "
                                    f"{row_count:,} HPA rows..."
                                ),
                            )

                        gene_name = self._required_value(
                            row,
                            "Gene name",
                        )

                        selected_gene = (
                            target_by_upper.get(
                                gene_name.upper()
                            )
                        )

                        if selected_gene is None:
                            continue

                        gene_present_anywhere[
                            selected_gene
                        ] = True

                        row_tissue = self._required_value(
                            row,
                            "Tissue",
                        )

                        if (
                            row_tissue.casefold()
                            != cleaned_tissue.casefold()
                        ):
                            continue

                        ncpm = self._parse_nonnegative_float(
                            self._required_value(
                                row,
                                "nCPM",
                            ),
                            field_name="nCPM",
                            gene=selected_gene,
                        )

                        if ncpm <= 0:
                            continue

                        read_count = (
                            self._parse_nonnegative_float(
                                self._required_value(
                                    row,
                                    "Read count",
                                ),
                                field_name="Read count",
                                gene=selected_gene,
                            )
                        )

                        records_by_gene[
                            selected_gene
                        ].append(
                            HPAClusterExpressionRecord(
                                gene=selected_gene,
                                ensembl_gene_id=(
                                    self._required_value(
                                        row,
                                        "Gene",
                                    )
                                ),
                                tissue=row_tissue,
                                cell_type=(
                                    self._required_value(
                                        row,
                                        "Cell type",
                                    )
                                ),
                                cluster=(
                                    self._required_value(
                                        row,
                                        "Cluster",
                                    )
                                ),
                                read_count=read_count,
                                ncpm=ncpm,
                            )
                        )

        self._notify(
            progress_callback,
            f"Finished reading {row_count:,} HPA rows.",
        )

        gene_summaries: list[
            HPAGeneExpressionSummary
        ] = []

        expression_records: list[
            HPAClusterExpressionRecord
        ] = []

        for gene in selected_genes:
            records = sorted(
                records_by_gene[gene],
                key=lambda record: (
                    -record.ncpm,
                    record.cell_type.casefold(),
                    record.cluster.casefold(),
                ),
            )

            expression_records.extend(
                records
            )

            cell_type_summaries = (
                self._summarize_cell_types(
                    records
                )
            )

            detected_cell_types = [
                summary.cell_type
                for summary in cell_type_summaries
            ]

            breast_expression_found = bool(
                cell_type_summaries
            )

            source_status: HPAStatus

            top_breast_cell_type: str | None
            highest_ncpm: float
            top_cell_type_max_read_count: float
            top_cell_type_cluster_count: int

            second_breast_cell_type: str | None
            second_highest_ncpm: float

            ncpm_gap_top_vs_second: float

            top_to_second_ncpm_ratio: (
                float | None
            )

            if breast_expression_found:
                first_summary = (
                    cell_type_summaries[0]
                )

                source_status = (
                    "breast_expression_found"
                )

                top_breast_cell_type = (
                    first_summary.cell_type
                )

                highest_ncpm = (
                    first_summary.max_ncpm
                )

                top_cell_type_max_read_count = (
                    first_summary.max_read_count
                )

                top_cell_type_cluster_count = (
                    first_summary
                    .expressing_cluster_count
                )

                if len(cell_type_summaries) >= 2:
                    second_summary = (
                        cell_type_summaries[1]
                    )

                    second_breast_cell_type = (
                        second_summary.cell_type
                    )

                    second_highest_ncpm = (
                        second_summary.max_ncpm
                    )

                    ncpm_gap_top_vs_second = (
                        highest_ncpm
                        - second_highest_ncpm
                    )

                    top_to_second_ncpm_ratio = (
                        highest_ncpm
                        / second_highest_ncpm
                    )

                else:
                    second_breast_cell_type = None
                    second_highest_ncpm = 0.0

                    ncpm_gap_top_vs_second = (
                        highest_ncpm
                    )

                    top_to_second_ncpm_ratio = None

            elif gene_present_anywhere[gene]:
                source_status = (
                    "gene_found_without_breast_expression"
                )

                top_breast_cell_type = None
                highest_ncpm = 0.0
                top_cell_type_max_read_count = 0.0
                top_cell_type_cluster_count = 0

                second_breast_cell_type = None
                second_highest_ncpm = 0.0

                ncpm_gap_top_vs_second = 0.0
                top_to_second_ncpm_ratio = None

            else:
                source_status = (
                    "gene_not_found_in_hpa"
                )

                top_breast_cell_type = None
                highest_ncpm = 0.0
                top_cell_type_max_read_count = 0.0
                top_cell_type_cluster_count = 0

                second_breast_cell_type = None
                second_highest_ncpm = 0.0

                ncpm_gap_top_vs_second = 0.0
                top_to_second_ncpm_ratio = None

            gene_summaries.append(
                HPAGeneExpressionSummary(
                    gene=gene,
                    gene_present_in_hpa=(
                        gene_present_anywhere[gene]
                    ),
                    breast_expression_found=(
                        breast_expression_found
                    ),
                    source_status=source_status,
                    positive_record_count=len(
                        records
                    ),
                    detected_cell_type_count=len(
                        detected_cell_types
                    ),
                    detected_cell_types=(
                        detected_cell_types
                    ),
                    top_breast_cell_type=(
                        top_breast_cell_type
                    ),
                    highest_ncpm=highest_ncpm,
                    top_cell_type_max_read_count=(
                        top_cell_type_max_read_count
                    ),
                    top_cell_type_cluster_count=(
                        top_cell_type_cluster_count
                    ),
                    second_breast_cell_type=(
                        second_breast_cell_type
                    ),
                    second_highest_ncpm=(
                        second_highest_ncpm
                    ),
                    ncpm_gap_top_vs_second=(
                        ncpm_gap_top_vs_second
                    ),
                    top_to_second_ncpm_ratio=(
                        top_to_second_ncpm_ratio
                    ),
                    cell_type_summaries=(
                        cell_type_summaries
                    ),
                )
            )

        run_summary = HPARunSummary(
            requested_gene_count=len(
                selected_genes
            ),
            breast_expression_found_count=sum(
                1
                for result in gene_summaries
                if result.source_status
                == "breast_expression_found"
            ),
            gene_found_without_breast_expression_count=sum(
                1
                for result in gene_summaries
                if result.source_status
                == (
                    "gene_found_without_"
                    "breast_expression"
                )
            ),
            gene_not_found_in_hpa_count=sum(
                1
                for result in gene_summaries
                if result.source_status
                == "gene_not_found_in_hpa"
            ),
            expression_record_count=len(
                expression_records
            ),
        )

        self._notify(
            progress_callback,
            "Calculating the HPA file checksum...",
        )

        source_sha256 = self._sha256(
            reference_path
        )

        self._notify(
            progress_callback,
            "HPA retrieval summary completed.",
        )

        return HPARetrievalResult(
            cluster_id=cluster_id,
            run_mode=run_mode,
            query_tissue=cleaned_tissue,
            source_release=self.source_release,
            source_url=self.source_url,
            source_sha256=source_sha256,
            source_file_size_bytes=(
                reference_path.stat().st_size
            ),
            retrieved_at_utc=datetime.now(
                timezone.utc
            ),
            selected_genes=selected_genes,
            summary=run_summary,
            gene_summaries=gene_summaries,
            expression_records=expression_records,
            warnings=[
                (
                    "HPA breast evidence in this workflow "
                    "comes from a healthy-breast reference."
                ),
                (
                    "Every positive nCPM record is preserved "
                    "as raw expression evidence."
                ),
                (
                    "The pipeline does not automatically "
                    "label an nCPM value as biologically "
                    "weak, moderate or strong."
                ),
                (
                    "Top-versus-second measurements are "
                    "descriptive and are not final marker "
                    "confidence scores."
                ),
                (
                    "HPA expression evidence remains "
                    "separate from CellMarker and "
                    "PanglaoDB marker votes."
                ),
            ],
        )

    @staticmethod
    def _clean_genes(
        genes: list[str],
    ) -> list[str]:
        """Cleans the requested gene list."""

        cleaned_genes = [
            gene.strip()
            for gene in genes
            if gene.strip()
        ]

        if not cleaned_genes:
            raise ValueError(
                "At least one gene is required."
            )

        normalized_genes = [
            gene.upper()
            for gene in cleaned_genes
        ]

        if len(normalized_genes) != len(
            set(normalized_genes)
        ):
            raise ValueError(
                "Requested genes must be unique."
            )

        return cleaned_genes

    @classmethod
    def _validate_header(
        cls,
        fieldnames: Sequence[str] | None,
    ) -> None:
        """Validates the HPA table columns."""

        if fieldnames is None:
            raise ValueError(
                "The HPA TSV is missing its header."
            )

        cleaned_fieldnames = {
            fieldname.strip()
            for fieldname in fieldnames
            if fieldname.strip()
        }

        missing_columns = (
            cls.REQUIRED_COLUMNS
            - cleaned_fieldnames
        )

        if missing_columns:
            raise ValueError(
                "The HPA TSV is missing required columns: "
                f"{sorted(missing_columns)}. "
                f"Available columns: "
                f"{list(fieldnames)}"
            )

    @staticmethod
    def _find_tsv_member(
        archive: zipfile.ZipFile,
    ) -> str:
        """Finds the expression TSV in the ZIP."""

        tsv_members = [
            member
            for member in archive.namelist()
            if member.casefold().endswith(
                ".tsv"
            )
        ]

        preferred_members = [
            member
            for member in tsv_members
            if Path(member).name.casefold()
            == "rna_single_cell_cluster.tsv"
        ]

        if len(preferred_members) == 1:
            return preferred_members[0]

        if len(tsv_members) == 1:
            return tsv_members[0]

        raise ValueError(
            "Could not uniquely identify the HPA "
            "cluster-expression TSV inside the ZIP. "
            f"TSV files found: {tsv_members}"
        )

    @staticmethod
    def _summarize_cell_types(
        records: list[
            HPAClusterExpressionRecord
        ],
    ) -> list[
        HPACellTypeExpressionSummary
    ]:
        """Summarizes records by cell type."""

        records_by_cell_type: dict[
            str,
            list[HPAClusterExpressionRecord],
        ] = {}

        for record in records:
            records_by_cell_type.setdefault(
                record.cell_type,
                [],
            ).append(record)

        summaries: list[
            HPACellTypeExpressionSummary
        ] = []

        for cell_type, cell_records in (
            records_by_cell_type.items()
        ):
            clusters = sorted(
                {
                    record.cluster
                    for record in cell_records
                },
                key=str.casefold,
            )

            summaries.append(
                HPACellTypeExpressionSummary(
                    cell_type=cell_type,
                    max_ncpm=max(
                        record.ncpm
                        for record in cell_records
                    ),
                    max_read_count=max(
                        record.read_count
                        for record in cell_records
                    ),
                    positive_record_count=len(
                        cell_records
                    ),
                    expressing_cluster_count=len(
                        clusters
                    ),
                    clusters=clusters,
                )
            )

        return sorted(
            summaries,
            key=lambda summary: (
                -summary.max_ncpm,
                summary.cell_type.casefold(),
            ),
        )

    @staticmethod
    def _required_value(
        row: Mapping[str, object],
        column: str,
    ) -> str:
        """Returns one required table value."""

        value = row.get(column)

        if not isinstance(value, str):
            raise ValueError(
                "Required HPA value is missing "
                f"for column: {column}"
            )

        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError(
                "Required HPA value is empty "
                f"for column: {column}"
            )

        return cleaned_value

    @staticmethod
    def _parse_nonnegative_float(
        value: str,
        field_name: str,
        gene: str,
    ) -> float:
        """Parses one nonnegative number."""

        try:
            parsed_value = float(value)

        except ValueError as exc:
            raise ValueError(
                f"Invalid {field_name} value "
                f"for gene {gene}: {value}"
            ) from exc

        if parsed_value < 0:
            raise ValueError(
                f"{field_name} cannot be "
                f"negative for gene {gene}."
            )

        return parsed_value

    @staticmethod
    def _sha256(
        file_path: Path,
    ) -> str:
        """Calculates the HPA ZIP checksum."""

        digest = hashlib.sha256()

        with file_path.open(
            "rb"
        ) as handle:
            while chunk := handle.read(
                1024 * 1024
            ):
                digest.update(chunk)

        return digest.hexdigest()

    @staticmethod
    def _notify(
        callback: Callable[[str], None] | None,
        message: str,
    ) -> None:
        """Sends an optional progress message."""

        if callback is not None:
            callback(message)