from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hpa_evidence_agent.config import (
    DEFAULT_CONFIG,
    HPAEvidenceConfig,
)
from hpa_evidence_agent.schemas import (
    HPAClusterEvidence,
    HPAClusterGeneEvidence,
    HPAGeneSummary,
    HPAPanelEvidence,
)


@dataclass(frozen=True, slots=True)
class ClusterMarkerRecord:
    """Store one validated cluster marker row."""

    cluster_id: str | int
    gene: str
    score: float | None
    log_fold_change: float | None
    adjusted_p_value: float | None
    source_row_number: int


class HPAClusterProjector:
    """Project independent HPA evidence onto marker clusters."""

    BASE_WARNINGS = [
        (
            "HPA evidence is projected independently and is not "
            "combined with CellMarker or PanglaoDB in this report."
        ),
        (
            "HPA breast evidence represents a healthy-breast "
            "reference and is not direct breast-cancer marker evidence."
        ),
        (
            "Projected HPA expression is descriptive and does not "
            "assign a final cluster cell type."
        ),
    ]

    def __init__(
        self,
        config: HPAEvidenceConfig = DEFAULT_CONFIG,
    ) -> None:
        """Store and validate the HPA configuration."""

        config.validate()
        self.config = config

    @staticmethod
    def _normalize_gene(
        value: str | None,
        *,
        row_number: int,
    ) -> str:
        """Clean one marker gene symbol."""

        gene = (
            ""
            if value is None
            else value.strip().upper()
        )

        if not gene:
            raise ValueError(
                "Marker gene is empty at CSV row "
                f"{row_number}."
            )

        return gene

    @staticmethod
    def _normalize_cluster_id(
        value: str | None,
        *,
        row_number: int,
    ) -> str | int:
        """Clean one cluster ID."""

        cleaned = (
            ""
            if value is None
            else value.strip()
        )

        if not cleaned:
            raise ValueError(
                "Cluster ID is empty at CSV row "
                f"{row_number}."
            )

        try:
            numeric_value = float(cleaned)
        except ValueError:
            return cleaned

        if (
            math.isfinite(numeric_value)
            and numeric_value.is_integer()
        ):
            return int(numeric_value)

        return cleaned

    @staticmethod
    def _optional_float(
        value: str | None,
        *,
        field_name: str,
        row_number: int,
        nonnegative: bool = False,
    ) -> float | None:
        """Read an optional numeric value safely."""

        cleaned = (
            ""
            if value is None
            else value.strip()
        )

        if not cleaned:
            return None

        try:
            parsed = float(cleaned)
        except ValueError as error:
            raise ValueError(
                f"{field_name} is not numeric at CSV row "
                f"{row_number}: {cleaned!r}."
            ) from error

        if not math.isfinite(parsed):
            raise ValueError(
                f"{field_name} must be finite at CSV row "
                f"{row_number}."
            )

        if nonnegative and parsed < 0.0:
            raise ValueError(
                f"{field_name} cannot be negative at CSV row "
                f"{row_number}."
            )

        return parsed

    @staticmethod
    def _cluster_sort_key(
        cluster_id: str | int,
    ) -> tuple[int, int, str]:
        """Sort numeric clusters before text cluster labels."""

        if isinstance(cluster_id, int):
            return (
                0,
                cluster_id,
                "",
            )

        return (
            1,
            0,
            str(cluster_id).casefold(),
        )

    @staticmethod
    def _marker_sort_key(
        marker: ClusterMarkerRecord,
    ) -> tuple[float, int, str]:
        """Sort markers by score and original CSV order."""

        score = (
            marker.score
            if marker.score is not None
            else float("-inf")
        )

        return (
            -score,
            marker.source_row_number,
            marker.gene,
        )

    def _required_columns(
        self,
    ) -> tuple[str, ...]:
        """Return marker CSV columns needed for projection."""

        return (
            self.config.marker_gene_column,
            self.config.marker_cluster_column,
            self.config.marker_score_column,
            self.config.marker_logfc_column,
            self.config.marker_padj_column,
        )

    def _load_cluster_markers(
        self,
        *,
        selected_gene_set: set[str],
        path: Path,
    ) -> tuple[
        list[str | int],
        dict[
            str | int,
            list[ClusterMarkerRecord],
        ],
    ]:
        """Read panel-matched markers for every cluster."""

        markers_by_cluster: dict[
            str | int,
            list[ClusterMarkerRecord],
        ] = defaultdict(list)

        cluster_order: list[
            str | int
        ] = []

        seen_clusters: set[
            str | int
        ] = set()

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                raise ValueError(
                    "Cluster marker CSV does not "
                    "contain a header row."
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
                if column not in normalized_headers
            ]

            if missing_columns:
                available_columns = ", ".join(
                    sorted(normalized_headers)
                )

                raise ValueError(
                    "Cluster marker CSV is missing "
                    "required columns: "
                    f"{', '.join(missing_columns)}. "
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
                cluster_id = (
                    self._normalize_cluster_id(
                        row.get(
                            source_columns[
                                self.config
                                .marker_cluster_column
                            ]
                        ),
                        row_number=row_number,
                    )
                )

                if cluster_id not in seen_clusters:
                    seen_clusters.add(
                        cluster_id
                    )
                    cluster_order.append(
                        cluster_id
                    )

                gene = self._normalize_gene(
                    row.get(
                        source_columns[
                            self.config
                            .marker_gene_column
                        ]
                    ),
                    row_number=row_number,
                )

                # Only cleaned panel genes are projected.
                if gene not in selected_gene_set:
                    continue

                markers_by_cluster[
                    cluster_id
                ].append(
                    ClusterMarkerRecord(
                        cluster_id=cluster_id,
                        gene=gene,
                        score=(
                            self._optional_float(
                                row.get(
                                    source_columns[
                                        self.config
                                        .marker_score_column
                                    ]
                                ),
                                field_name=(
                                    self.config
                                    .marker_score_column
                                ),
                                row_number=(
                                    row_number
                                ),
                            )
                        ),
                        log_fold_change=(
                            self._optional_float(
                                row.get(
                                    source_columns[
                                        self.config
                                        .marker_logfc_column
                                    ]
                                ),
                                field_name=(
                                    self.config
                                    .marker_logfc_column
                                ),
                                row_number=(
                                    row_number
                                ),
                            )
                        ),
                        adjusted_p_value=(
                            self._optional_float(
                                row.get(
                                    source_columns[
                                        self.config
                                        .marker_padj_column
                                    ]
                                ),
                                field_name=(
                                    self.config
                                    .marker_padj_column
                                ),
                                row_number=(
                                    row_number
                                ),
                                nonnegative=True,
                            )
                        ),
                        source_row_number=(
                            row_number
                        ),
                    )
                )

        if not cluster_order:
            raise ValueError(
                "Cluster marker CSV does not "
                "contain any rows."
            )

        return (
            sorted(
                cluster_order,
                key=self._cluster_sort_key,
            ),
            markers_by_cluster,
        )

    def _select_cluster_markers(
        self,
        records: Sequence[
            ClusterMarkerRecord
        ],
    ) -> list[ClusterMarkerRecord]:
        """Keep the top unique panel genes for one cluster."""

        selected: list[
            ClusterMarkerRecord
        ] = []

        seen_genes: set[str] = set()

        for record in sorted(
            records,
            key=self._marker_sort_key,
        ):
            if record.gene in seen_genes:
                continue

            seen_genes.add(
                record.gene
            )

            selected.append(
                record
            )

            if (
                len(selected)
                >= self.config
                .top_genes_per_cluster
            ):
                break

        return selected

    @classmethod
    def _cluster_warnings(
        cls,
        *,
        marker_gene_count: int,
        requested_marker_count: int,
        missing_genes: Sequence[str],
    ) -> list[str]:
        """Create transparent warnings for one cluster."""

        warnings = list(
            cls.BASE_WARNINGS
        )

        if (
            marker_gene_count
            < requested_marker_count
        ):
            warnings.append(
                "Only "
                f"{marker_gene_count} unique panel-matched "
                "marker genes were available; "
                f"{requested_marker_count} were requested."
            )

        if missing_genes:
            warnings.append(
                "Exact HPA symbols not found for: "
                + " | ".join(
                    missing_genes
                )
                + "."
            )

        return warnings

    def project(
        self,
        panel_evidence: HPAPanelEvidence,
        *,
        marker_path: str | Path | None = None,
    ) -> list[HPAClusterEvidence]:
        """Project panel HPA summaries onto all clusters."""

        validated_panel = (
            HPAPanelEvidence.model_validate(
                panel_evidence.model_dump()
            )
        )

        resolved_marker_path = (
            Path(marker_path)
            if marker_path is not None
            else self.config.cluster_marker_path
        )

        if not resolved_marker_path.exists():
            raise FileNotFoundError(
                "Cluster marker CSV was not found: "
                f"{resolved_marker_path.resolve()}"
            )

        if not resolved_marker_path.is_file():
            raise ValueError(
                "Cluster marker path is not a file: "
                f"{resolved_marker_path.resolve()}"
            )

        selected_gene_set = set(
            validated_panel.selected_genes
        )

        gene_summary_map: dict[
            str,
            HPAGeneSummary,
        ] = {
            summary.gene: summary
            for summary
            in validated_panel.gene_summaries
        }

        (
            cluster_ids,
            markers_by_cluster,
        ) = self._load_cluster_markers(
            selected_gene_set=(
                selected_gene_set
            ),
            path=resolved_marker_path,
        )

        cluster_evidence: list[
            HPAClusterEvidence
        ] = []

        for cluster_id in cluster_ids:
            selected_markers = (
                self._select_cluster_markers(
                    markers_by_cluster.get(
                        cluster_id,
                        [],
                    )
                )
            )

            projected_genes: list[
                HPAClusterGeneEvidence
            ] = []

            for rank, marker in enumerate(
                selected_markers,
                start=1,
            ):
                hpa_summary = (
                    gene_summary_map.get(
                        marker.gene
                    )
                )

                if hpa_summary is None:
                    raise ValueError(
                        "No HPA gene summary exists "
                        "for projected marker "
                        f"{marker.gene} in cluster "
                        f"{cluster_id}."
                    )

                projected_genes.append(
                    HPAClusterGeneEvidence(
                        cluster_id=cluster_id,
                        gene=marker.gene,
                        rank_within_cluster=rank,
                        score=marker.score,
                        log_fold_change=(
                            marker.log_fold_change
                        ),
                        adjusted_p_value=(
                            marker.adjusted_p_value
                        ),
                        hpa=hpa_summary,
                    )
                )

            breast_expression_found_count = sum(
                item.hpa.source_status
                == "breast_expression_found"
                for item in projected_genes
            )

            found_without_breast_count = sum(
                item.hpa.source_status
                == (
                    "gene_found_without_"
                    "breast_expression"
                )
                for item in projected_genes
            )

            gene_not_found_count = sum(
                item.hpa.source_status
                == "gene_not_found_in_hpa"
                for item in projected_genes
            )

            missing_genes = [
                item.gene
                for item in projected_genes
                if (
                    item.hpa.source_status
                    == "gene_not_found_in_hpa"
                )
            ]

            cluster_evidence.append(
                HPAClusterEvidence(
                    cluster_id=cluster_id,
                    query_tissue=(
                        validated_panel
                        .query_tissue
                    ),
                    source=(
                        validated_panel.source
                    ),
                    source_release=(
                        validated_panel
                        .source_release
                    ),
                    marker_gene_count=len(
                        projected_genes
                    ),
                    breast_expression_found_count=(
                        breast_expression_found_count
                    ),
                    gene_found_without_breast_expression_count=(
                        found_without_breast_count
                    ),
                    gene_not_found_in_hpa_count=(
                        gene_not_found_count
                    ),
                    genes=projected_genes,
                    warnings=(
                        self._cluster_warnings(
                            marker_gene_count=len(
                                projected_genes
                            ),
                            requested_marker_count=(
                                self.config
                                .top_genes_per_cluster
                            ),
                            missing_genes=(
                                missing_genes
                            ),
                        )
                    ),
                )
            )

        return cluster_evidence


def project_hpa_to_clusters(
    panel_evidence: HPAPanelEvidence,
    config: HPAEvidenceConfig = DEFAULT_CONFIG,
    *,
    marker_path: str | Path | None = None,
) -> list[HPAClusterEvidence]:
    """Project independent HPA evidence to all clusters."""

    return HPAClusterProjector(
        config
    ).project(
        panel_evidence,
        marker_path=marker_path,
    )