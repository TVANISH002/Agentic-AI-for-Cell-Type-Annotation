from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from marker_interpretation_agent_v2.paths import (
    PANEL_GENE_LIST_CSV,
    get_marker_file,
    normalize_path,
)
from marker_interpretation_agent_v2.schemas import InputGene


PanelGeneOrder = Literal["alphabetical", "source"]


class GenePanelLoader:
    """
    Load and validate the complete cluster marker-gene table.

    The loader discovers every unique panel gene and preserves the
    cluster-specific differential-expression statistics used by Agent 2.
    """

    COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
        "gene": (
            "gene",
            "genes",
            "name",
            "names",
            "gene_name",
            "gene_symbol",
            "symbol",
        ),
        "cluster_id": (
            "cluster_id",
            "cluster",
            "spatial_cluster",
            "leiden",
            "group",
        ),
        "rank_within_cluster": (
            "rank_within_cluster",
            "gene_rank",
            "marker_rank",
            "rank",
        ),
        "score": (
            "score",
            "scores",
            "test_score",
            "wilcoxon_score",
        ),
        "log_fold_change": (
            "log_fold_change",
            "logfoldchange",
            "logfoldchanges",
            "log2fc",
            "log2_fold_change",
            "avg_log2fc",
        ),
        "p_value": (
            "p_value",
            "pvalue",
            "pvalues",
            "pvals",
            "p_val",
        ),
        "adjusted_p_value": (
            "adjusted_p_value",
            "adjusted_pvalue",
            "adjusted_pvalues",
            "pvals_adj",
            "p_val_adj",
            "padj",
            "fdr",
        ),
        "pct_nz_group": (
            "pct_nz_group",
            "pct_group",
            "pct_in",
            "pct_in_cluster",
            "pct_1",
        ),
        "pct_nz_reference": (
            "pct_nz_reference",
            "pct_reference",
            "pct_out",
            "pct_out_cluster",
            "pct_2",
        ),
    }

    REQUIRED_COLUMNS = (
        "gene",
        "cluster_id",
    )

    NUMERIC_COLUMNS = (
        "score",
        "log_fold_change",
        "p_value",
        "adjusted_p_value",
        "pct_nz_group",
        "pct_nz_reference",
    )

    PROPORTION_COLUMNS = (
        "pct_nz_group",
        "pct_nz_reference",
    )

    def __init__(
        self,
        marker_file: str | Path | None = None,
    ) -> None:
        """Resolve the marker CSV path without loading it immediately."""

        self.marker_file = get_marker_file(marker_file)
        self._dataframe: pd.DataFrame | None = None

    @staticmethod
    def _normalize_column_name(value: Any) -> str:
        """Normalize a source column name for alias matching."""

        return str(value).strip().lower()

    @staticmethod
    def _normalize_gene(value: Any) -> str:
        """Normalize one gene symbol."""

        if value is None or bool(pd.isna(value)):
            raise ValueError("Gene symbol cannot be missing.")

        gene = str(value).strip().upper()

        if not gene:
            raise ValueError("Gene symbol cannot be empty.")

        return gene

    @staticmethod
    def _normalize_cluster_id(value: Any) -> str:
        """Convert a cluster identifier into a stable string."""

        if value is None or bool(pd.isna(value)):
            raise ValueError("Cluster identifier cannot be missing.")

        text = str(value).strip()

        if not text:
            raise ValueError("Cluster identifier cannot be empty.")

        try:
            numeric_value = float(text)
        except (TypeError, ValueError):
            return text

        if numeric_value.is_integer():
            return str(int(numeric_value))

        return text

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        """Convert a nullable value to float or None."""

        if value is None or bool(pd.isna(value)):
            return None

        return float(value)

    @staticmethod
    def _required_positive_int(
        value: Any,
        *,
        field_name: str,
    ) -> int:
        """Convert a value to a positive whole number."""

        if value is None or bool(pd.isna(value)):
            raise ValueError(f"{field_name} cannot be missing.")

        numeric_value = float(value)

        if numeric_value <= 0 or not numeric_value.is_integer():
            raise ValueError(
                f"{field_name} must be a positive whole number. "
                f"Received: {value!r}"
            )

        return int(numeric_value)

    def _resolve_source_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> dict[str, str]:
        """Map source CSV columns to canonical Agent 2 names."""

        available_columns = {
            self._normalize_column_name(column): str(column)
            for column in dataframe.columns
        }

        resolved: dict[str, str] = {}

        for canonical_name, aliases in self.COLUMN_ALIASES.items():
            for alias in aliases:
                source_name = available_columns.get(
                    self._normalize_column_name(alias)
                )

                if source_name is not None:
                    resolved[canonical_name] = source_name
                    break

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in resolved
        ]

        if missing_columns:
            missing_text = ", ".join(missing_columns)
            available_text = ", ".join(
                str(column)
                for column in dataframe.columns
            )

            raise ValueError(
                "Marker CSV is missing required column(s): "
                f"{missing_text}.\n"
                f"Available columns: {available_text}"
            )

        return resolved

    def _convert_numeric_column(
        self,
        source: pd.Series,
        *,
        canonical_name: str,
        source_name: str,
    ) -> pd.Series:
        """Convert a source column to numeric values with validation."""

        converted = pd.to_numeric(
            source,
            errors="coerce",
        )

        invalid_mask = source.notna() & converted.isna()

        if bool(invalid_mask.any()):
            invalid_values = (
                source.loc[invalid_mask]
                .astype(str)
                .drop_duplicates()
                .head(5)
                .tolist()
            )

            raise ValueError(
                f"Column {source_name!r}, mapped to "
                f"{canonical_name!r}, contains non-numeric values: "
                f"{invalid_values}"
            )

        return converted.astype(float)

    def _build_standardized_dataframe(
        self,
        raw_dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create the validated canonical marker table."""

        if raw_dataframe.empty:
            raise ValueError(
                f"Marker CSV contains no rows: {self.marker_file}"
            )

        resolved_columns = self._resolve_source_columns(
            raw_dataframe
        )

        standardized = pd.DataFrame(
            index=raw_dataframe.index
        )

        standardized["_source_row"] = range(
            1,
            len(raw_dataframe) + 1,
        )

        standardized["gene"] = raw_dataframe[
            resolved_columns["gene"]
        ].map(self._normalize_gene)

        standardized["cluster_id"] = raw_dataframe[
            resolved_columns["cluster_id"]
        ].map(self._normalize_cluster_id)

        for canonical_name in self.NUMERIC_COLUMNS:
            source_name = resolved_columns.get(canonical_name)

            if source_name is None:
                standardized[canonical_name] = float("nan")
                continue

            standardized[canonical_name] = (
                self._convert_numeric_column(
                    raw_dataframe[source_name],
                    canonical_name=canonical_name,
                    source_name=source_name,
                )
            )

        rank_source_name = resolved_columns.get(
            "rank_within_cluster"
        )

        if rank_source_name is None:
            standardized["rank_within_cluster"] = (
                standardized.groupby(
                    "cluster_id",
                    sort=False,
                )
                .cumcount()
                .add(1)
                .astype(int)
            )
        else:
            rank_values = pd.to_numeric(
                raw_dataframe[rank_source_name],
                errors="coerce",
            )

            invalid_rank_mask = (
                raw_dataframe[rank_source_name].isna()
                | rank_values.isna()
                | rank_values.le(0)
                | rank_values.mod(1).ne(0)
            )

            if bool(invalid_rank_mask.any()):
                invalid_values = (
                    raw_dataframe.loc[
                        invalid_rank_mask,
                        rank_source_name,
                    ]
                    .astype(str)
                    .drop_duplicates()
                    .head(5)
                    .tolist()
                )

                raise ValueError(
                    f"Rank column {rank_source_name!r} must "
                    "contain positive whole numbers. "
                    f"Invalid values: {invalid_values}"
                )

            standardized["rank_within_cluster"] = (
                rank_values.astype(int)
            )

        self._validate_standardized_dataframe(
            standardized
        )

        return standardized.sort_values(
            by="_source_row",
            kind="stable",
        ).reset_index(drop=True)

    def _validate_standardized_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate duplicate, probability, and proportion rules."""

        duplicate_mask = dataframe.duplicated(
            subset=[
                "cluster_id",
                "gene",
            ],
            keep=False,
        )

        if bool(duplicate_mask.any()):
            duplicates = (
                dataframe.loc[
                    duplicate_mask,
                    [
                        "cluster_id",
                        "gene",
                    ],
                ]
                .drop_duplicates()
                .head(10)
                .to_dict(orient="records")
            )

            raise ValueError(
                "Marker CSV contains duplicate gene records "
                "within the same cluster: "
                f"{duplicates}"
            )

        duplicate_rank_mask = dataframe.duplicated(
            subset=[
                "cluster_id",
                "rank_within_cluster",
            ],
            keep=False,
        )

        if bool(duplicate_rank_mask.any()):
            duplicates = (
                dataframe.loc[
                    duplicate_rank_mask,
                    [
                        "cluster_id",
                        "rank_within_cluster",
                    ],
                ]
                .drop_duplicates()
                .head(10)
                .to_dict(orient="records")
            )

            raise ValueError(
                "Marker CSV contains duplicate ranks within "
                f"the same cluster: {duplicates}"
            )

        for column in (
            "p_value",
            "adjusted_p_value",
        ):
            values = dataframe[column].dropna()

            if bool(values.lt(0.0).any()):
                raise ValueError(
                    f"{column} cannot contain negative values."
                )

        for column in self.PROPORTION_COLUMNS:
            values = dataframe[column].dropna()

            invalid_mask = (
                values.lt(0.0)
                | values.gt(1.0)
            )

            if bool(invalid_mask.any()):
                examples = (
                    values.loc[invalid_mask]
                    .drop_duplicates()
                    .head(5)
                    .tolist()
                )

                raise ValueError(
                    f"{column} must contain proportions "
                    "between 0.0 and 1.0. "
                    f"Invalid values: {examples}"
                )

    def load_dataframe(
        self,
        *,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """Load the marker file and return a defensive copy."""

        if self._dataframe is None or force_reload:
            raw_dataframe = pd.read_csv(
                self.marker_file
            )

            self._dataframe = (
                self._build_standardized_dataframe(
                    raw_dataframe
                )
            )

        return self._dataframe.copy()

    def get_cluster_ids(self) -> list[str]:
        """Return cluster identifiers in source-file order."""

        dataframe = self.load_dataframe()

        values = cast(
            list[Any],
            dataframe[
                "cluster_id"
            ].drop_duplicates().tolist(),
        )

        return [
            self._normalize_cluster_id(value)
            for value in values
        ]

    def get_panel_genes(
        self,
        *,
        order: PanelGeneOrder = "alphabetical",
    ) -> list[str]:
        """Return every unique panel gene exactly once."""

        dataframe = self.load_dataframe()

        values = cast(
            list[Any],
            dataframe[
                "gene"
            ].drop_duplicates().tolist(),
        )

        genes_in_source_order = [
            self._normalize_gene(value)
            for value in values
        ]

        if order == "source":
            return genes_in_source_order

        if order == "alphabetical":
            return sorted(genes_in_source_order)

        raise ValueError(
            "order must be either "
            "'alphabetical' or 'source'."
        )

    def get_cluster_dataframe(
        self,
        cluster_id: str | int,
        *,
        top_k: int | None = None,
    ) -> pd.DataFrame:
        """Return one cluster's marker rows in DEG-rank order."""

        normalized_cluster_id = (
            self._normalize_cluster_id(
                cluster_id
            )
        )

        dataframe = self.load_dataframe()

        cluster_dataframe = dataframe.loc[
            dataframe["cluster_id"]
            == normalized_cluster_id
        ].copy()

        if cluster_dataframe.empty:
            available_clusters = ", ".join(
                self.get_cluster_ids()
            )

            raise KeyError(
                f"Cluster {normalized_cluster_id!r} "
                "was not found. "
                f"Available clusters: {available_clusters}"
            )

        cluster_dataframe = (
            cluster_dataframe.sort_values(
                by=[
                    "rank_within_cluster",
                    "_source_row",
                ],
                kind="stable",
            ).reset_index(drop=True)
        )

        if top_k is not None:
            if top_k <= 0:
                raise ValueError(
                    "top_k must be greater than 0."
                )

            cluster_dataframe = (
                cluster_dataframe.head(top_k).copy()
            )

        return cluster_dataframe

    def get_cluster_markers(
        self,
        cluster_id: str | int,
        *,
        top_k: int | None = None,
    ) -> list[InputGene]:
        """Return one cluster's markers as validated schemas."""

        cluster_dataframe = (
            self.get_cluster_dataframe(
                cluster_id,
                top_k=top_k,
            )
        )

        records = cast(
            list[dict[str, Any]],
            cluster_dataframe.to_dict(
                orient="records"
            ),
        )

        markers: list[InputGene] = []

        for record in records:
            markers.append(
                InputGene(
                    gene=self._normalize_gene(
                        record.get("gene")
                    ),
                    rank_within_cluster=(
                        self._required_positive_int(
                            record.get(
                                "rank_within_cluster"
                            ),
                            field_name=(
                                "rank_within_cluster"
                            ),
                        )
                    ),
                    score=self._optional_float(
                        record.get("score")
                    ),
                    log_fold_change=(
                        self._optional_float(
                            record.get(
                                "log_fold_change"
                            )
                        )
                    ),
                    p_value=self._optional_float(
                        record.get("p_value")
                    ),
                    adjusted_p_value=(
                        self._optional_float(
                            record.get(
                                "adjusted_p_value"
                            )
                        )
                    ),
                    pct_nz_group=(
                        self._optional_float(
                            record.get(
                                "pct_nz_group"
                            )
                        )
                    ),
                    pct_nz_reference=(
                        self._optional_float(
                            record.get(
                                "pct_nz_reference"
                            )
                        )
                    ),
                )
            )

        return markers

    def build_panel_gene_dataframe(
        self,
    ) -> pd.DataFrame:
        """Build a one-row-per-gene panel inventory."""

        dataframe = self.load_dataframe()

        grouped = cast(
            pd.DataFrame,
            dataframe.groupby(
                "gene",
                sort=False,
            )
            .agg(
                cluster_count=(
                    "cluster_id",
                    "nunique",
                ),
                occurrence_count=(
                    "gene",
                    "size",
                ),
                best_rank_across_clusters=(
                    "rank_within_cluster",
                    "min",
                ),
                best_score_across_clusters=(
                    "score",
                    "max",
                ),
                maximum_log_fold_change=(
                    "log_fold_change",
                    "max",
                ),
                first_source_row=(
                    "_source_row",
                    "min",
                ),
            )
            .reset_index(),
        )

        grouped = grouped.sort_values(
            by="gene",
            kind="stable",
        ).reset_index(drop=True)

        grouped.insert(
            0,
            "panel_gene_index",
            range(1, len(grouped) + 1),
        )

        return grouped[
            [
                "panel_gene_index",
                "gene",
                "cluster_count",
                "occurrence_count",
                "best_rank_across_clusters",
                "best_score_across_clusters",
                "maximum_log_fold_change",
                "first_source_row",
            ]
        ].copy()

    def write_panel_gene_list(
        self,
        output_path: str | Path | None = None,
    ) -> Path:
        """Write the one-row-per-gene panel inventory."""

        resolved_output_path = (
            normalize_path(output_path)
            if output_path is not None
            else PANEL_GENE_LIST_CSV
        )

        resolved_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.build_panel_gene_dataframe().to_csv(
            resolved_output_path,
            index=False,
        )

        return resolved_output_path

    def summary(self) -> dict[str, Any]:
        """Return a diagnostic summary of the marker panel."""

        dataframe = self.load_dataframe()

        rows_per_cluster = dataframe.groupby(
            "cluster_id",
            sort=False,
        ).size()

        minimum_rows = int(
            cast(
                Any,
                rows_per_cluster.min(),
            )
        )

        maximum_rows = int(
            cast(
                Any,
                rows_per_cluster.max(),
            )
        )

        unique_gene_count = int(
            cast(
                Any,
                dataframe["gene"].nunique(),
            )
        )

        cluster_count = int(
            cast(
                Any,
                dataframe["cluster_id"].nunique(),
            )
        )

        duplicate_count = int(
            cast(
                Any,
                dataframe.duplicated(
                    subset=[
                        "cluster_id",
                        "gene",
                    ]
                ).sum(),
            )
        )

        return {
            "marker_file": str(self.marker_file),
            "total_cluster_gene_rows": len(dataframe),
            "unique_panel_gene_count": unique_gene_count,
            "cluster_count": cluster_count,
            "cluster_ids": self.get_cluster_ids(),
            "minimum_rows_per_cluster": minimum_rows,
            "maximum_rows_per_cluster": maximum_rows,
            "duplicate_cluster_gene_pairs": duplicate_count,
        }