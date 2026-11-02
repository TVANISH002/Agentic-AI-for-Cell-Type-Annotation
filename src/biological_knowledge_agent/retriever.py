from typing import cast

import pandas as pd


class BiologicalKnowledgeRetriever:
    MARKER_REQUIRED_COLUMNS = {
        "spatial_cluster",
        "rank_within_cluster",
        "names",
        "scores",
        "logfoldchanges",
        "pvals",
        "pvals_adj",
        "pct_nz_group",
        "pct_nz_reference",
        "deg_validation_warnings",
    }

    INPUT_COLUMNS = [
        "spatial_cluster",
        "rank_within_cluster",
        "names",
        "gene_key",
        "scores",
        "logfoldchanges",
        "pvals",
        "pvals_adj",
        "pct_nz_group",
        "pct_nz_reference",
        "deg_validation_warnings",
    ]

    def __init__(
        self,
        cellmarker: pd.DataFrame,
        panglaodb: pd.DataFrame,
    ) -> None:
        # Initializes reference data and alias lookup.
        self.cellmarker = cellmarker.copy()
        self.panglaodb = panglaodb.copy()

        self._validate_reference_tables()

        self.panglaodb_alias_lookup = (
            self._build_panglaodb_alias_lookup()
        )

    @staticmethod
    def _normalize_gene_key(
        series: pd.Series,
    ) -> pd.Series:
        # Normalizes gene symbols for matching.
        normalized = (
            series.astype("string")
            .str.strip()
            .str.upper()
        )

        return normalized.mask(
            normalized.isna() | normalized.eq(""),
            pd.NA,
        )

    def _validate_reference_tables(
        self,
    ) -> None:
        # Validates required reference columns.
        missing_cm = {
            "gene_key",
            "marker_key",
        } - set(self.cellmarker.columns)

        missing_pg = {
            "gene_key",
            "nicknames",
            "human_only_alias_safe",
        } - set(self.panglaodb.columns)

        if missing_cm:
            raise ValueError(
                "CellMarker lookup table is missing "
                f"columns: {sorted(missing_cm)}"
            )

        if missing_pg:
            raise ValueError(
                "PanglaoDB lookup table is missing "
                f"columns: {sorted(missing_pg)}"
            )

    def _build_panglaodb_alias_lookup(
        self,
    ) -> pd.DataFrame:
        # Builds the PanglaoDB alias lookup.
        safe_mask = (
            self.panglaodb[
                "human_only_alias_safe"
            ]
            .fillna(False)
            .astype(bool)
        )

        alias_mask = (
            self.panglaodb[
                "nicknames"
            ].notna()
        )

        alias_lookup = self.panglaodb.loc[
            safe_mask & alias_mask
        ].copy()

        alias_lookup["alias_key"] = (
            alias_lookup["nicknames"]
            .astype("string")
            .str.split(
                "|",
                regex=False,
            )
        )

        alias_lookup = alias_lookup.explode(
            "alias_key",
            ignore_index=True,
        )

        alias_series = cast(
            pd.Series,
            alias_lookup["alias_key"],
        )

        alias_lookup["alias_key"] = (
            self._normalize_gene_key(
                alias_series
            )
        )

        alias_lookup = alias_lookup.loc[
            alias_lookup[
                "alias_key"
            ].notna()
        ].copy()

        return (
            alias_lookup
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def prepare_marker_input(
        self,
        marker_data: pd.DataFrame,
    ) -> pd.DataFrame:
        # Validates and prepares marker input.
        missing = (
            self.MARKER_REQUIRED_COLUMNS
            - set(marker_data.columns)
        )

        if missing:
            raise ValueError(
                "Marker input is missing required "
                f"columns: {sorted(missing)}"
            )

        marker_status = marker_data.copy()

        name_series = cast(
            pd.Series,
            marker_status["names"],
        )

        marker_status["gene_key"] = (
            self._normalize_gene_key(
                name_series
            )
        )

        empty_mask = (
            marker_status[
                "gene_key"
            ].isna()
        )

        if bool(empty_mask.any()):
            bad = marker_status.loc[
                empty_mask,
                [
                    "spatial_cluster",
                    "rank_within_cluster",
                    "names",
                ],
            ]

            raise ValueError(
                "Marker input contains empty gene "
                f"names: {bad.to_dict(orient='records')}"
            )

        return marker_status

    def match_markers(
        self,
        marker_data: pd.DataFrame,
    ) -> pd.DataFrame:
        # Matches genes using symbols and aliases.
        status = self.prepare_marker_input(
            marker_data
        )

        cm_genes = (
            self.cellmarker["gene_key"]
            .dropna()
            .astype(str)
            .tolist()
        )

        pg_genes = (
            self.panglaodb["gene_key"]
            .dropna()
            .astype(str)
            .tolist()
        )

        cm_aliases = (
            self.cellmarker["marker_key"]
            .dropna()
            .astype(str)
            .tolist()
        )

        pg_aliases = (
            self.panglaodb_alias_lookup[
                "alias_key"
            ]
            .dropna()
            .astype(str)
            .tolist()
        )

        status["matched_cellmarker"] = (
            status["gene_key"].isin(
                cm_genes
            )
        )

        status["matched_panglaodb"] = (
            status["gene_key"].isin(
                pg_genes
            )
        )

        status["matched_cellmarker_alias"] = (
            ~status["matched_cellmarker"]
        ) & status["gene_key"].isin(
            cm_aliases
        )

        status["matched_panglaodb_alias"] = (
            ~status["matched_panglaodb"]
        ) & status["gene_key"].isin(
            pg_aliases
        )

        status["matched_cellmarker_final"] = (
            status["matched_cellmarker"]
            | status["matched_cellmarker_alias"]
        )

        status["matched_panglaodb_final"] = (
            status["matched_panglaodb"]
            | status["matched_panglaodb_alias"]
        )

        status["matched_both"] = (
            status["matched_cellmarker_final"]
            & status["matched_panglaodb_final"]
        )

        status["matched_either"] = (
            status["matched_cellmarker"]
            | status["matched_panglaodb"]
        )

        status["matched_alias"] = (
            status["matched_cellmarker_alias"]
            | status["matched_panglaodb_alias"]
        )

        status["matched_final"] = (
            status["matched_cellmarker_final"]
            | status["matched_panglaodb_final"]
        )

        return status

    def retrieve_cellmarker_evidence(
        self,
        status: pd.DataFrame,
    ) -> pd.DataFrame:
        # Retrieves CellMarker evidence.
        official = status.loc[
            status["matched_cellmarker"],
            self.INPUT_COLUMNS,
        ].rename(
            columns={
                "gene_key": "input_gene_key"
            }
        )

        official = official.merge(
            self.cellmarker,
            left_on="input_gene_key",
            right_on="gene_key",
            how="inner",
        )

        official[
            "retrieval_match_method"
        ] = "official_symbol"

        alias_mask = (
            ~status["matched_cellmarker"]
        ) & status[
            "matched_cellmarker_alias"
        ]

        alias = status.loc[
            alias_mask,
            self.INPUT_COLUMNS,
        ].rename(
            columns={
                "gene_key": "input_gene_key"
            }
        )

        alias = alias.merge(
            self.cellmarker,
            left_on="input_gene_key",
            right_on="marker_key",
            how="inner",
        )

        alias[
            "retrieval_match_method"
        ] = "alias"

        return pd.concat(
            [official, alias],
            ignore_index=True,
        )

    def retrieve_panglaodb_evidence(
        self,
        status: pd.DataFrame,
    ) -> pd.DataFrame:
        # Retrieves PanglaoDB evidence.
        official = status.loc[
            status["matched_panglaodb"],
            self.INPUT_COLUMNS,
        ].rename(
            columns={
                "gene_key": "input_gene_key"
            }
        )

        official = official.merge(
            self.panglaodb,
            left_on="input_gene_key",
            right_on="gene_key",
            how="inner",
        )

        official[
            "retrieval_match_method"
        ] = "official_symbol"

        alias_mask = (
            ~status["matched_panglaodb"]
        ) & status[
            "matched_panglaodb_alias"
        ]

        alias = status.loc[
            alias_mask,
            self.INPUT_COLUMNS,
        ].rename(
            columns={
                "gene_key": "input_gene_key"
            }
        )

        alias = alias.merge(
            self.panglaodb_alias_lookup,
            left_on="input_gene_key",
            right_on="alias_key",
            how="inner",
        )

        alias[
            "retrieval_match_method"
        ] = "alias"

        return pd.concat(
            [official, alias],
            ignore_index=True,
        )

    def retrieve_all(
        self,
        marker_data: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        # Retrieves all marker evidence.
        status = self.match_markers(
            marker_data
        )

        cellmarker = (
            self.retrieve_cellmarker_evidence(
                status
            )
        )

        panglaodb = (
            self.retrieve_panglaodb_evidence(
                status
            )
        )

        return (
            status,
            cellmarker,
            panglaodb,
        )


