from pathlib import Path
from typing import cast

import pandas as pd


class ReferenceDataLoader:
    CELLMARKER_REQUIRED_COLUMNS = {
        "Symbol",
        "marker",
        "cell_name",
        "cell_type",
        "tissue_type",
        "cancer_type",
        "cellontology_id",
        "marker_source",
        "PMID",
        "Genename",
        "Title",
    }

    PANGLAODB_REQUIRED_COLUMNS = {
        "species",
        "official gene symbol",
        "cell type",
        "nicknames",
        "organ",
        "product description",
        "canonical marker",
        "germ layer",
        "gene type",
    }

    def __init__(
        self,
        project_root: Path | str,
    ) -> None:
        self.project_root = Path(
            project_root
        ).resolve()

        self.cellmarker_path = (
            self.project_root
            / "data"
            / "reference"
            / "Cell_marker_Human.xlsx"
        )

        self.panglaodb_path = (
            self.project_root
            / "data"
            / "reference"
            / "PanglaoDB_markers_27_Mar_2020.tsv"
        )

    @staticmethod
    def _validate_columns(
        dataframe: pd.DataFrame,
        required: set[str],
        source: str,
    ) -> None:
        missing = required - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"{source} is missing required columns: "
                f"{sorted(missing)}"
            )

    @staticmethod
    def _normalize_gene_key(
        series: pd.Series,
    ) -> pd.Series:
        normalized = (
            series.astype("string")
            .str.strip()
            .str.upper()
        )

        return normalized.mask(
            normalized.isna() | normalized.eq(""),
            pd.NA,
        )

    def load_cellmarker(
        self,
    ) -> pd.DataFrame:
        if not self.cellmarker_path.exists():
            raise FileNotFoundError(
                f"CellMarker file not found: "
                f"{self.cellmarker_path}"
            )

        data = pd.read_excel(
            self.cellmarker_path,
            sheet_name="human",
        )

        self._validate_columns(
            data,
            self.CELLMARKER_REQUIRED_COLUMNS,
            "CellMarker",
        )

        data = data.copy()

        symbol_series = cast(
            pd.Series,
            data["Symbol"],
        )

        marker_series = cast(
            pd.Series,
            data["marker"],
        )

        data["gene_key"] = (
            self._normalize_gene_key(
                symbol_series
            )
        )

        data["marker_key"] = (
            self._normalize_gene_key(
                marker_series
            )
        )

        return data.reset_index(
            drop=True
        )

    def load_panglaodb(
        self,
    ) -> pd.DataFrame:
        if not self.panglaodb_path.exists():
            raise FileNotFoundError(
                f"PanglaoDB file not found: "
                f"{self.panglaodb_path}"
            )

        data = pd.read_csv(
            self.panglaodb_path,
            sep="\t",
        )

        self._validate_columns(
            data,
            self.PANGLAODB_REQUIRED_COLUMNS,
            "PanglaoDB",
        )

        data = data.copy()

        data["species"] = (
            data["species"]
            .astype("string")
            .str.strip()
        )

        human_mask = (
            data["species"]
            .str.contains(
                r"\bHs\b",
                na=False,
                regex=True,
            )
        )

        data = data.loc[
            human_mask
        ].copy()

        symbol_series = cast(
            pd.Series,
            data["official gene symbol"],
        )

        data["gene_key"] = (
            self._normalize_gene_key(
                symbol_series
            )
        )

        data["human_only_alias_safe"] = (
            data["species"]
            .eq("Hs")
            .fillna(False)
            .astype(bool)
        )

        return data.reset_index(
            drop=True
        )

    def load_all(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return (
            self.load_cellmarker(),
            self.load_panglaodb(),
        )




