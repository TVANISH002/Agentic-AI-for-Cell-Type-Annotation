from math import isfinite
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from . import evidence_builder
from .context import ContextMatcher
from .reference_loader import ReferenceDataLoader
from .retriever import BiologicalKnowledgeRetriever
from .schemas import DEGValidationSummary, KnowledgeEvidence


class BiologicalKnowledgeRetrievalAgent:
    """Deterministic CellMarker + PanglaoDB biological knowledge retriever."""

    SUPPORTED_HUMAN_SPECIES = {"human", "homo sapiens", "hs"}

    MARKER_REQUIRED_COLUMNS = {
        "spatial_cluster",
        "names",
        "scores",
        "logfoldchanges",
        "pvals",
        "pvals_adj",
        "pct_nz_group",
        "pct_nz_reference",
    }

    NUMERIC_COLUMNS = [
        "scores",
        "logfoldchanges",
        "pvals",
        "pvals_adj",
        "pct_nz_group",
        "pct_nz_reference",
    ]

    DEFAULT_MARKER_NAMES = (
        "brcr_top50_primary_marker_genes_by_cluster.csv",
        "cluster_marker_genes_brcr_Agentic_AI.csv",
    )

    def __init__(
        self,
        project_root: Path | str,
        marker_file: Path | str | None = None,
        top_k: int = 25,
        candidate_pool_size: int = 50,
        min_log_fold_change: float = 0.5,
        max_adjusted_p_value: float = 0.05,
    ) -> None:
        self.project_root = Path(project_root).resolve()

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if candidate_pool_size < top_k:
            raise ValueError(
                "candidate_pool_size must be greater than or equal to top_k."
            )

        if not isfinite(min_log_fold_change):
            raise ValueError("min_log_fold_change must be finite.")

        if not 0.0 <= max_adjusted_p_value <= 1.0:
            raise ValueError(
                "max_adjusted_p_value must be between 0 and 1."
            )

        self.top_k = top_k
        self.candidate_pool_size = candidate_pool_size
        self.min_log_fold_change = float(min_log_fold_change)
        self.max_adjusted_p_value = float(max_adjusted_p_value)
        self.marker_file = self._resolve_marker_file(marker_file)

        loader = ReferenceDataLoader(self.project_root)
        self.cellmarker, self.panglaodb = loader.load_all()

        self.retriever = BiologicalKnowledgeRetriever(
            self.cellmarker,
            self.panglaodb,
        )

        self.evidence_builder = evidence_builder.EvidenceBuilder()

    @staticmethod
    def _records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
        return cast(
            list[dict[str, Any]],
            dataframe.to_dict(orient="records"),
        )

    def _resolve_marker_file(
        self,
        marker_file: Path | str | None,
    ) -> Path:
        if marker_file is not None:
            path = Path(marker_file).resolve()

            if not path.exists():
                raise FileNotFoundError(
                    f"Marker file not found: {path}"
                )

            return path

        processed_dir = self.project_root / "data" / "processed"

        for filename in self.DEFAULT_MARKER_NAMES:
            path = processed_dir / filename

            if path.exists():
                return path.resolve()

        searched = [
            str(processed_dir / filename)
            for filename in self.DEFAULT_MARKER_NAMES
        ]

        raise FileNotFoundError(
            f"Marker file not found. Searched: {searched}"
        )

    @classmethod
    def _validate_species(cls, species: str) -> str:
        normalized = species.strip().casefold()

        if normalized not in cls.SUPPORTED_HUMAN_SPECIES:
            raise ValueError(
                "Agent 1 V1 currently supports only human evidence."
            )

        return "Human"

    @staticmethod
    def _validate_context(
        tissue: str,
        platform: str,
    ) -> tuple[str, str]:
        clean_tissue = " ".join(tissue.strip().split())
        clean_platform = " ".join(platform.strip().split())

        if not clean_tissue:
            raise ValueError("tissue cannot be empty.")

        if not clean_platform:
            raise ValueError("platform cannot be empty.")

        return clean_tissue, clean_platform

    def _load_marker_file(self) -> pd.DataFrame:
        markers = pd.read_csv(self.marker_file)

        if markers.empty:
            raise ValueError("The marker file is empty.")

        missing = self.MARKER_REQUIRED_COLUMNS - set(markers.columns)

        if missing:
            raise ValueError(
                "The marker file is missing required columns: "
                f"{sorted(missing)}"
            )

        markers = markers.copy()

        markers["names"] = (
            markers["names"]
            .astype("string")
            .str.strip()
        )

        invalid_names = (
            markers["names"].isna()
            | markers["names"].eq("")
        )

        if bool(invalid_names.any()):
            raise ValueError(
                "The marker file contains missing or empty gene names."
            )

        for column in self.NUMERIC_COLUMNS:
            markers[column] = pd.to_numeric(
                markers[column],
                errors="coerce",
            )

        invalid_numeric_columns = [
            column
            for column in self.NUMERIC_COLUMNS
            if bool(markers[column].isna().any())
        ]

        if invalid_numeric_columns:
            raise ValueError(
                "The marker file contains invalid numeric values "
                f"in columns: {invalid_numeric_columns}"
            )

        numeric_values = markers[self.NUMERIC_COLUMNS].to_numpy(
            dtype=float
        )

        if not bool(np.isfinite(numeric_values).all()):
            raise ValueError(
                "The marker file contains non-finite numeric values."
            )

        bounded_columns = (
            "pvals",
            "pvals_adj",
            "pct_nz_group",
            "pct_nz_reference",
        )

        for column in bounded_columns:
            invalid = ~markers[column].between(
                0.0,
                1.0,
                inclusive="both",
            )

            if bool(invalid.any()):
                raise ValueError(
                    f"{column} must contain values between 0 and 1."
                )

        return markers

    @staticmethod
    def _row_warnings(row: dict[str, Any]) -> list[str]:
        warnings: list[str] = []

        log_fold_change = float(row["logfoldchanges"])
        percent_group = float(row["pct_nz_group"])
        percent_reference = float(row["pct_nz_reference"])
        p_value = float(row["pvals"])
        adjusted_p_value = float(row["pvals_adj"])

        if (
            abs(log_fold_change) > 0.0
            and percent_group == 0.0
            and percent_reference == 0.0
        ):
            warnings.append(
                "Both expression percentages are zero despite "
                "non-zero log fold change; verify the upstream "
                "expression layer and rounding."
            )

        if (
            log_fold_change > 0.0
            and percent_group < percent_reference
        ):
            warnings.append(
                "Positive log fold change with lower detection "
                "percentage; this can be valid when expression "
                "intensity differs, but should be reviewed upstream."
            )

        if adjusted_p_value < p_value:
            warnings.append(
                "Adjusted p-value is smaller than the raw p-value; "
                "verify the multiple-testing output."
            )

        return warnings

    def _load_cluster_markers(
        self,
        cluster_id: int | str,
    ) -> tuple[
        pd.DataFrame,
        DEGValidationSummary,
        list[str],
    ]:
        markers = self._load_marker_file()

        cluster_mask = (
            markers["spatial_cluster"].astype(str)
            == str(cluster_id)
        )

        cluster = markers.loc[cluster_mask].copy()

        if cluster.empty:
            available = sorted(
                markers["spatial_cluster"]
                .astype(str)
                .unique()
                .tolist()
            )

            raise ValueError(
                f"Cluster {cluster_id} was not found. "
                f"Available clusters: {available}"
            )

        total_rows = len(cluster)

        threshold_mask = (
            cluster["logfoldchanges"]
            >= self.min_log_fold_change
        ) & (
            cluster["pvals_adj"]
            <= self.max_adjusted_p_value
        )

        passing = cluster.loc[threshold_mask].copy()
        rows_passing_thresholds = len(passing)

        if passing.empty:
            raise ValueError(
                f"Cluster {cluster_id} has no markers passing "
                f"logFC >= {self.min_log_fold_change} and "
                f"adjusted p-value <= "
                f"{self.max_adjusted_p_value}."
            )

        passing = passing.sort_values(
            ["scores", "logfoldchanges"],
            ascending=[False, False],
        )

        passing["gene_key_for_dedup"] = (
            passing["names"]
            .astype("string")
            .str.upper()
        )

        before_deduplication = len(passing)

        passing = passing.drop_duplicates(
            subset=["gene_key_for_dedup"],
            keep="first",
        )

        duplicate_genes_removed = (
            before_deduplication
            - len(passing)
        )

        candidate_pool = passing.head(
            self.candidate_pool_size
        ).copy()

        candidate_pool = candidate_pool.sort_values(
            ["logfoldchanges", "scores"],
            ascending=[False, False],
        )

        selected = (
            candidate_pool
            .head(self.top_k)
            .reset_index(drop=True)
        )

        selected["rank_within_cluster"] = list(
            range(1, len(selected) + 1)
        )

        warning_lists = [
            self._row_warnings(row)
            for row in self._records(selected)
        ]

        selected["deg_validation_warnings"] = pd.Series(
            warning_lists,
            index=selected.index,
            dtype=object,
        )

        selected = selected.drop(
            columns=["gene_key_for_dedup"]
        )

        warning_count = sum(
            len(items)
            for items in warning_lists
        )

        warnings = [
            (
                f"Selection strategy: top "
                f"{self.candidate_pool_size} score-ranked "
                f"candidates, then top {self.top_k} by "
                "log fold change."
            ),
            (
                "CellMarker Genename is preserved only as raw "
                "metadata and excluded from validated gene "
                "descriptions."
            ),
            (
                "PanglaoDB nicknames from mixed Mm/Hs rows are "
                "preserved in metadata but not promoted as "
                "human aliases."
            ),
        ]

        if len(selected) < self.top_k:
            warnings.append(
                f"Only {len(selected)} markers passed validation "
                f"although top_k={self.top_k}."
            )

        zero_percentage_count = sum(
            any(
                "Both expression percentages are zero"
                in warning
                for warning in items
            )
            for items in warning_lists
        )

        lower_percentage_count = sum(
            any(
                "lower detection percentage"
                in warning
                for warning in items
            )
            for items in warning_lists
        )

        if zero_percentage_count:
            warnings.append(
                f"{zero_percentage_count} selected gene(s) have "
                "zero detection percentages despite non-zero "
                "log fold change."
            )

        if lower_percentage_count:
            warnings.append(
                f"{lower_percentage_count} selected gene(s) have "
                "positive log fold change but lower detection "
                "percentage in the group."
            )

        summary = DEGValidationSummary.model_validate(
            {
                "total_cluster_rows": total_rows,
                "rows_passing_thresholds": rows_passing_thresholds,
                "rows_filtered_out": (
                    total_rows
                    - rows_passing_thresholds
                ),
                "rows_selected": len(selected),
                "duplicate_genes_removed": (
                    duplicate_genes_removed
                ),
                "min_log_fold_change": (
                    self.min_log_fold_change
                ),
                "max_adjusted_p_value": (
                    self.max_adjusted_p_value
                ),
                "warning_count": warning_count,
                "warnings": warnings,
            }
        )

        return selected, summary, warnings

    def run(
        self,
        cluster_id: int | str,
        species: str,
        tissue: str,
        platform: str,
    ) -> KnowledgeEvidence:
        normalized_species = self._validate_species(
            species
        )

        clean_tissue, clean_platform = (
            self._validate_context(
                tissue,
                platform,
            )
        )

        markers, deg_validation, warnings = (
            self._load_cluster_markers(
                cluster_id
            )
        )

        (
            marker_status,
            cellmarker_evidence,
            panglaodb_evidence,
        ) = self.retriever.retrieve_all(markers)

        result = self.evidence_builder.build_cluster(
            cluster_id=cluster_id,
            marker_status=marker_status,
            cellmarker_evidence=cellmarker_evidence,
            panglaodb_evidence=panglaodb_evidence,
            species=normalized_species,
            tissue=clean_tissue,
            platform=clean_platform,
            context_matcher=ContextMatcher.from_request(
                clean_tissue
            ),
            deg_validation=deg_validation,
            data_quality_warnings=warnings,
        )

        if (
            len(result.input_genes)
            != len(result.gene_evidence)
        ):
            raise ValueError(
                "Agent integrity check failed: input gene "
                "count and GeneEvidence count differ."
            )

        return result
