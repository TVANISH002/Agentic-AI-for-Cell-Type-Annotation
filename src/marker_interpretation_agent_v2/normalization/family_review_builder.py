"""Compare raw candidate leaders with conservative subtype summaries.

The comparison uses subtype groups, not broad biological lineages, when
determining whether the original and normalized interpretations agree.
Broad lineage is retained only as additional review context.
"""

from __future__ import annotations

from math import isfinite, isnan
from pathlib import Path

import pandas as pd

from .family_mapping import map_cell_type


_RAW_REQUIRED_COLUMNS = {
    "cluster_id",
    "candidate_rank",
    "cell_type",
    "unique_supporting_gene_count",
}

_SUBTYPE_REQUIRED_COLUMNS = {
    "cluster_id",
    "family_rank",
    "subtype_group",
    "broad_lineage_family",
    "unique_supporting_gene_count",
    "raw_candidate_count",
    "raw_cell_types",
}


def _is_missing(value: object) -> bool:
    """Return True for scalar missing values read from CSV files."""

    if value is None or value is pd.NA:
        return True

    if isinstance(value, float):
        return isnan(value)

    return str(value).strip().casefold() in {
        "",
        "nan",
        "none",
        "<na>",
        "nat",
    }


def _as_text(value: object) -> str:
    """Convert one scalar CSV value into clean text."""

    if _is_missing(value):
        return ""

    return str(value).strip()


def _as_int(value: object) -> int:
    """Convert one scalar CSV value into a validated integer."""

    if _is_missing(value):
        raise ValueError(
            "Expected an integer value, found a missing value."
        )

    if isinstance(value, bool):
        return int(value)

    text = str(value).strip()

    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Expected a numeric integer value, found: {value!r}."
        ) from exc

    if not isfinite(number):
        raise ValueError(
            f"Expected a finite integer value, found: {value!r}."
        )

    if not number.is_integer():
        raise ValueError(
            f"Expected an integer value, found: {value!r}."
        )

    return int(number)


def _normalize_cluster_id(value: object) -> str:
    """Create a stable cluster identifier from a CSV value."""

    if _is_missing(value):
        return ""

    if isinstance(value, bool):
        return str(value)

    text = str(value).strip()

    try:
        number = float(text)
    except ValueError:
        return text

    if isfinite(number) and number.is_integer():
        return str(int(number))

    return text


def _single_cluster_id(
    frame: pd.DataFrame,
    source_name: str,
) -> str:
    """Validate and return the single cluster identifier in a table."""

    cluster_ids = {
        cluster_id
        for value in frame["cluster_id"].tolist()
        if (cluster_id := _normalize_cluster_id(value))
    }

    if len(cluster_ids) != 1:
        raise ValueError(
            f"{source_name} must contain exactly one cluster_id; "
            f"found {sorted(cluster_ids)}."
        )

    return next(iter(cluster_ids))


class CandidateFamilyReviewBuilder:
    """Compare raw leaders with normalized subtype-level summaries."""

    def build_dataframe(
        self,
        raw_candidates: pd.DataFrame,
        family_summary: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build one raw-versus-subtype review record."""

        raw_missing = sorted(
            _RAW_REQUIRED_COLUMNS.difference(
                raw_candidates.columns
            )
        )

        subtype_missing = sorted(
            _SUBTYPE_REQUIRED_COLUMNS.difference(
                family_summary.columns
            )
        )

        if raw_missing:
            raise ValueError(
                "Raw candidate table is missing required columns: "
                + ", ".join(raw_missing)
            )

        if subtype_missing:
            raise ValueError(
                "Subtype summary is missing required columns: "
                + ", ".join(subtype_missing)
            )

        if raw_candidates.empty:
            raise ValueError(
                "Raw candidate table is empty."
            )

        if family_summary.empty:
            raise ValueError(
                "Subtype summary table is empty."
            )

        raw_cluster_id = _single_cluster_id(
            raw_candidates,
            "Raw candidate table",
        )

        subtype_cluster_id = _single_cluster_id(
            family_summary,
            "Subtype summary",
        )

        if raw_cluster_id != subtype_cluster_id:
            raise ValueError(
                "Cluster mismatch between raw and subtype tables: "
                f"raw={raw_cluster_id}, "
                f"subtype={subtype_cluster_id}."
            )

        raw_sorted = raw_candidates.sort_values(
            by="candidate_rank",
            kind="stable",
        )

        subtype_sorted = family_summary.sort_values(
            by="family_rank",
            kind="stable",
        )

        raw_top = raw_sorted.iloc[0]
        subtype_top = subtype_sorted.iloc[0]

        subtype_second = (
            subtype_sorted.iloc[1]
            if len(subtype_sorted) > 1
            else None
        )

        raw_top_cell_type = _as_text(
            raw_top["cell_type"]
        )

        raw_top_support = _as_int(
            raw_top["unique_supporting_gene_count"]
        )

        raw_mapping = map_cell_type(
            raw_top_cell_type
        )

        raw_top_canonical = (
            raw_mapping.canonical_cell_type
        )

        raw_top_subtype = (
            raw_mapping.subtype_group
        )

        raw_top_broad_lineage = (
            raw_mapping.broad_lineage_family
        )

        subtype_top_name = _as_text(
            subtype_top["subtype_group"]
        )

        subtype_top_broad_lineage = _as_text(
            subtype_top["broad_lineage_family"]
        )

        subtype_top_support = _as_int(
            subtype_top[
                "unique_supporting_gene_count"
            ]
        )

        subtype_top_raw_candidate_count = _as_int(
            subtype_top["raw_candidate_count"]
        )

        subtype_top_raw_cell_types = _as_text(
            subtype_top["raw_cell_types"]
        )

        subtype_second_name = ""
        subtype_second_broad_lineage = ""
        subtype_second_support = 0

        if subtype_second is not None:
            subtype_second_name = _as_text(
                subtype_second["subtype_group"]
            )

            subtype_second_broad_lineage = _as_text(
                subtype_second[
                    "broad_lineage_family"
                ]
            )

            subtype_second_support = _as_int(
                subtype_second[
                    "unique_supporting_gene_count"
                ]
            )

        top_subtype_tie = bool(
            subtype_second is not None
            and subtype_top_support
            == subtype_second_support
        )

        raw_subtype_leader_match = (
            raw_top_subtype == subtype_top_name
        )

        subtype_aggregation = (
            subtype_top_raw_candidate_count > 1
        )

        subtype_support_gap = (
            max(
                subtype_top_support
                - subtype_second_support,
                0,
            )
            if subtype_second is not None
            else subtype_top_support
        )

        subtype_support_gap_ratio = (
            subtype_support_gap
            / subtype_top_support
            if subtype_top_support > 0
            else 0.0
        )

        broad_lineage_match = (
            raw_top_broad_lineage
            == subtype_top_broad_lineage
        )

        review_flags: list[str] = []

        if not raw_subtype_leader_match:
            review_flags.append(
                "raw_subtype_leader_conflict"
            )

        if top_subtype_tie:
            review_flags.append(
                "top_subtype_tie"
            )

        if subtype_aggregation:
            review_flags.append(
                "multiple_raw_labels_in_subtype"
            )

        if broad_lineage_match:
            review_flags.append(
                "broad_lineage_consistent"
            )
        else:
            review_flags.append(
                "broad_lineage_conflict"
            )

        if (
            raw_subtype_leader_match
            and not top_subtype_tie
        ):
            review_flags.append(
                "stable_leading_subtype"
            )

        if (
            not raw_subtype_leader_match
            or top_subtype_tie
        ):
            review_status = (
                "manual_review_required"
            )
        elif subtype_aggregation:
            review_status = (
                "review_recommended"
            )
        else:
            review_status = (
                "stable_candidate_interpretation"
            )

        review_note = self._build_review_note(
            raw_top_cell_type=raw_top_cell_type,
            raw_top_subtype=raw_top_subtype,
            raw_top_broad_lineage=(
                raw_top_broad_lineage
            ),
            subtype_top_name=subtype_top_name,
            subtype_top_broad_lineage=(
                subtype_top_broad_lineage
            ),
            subtype_second_name=(
                subtype_second_name
            ),
            subtype_top_support=(
                subtype_top_support
            ),
            raw_subtype_leader_match=(
                raw_subtype_leader_match
            ),
            top_subtype_tie=(
                top_subtype_tie
            ),
            subtype_aggregation=(
                subtype_aggregation
            ),
            subtype_top_raw_candidate_count=(
                subtype_top_raw_candidate_count
            ),
        )

        return pd.DataFrame(
            [
                {
                    "cluster_id": raw_cluster_id,
                    "review_status": review_status,
                    "review_flags": " | ".join(
                        review_flags
                    ),
                    "raw_top_cell_type": (
                        raw_top_cell_type
                    ),
                    "raw_top_unique_supporting_gene_count": (
                        raw_top_support
                    ),
                    "raw_top_canonical_cell_type": (
                        raw_top_canonical
                    ),
                    "raw_top_subtype_group": (
                        raw_top_subtype
                    ),
                    "raw_top_broad_lineage_family": (
                        raw_top_broad_lineage
                    ),

                    # Backward-compatible field used by the runner.
                    "raw_top_mapped_family": (
                        raw_top_subtype
                    ),

                    # Backward-compatible field used by the runner.
                    "family_top_cell_type": (
                        subtype_top_name
                    ),

                    "family_top_unique_supporting_gene_count": (
                        subtype_top_support
                    ),
                    "family_top_broad_lineage_family": (
                        subtype_top_broad_lineage
                    ),
                    "family_second_cell_type": (
                        subtype_second_name
                    ),
                    "family_second_unique_supporting_gene_count": (
                        subtype_second_support
                    ),
                    "family_second_broad_lineage_family": (
                        subtype_second_broad_lineage
                    ),
                    "family_support_gap": (
                        subtype_support_gap
                    ),
                    "family_support_gap_ratio": round(
                        subtype_support_gap_ratio,
                        4,
                    ),
                    "top_family_tie": (
                        top_subtype_tie
                    ),
                    "raw_family_leader_match": (
                        raw_subtype_leader_match
                    ),
                    "broad_lineage_match": (
                        broad_lineage_match
                    ),
                    "top_family_raw_candidate_count": (
                        subtype_top_raw_candidate_count
                    ),
                    "top_family_raw_cell_types": (
                        subtype_top_raw_cell_types
                    ),
                    "comparison_level": (
                        "subtype_group"
                    ),
                    "review_note": review_note,
                }
            ]
        )

    def build_csv(
        self,
        raw_candidates_csv: Path | str,
        family_summary_csv: Path | str,
        output_csv: Path | str,
    ) -> pd.DataFrame:
        """Read input CSVs and write one cluster review record."""

        raw_path = Path(
            raw_candidates_csv
        )

        subtype_path = Path(
            family_summary_csv
        )

        output_path = Path(
            output_csv
        )

        if not raw_path.exists():
            raise FileNotFoundError(
                "Raw candidate CSV was not found: "
                f"{raw_path}"
            )

        if not subtype_path.exists():
            raise FileNotFoundError(
                "Subtype summary CSV was not found: "
                f"{subtype_path}"
            )

        review = self.build_dataframe(
            raw_candidates=pd.read_csv(
                raw_path
            ),
            family_summary=pd.read_csv(
                subtype_path
            ),
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        review.to_csv(
            output_path,
            index=False,
        )

        return review

    @staticmethod
    def _build_review_note(
        raw_top_cell_type: str,
        raw_top_subtype: str,
        raw_top_broad_lineage: str,
        subtype_top_name: str,
        subtype_top_broad_lineage: str,
        subtype_second_name: str,
        subtype_top_support: int,
        raw_subtype_leader_match: bool,
        top_subtype_tie: bool,
        subtype_aggregation: bool,
        subtype_top_raw_candidate_count: int,
    ) -> str:
        """Create a readable subtype-level comparison explanation."""

        if (
            not raw_subtype_leader_match
            and top_subtype_tie
        ):
            return (
                f"Raw leader '{raw_top_cell_type}' maps to subtype "
                f"'{raw_top_subtype}', while the normalized subtype "
                f"summary is tied between '{subtype_top_name}' and "
                f"'{subtype_second_name}'."
            )

        if not raw_subtype_leader_match:
            return (
                f"Raw leader '{raw_top_cell_type}' maps to subtype "
                f"'{raw_top_subtype}' and broad lineage "
                f"'{raw_top_broad_lineage}', but the leading "
                f"normalized subtype is '{subtype_top_name}' under "
                f"the '{subtype_top_broad_lineage}' lineage."
            )

        if top_subtype_tie:
            return (
                f"Raw leader agrees with subtype '{subtype_top_name}', "
                f"but it is tied with '{subtype_second_name}' at "
                f"{subtype_top_support} supporting genes."
            )

        if subtype_aggregation:
            return (
                f"Raw leader agrees with subtype '{subtype_top_name}'. "
                f"The subtype combines "
                f"{subtype_top_raw_candidate_count} equivalent or "
                "closely related raw labels."
            )

        return (
            f"Raw leader and normalized subtype agree on "
            f"'{subtype_top_name}'. Broad lineage: "
            f"'{subtype_top_broad_lineage}'."
        )