"""Build a concise cluster-level interpretation summary for Agent 2 V2.

The summary preserves the original leading candidate, normalized subtype,
competing candidate, supporting genes, and review status. It does not assign
confidence percentages or replace ambiguous results with a forced annotation.
"""

from __future__ import annotations

from math import isfinite, isnan
from pathlib import Path

import pandas as pd


_RAW_REQUIRED_COLUMNS = {
    "cluster_id",
    "candidate_rank",
    "cell_type",
    "unique_supporting_gene_count",
    "supporting_genes",
}

_SUBTYPE_REQUIRED_COLUMNS = {
    "cluster_id",
    "family_rank",
    "subtype_group",
    "broad_lineage_family",
    "unique_supporting_gene_count",
    "supporting_genes",
}

_REVIEW_REQUIRED_COLUMNS = {
    "cluster_id",
    "review_status",
    "review_flags",
    "raw_top_cell_type",
    "raw_top_unique_supporting_gene_count",
    "raw_top_subtype_group",
    "raw_top_broad_lineage_family",
    "family_top_cell_type",
    "family_top_unique_supporting_gene_count",
    "family_second_cell_type",
    "family_second_unique_supporting_gene_count",
    "top_family_tie",
    "raw_family_leader_match",
    "review_note",
}

_ALLOWED_REVIEW_STATUSES = {
    "stable_candidate_interpretation",
    "review_recommended",
    "manual_review_required",
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
    """Convert a scalar value into clean text."""

    if _is_missing(value):
        return ""

    return str(value).strip()


def _as_int(value: object) -> int:
    """Convert a scalar numeric value into a validated integer."""

    if _is_missing(value):
        return 0

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


def _as_bool(value: object) -> bool:
    """Convert a scalar value into a Boolean."""

    if isinstance(value, bool):
        return value

    if _is_missing(value):
        return False

    return str(value).strip().casefold() in {
        "true",
        "1",
        "yes",
        "y",
    }


def _normalize_cluster_id(value: object) -> str:
    """Create a stable cluster identifier."""

    if _is_missing(value):
        return ""

    text = str(value).strip()

    try:
        number = float(text)
    except ValueError:
        return text

    if isfinite(number) and number.is_integer():
        return str(int(number))

    return text


def _validate_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    source_name: str,
) -> None:
    """Validate required columns for one input table."""

    missing = sorted(
        required_columns.difference(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: "
            + ", ".join(missing)
        )


def _single_cluster_id(
    frame: pd.DataFrame,
    source_name: str,
) -> str:
    """Return the only cluster identifier present in a table."""

    cluster_ids = {
        cluster_id
        for value in frame["cluster_id"].tolist()
        if (
            cluster_id
            := _normalize_cluster_id(value)
        )
    }

    if len(cluster_ids) != 1:
        raise ValueError(
            f"{source_name} must contain exactly one cluster_id; "
            f"found {sorted(cluster_ids)}."
        )

    return next(iter(cluster_ids))


def _find_subtype_row(
    subtype_summary: pd.DataFrame,
    subtype_name: str,
) -> pd.Series | None:
    """Find the summary row corresponding to one normalized subtype."""

    if not subtype_name:
        return None

    matches = subtype_summary[
        subtype_summary["subtype_group"]
        .astype(str)
        .str.strip()
        .eq(subtype_name)
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


class ClusterInterpretationSummaryBuilder:
    """Create one concise interpretation record for a cluster."""

    def build_dataframe(
        self,
        raw_candidates: pd.DataFrame,
        subtype_summary: pd.DataFrame,
        review: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build one final review-oriented cluster summary."""

        _validate_columns(
            raw_candidates,
            _RAW_REQUIRED_COLUMNS,
            "Raw candidate table",
        )

        _validate_columns(
            subtype_summary,
            _SUBTYPE_REQUIRED_COLUMNS,
            "Subtype summary",
        )

        _validate_columns(
            review,
            _REVIEW_REQUIRED_COLUMNS,
            "Review table",
        )

        if raw_candidates.empty:
            raise ValueError(
                "Raw candidate table is empty."
            )

        if subtype_summary.empty:
            raise ValueError(
                "Subtype summary table is empty."
            )

        if len(review) != 1:
            raise ValueError(
                "Review table must contain exactly one record; "
                f"found {len(review)}."
            )

        raw_cluster_id = _single_cluster_id(
            raw_candidates,
            "Raw candidate table",
        )

        subtype_cluster_id = _single_cluster_id(
            subtype_summary,
            "Subtype summary",
        )

        review_cluster_id = _single_cluster_id(
            review,
            "Review table",
        )

        if not (
            raw_cluster_id
            == subtype_cluster_id
            == review_cluster_id
        ):
            raise ValueError(
                "Cluster identifiers do not match across inputs: "
                f"raw={raw_cluster_id}, "
                f"subtype={subtype_cluster_id}, "
                f"review={review_cluster_id}."
            )

        review_row = review.iloc[0]

        review_status = _as_text(
            review_row["review_status"]
        )

        if review_status not in _ALLOWED_REVIEW_STATUSES:
            raise ValueError(
                f"Unexpected review status: {review_status!r}."
            )

        raw_top_cell_type = _as_text(
            review_row["raw_top_cell_type"]
        )

        raw_top_support = _as_int(
            review_row[
                "raw_top_unique_supporting_gene_count"
            ]
        )

        normalized_top_subtype = _as_text(
            review_row["family_top_cell_type"]
        )

        normalized_top_support = _as_int(
            review_row[
                "family_top_unique_supporting_gene_count"
            ]
        )

        competing_subtype = _as_text(
            review_row["family_second_cell_type"]
        )

        competing_support = _as_int(
            review_row[
                "family_second_unique_supporting_gene_count"
            ]
        )

        top_subtype_tie = _as_bool(
            review_row["top_family_tie"]
        )

        raw_leader_match = _as_bool(
            review_row["raw_family_leader_match"]
        )

        normalized_top_row = _find_subtype_row(
            subtype_summary,
            normalized_top_subtype,
        )

        competing_row = _find_subtype_row(
            subtype_summary,
            competing_subtype,
        )

        normalized_top_genes = (
            ""
            if normalized_top_row is None
            else _as_text(
                normalized_top_row["supporting_genes"]
            )
        )

        normalized_top_lineage = (
            _as_text(
                review_row[
                    "raw_top_broad_lineage_family"
                ]
            )
            if normalized_top_row is None
            else _as_text(
                normalized_top_row[
                    "broad_lineage_family"
                ]
            )
        )

        competing_genes = (
            ""
            if competing_row is None
            else _as_text(
                competing_row["supporting_genes"]
            )
        )

        competing_lineage = (
            ""
            if competing_row is None
            else _as_text(
                competing_row[
                    "broad_lineage_family"
                ]
            )
        )

        raw_sorted = raw_candidates.sort_values(
            by="candidate_rank",
            kind="stable",
        )

        raw_top_row = raw_sorted.iloc[0]

        raw_top_genes = _as_text(
            raw_top_row["supporting_genes"]
        )

        interpretation_state = (
            self._interpretation_state(
                review_status
            )
        )

        professor_facing_interpretation = (
            self._build_interpretation(
                review_status=review_status,
                normalized_top_subtype=(
                    normalized_top_subtype
                ),
                competing_subtype=(
                    competing_subtype
                ),
                normalized_top_support=(
                    normalized_top_support
                ),
                competing_support=(
                    competing_support
                ),
                top_subtype_tie=(
                    top_subtype_tie
                ),
            )
        )

        return pd.DataFrame(
            [
                {
                    "cluster_id": raw_cluster_id,
                    "interpretation_state": (
                        interpretation_state
                    ),
                    "professor_facing_interpretation": (
                        professor_facing_interpretation
                    ),
                    "raw_top_cell_type": (
                        raw_top_cell_type
                    ),
                    "raw_top_support_count": (
                        raw_top_support
                    ),
                    "raw_top_supporting_genes": (
                        raw_top_genes
                    ),
                    "normalized_top_subtype": (
                        normalized_top_subtype
                    ),
                    "normalized_top_support_count": (
                        normalized_top_support
                    ),
                    "normalized_top_supporting_genes": (
                        normalized_top_genes
                    ),
                    "normalized_top_broad_lineage": (
                        normalized_top_lineage
                    ),
                    "competing_subtype": (
                        competing_subtype
                    ),
                    "competing_support_count": (
                        competing_support
                    ),
                    "competing_supporting_genes": (
                        competing_genes
                    ),
                    "competing_broad_lineage": (
                        competing_lineage
                    ),
                    "support_gap": max(
                        normalized_top_support
                        - competing_support,
                        0,
                    ),
                    "top_subtype_tie": (
                        top_subtype_tie
                    ),
                    "raw_normalized_leader_match": (
                        raw_leader_match
                    ),
                    "review_status": (
                        review_status
                    ),
                    "review_flags": _as_text(
                        review_row["review_flags"]
                    ),
                    "review_note": _as_text(
                        review_row["review_note"]
                    ),
                }
            ]
        )

    def build_csv(
        self,
        raw_candidates_csv: Path | str,
        subtype_summary_csv: Path | str,
        review_csv: Path | str,
        output_csv: Path | str,
    ) -> pd.DataFrame:
        """Read cluster inputs and write its concise summary."""

        raw_path = Path(
            raw_candidates_csv
        )

        subtype_path = Path(
            subtype_summary_csv
        )

        review_path = Path(
            review_csv
        )

        output_path = Path(
            output_csv
        )

        for source_path in (
            raw_path,
            subtype_path,
            review_path,
        ):
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Required input file was not found: "
                    f"{source_path}"
                )

        summary = self.build_dataframe(
            raw_candidates=pd.read_csv(
                raw_path
            ),
            subtype_summary=pd.read_csv(
                subtype_path
            ),
            review=pd.read_csv(
                review_path
            ),
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary.to_csv(
            output_path,
            index=False,
        )

        return summary

    @staticmethod
    def _interpretation_state(
        review_status: str,
    ) -> str:
        """Convert technical review status into a concise state."""

        status_mapping = {
            "stable_candidate_interpretation": (
                "stable"
            ),
            "review_recommended": (
                "consistent_review_recommended"
            ),
            "manual_review_required": (
                "ambiguous_manual_review"
            ),
        }

        return status_mapping[
            review_status
        ]

    @staticmethod
    def _build_interpretation(
        review_status: str,
        normalized_top_subtype: str,
        competing_subtype: str,
        normalized_top_support: int,
        competing_support: int,
        top_subtype_tie: bool,
    ) -> str:
        """Create a concise professor-facing interpretation."""

        if (
            review_status
            == "manual_review_required"
            and top_subtype_tie
        ):
            return (
                f"{normalized_top_subtype} versus "
                f"{competing_subtype} ambiguity "
                f"({normalized_top_support} supporting genes each)"
            )

        if (
            review_status
            == "manual_review_required"
        ):
            return (
                f"{normalized_top_subtype} is leading, "
                f"but {competing_subtype} also has "
                f"{competing_support} supporting genes"
            )

        if (
            review_status
            == "review_recommended"
        ):
            return (
                f"{normalized_top_subtype} remains the "
                f"leading interpretation "
                f"({normalized_top_support} supporting genes)"
            )

        return (
            f"{normalized_top_subtype} "
            f"({normalized_top_support} supporting genes)"
        )