from __future__ import annotations

import math
from numbers import Real
from pathlib import Path

import pandas as pd


def _is_missing_scalar(value: object) -> bool:
    """Return whether a scalar cluster identifier is missing."""

    if value is None or value is pd.NA or value is pd.NaT:
        return True

    # CSV numeric columns can contain Python or NumPy real-number scalars.
    # Converting a Real to float handles both without calling overloaded
    # pandas.isna(), which static type checkers cannot safely narrow here.
    return isinstance(value, Real) and math.isnan(float(value))


def _cluster_id(value: object) -> str:
    """Normalize a cluster identifier without changing text identifiers."""

    if _is_missing_scalar(value):
        raise ValueError("Cluster identifier cannot be missing.")

    text = str(value).strip()

    if not text:
        raise ValueError("Cluster identifier cannot be empty.")

    try:
        number = float(text)
    except ValueError:
        return text

    return str(int(number)) if number.is_integer() else text


def prepare_lung_input(
    source_csv: Path,
    marker_output_csv: Path,
    panel_gene_list_csv: Path,
    *,
    min_log2fc: float = 0.5,
    max_adjusted_p_value: float = 0.05,
    top_k: int = 50,
) -> dict[str, int]:
    """Filter the supplied lung markers and write Agent 1/2 inputs."""

    source_csv = source_csv.resolve()

    if not source_csv.exists():
        raise FileNotFoundError(
            f"Lung marker CSV not found: {source_csv}"
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    raw = pd.read_csv(source_csv)

    if raw.empty:
        raise ValueError(
            "The lung marker CSV is empty."
        )

    required = {
        "cluster",
        "gene",
        "avg_log2FC",
        "p_val",
        "p_val_adj",
        "pct.1",
        "pct.2",
    }

    missing = sorted(
        required - set(raw.columns)
    )

    if missing:
        raise ValueError(
            f"Lung marker CSV is missing columns {missing}. "
            f"Available columns: {list(raw.columns)}"
        )

    markers = pd.DataFrame(
        index=raw.index
    )

    markers["_source_row"] = range(
        1,
        len(raw) + 1,
    )

    markers["cluster_id"] = (
        raw["cluster"].map(_cluster_id)
    )

    markers["gene"] = (
        raw["gene"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    invalid_genes = (
        markers["gene"].isna()
        | markers["gene"].eq("")
    )

    if bool(invalid_genes.any()):
        raise ValueError(
            "Lung marker CSV contains a missing gene symbol."
        )

    numeric_mapping = {
        "log_fold_change": "avg_log2FC",
        "p_value": "p_val",
        "adjusted_p_value": "p_val_adj",
        "pct_nz_group": "pct.1",
        "pct_nz_reference": "pct.2",
    }

    for target, source in numeric_mapping.items():
        markers[target] = pd.to_numeric(
            raw[source],
            errors="coerce",
        )

        if bool(markers[target].isna().any()):
            raise ValueError(
                f"Column {source!r} contains invalid values."
            )

    for column in (
        "p_value",
        "adjusted_p_value",
    ):
        invalid = (
            markers[column].lt(0)
            | markers[column].gt(1)
        )

        if bool(invalid.any()):
            raise ValueError(
                f"{column} must be between 0 and 1."
            )

    for column in (
        "pct_nz_group",
        "pct_nz_reference",
    ):
        invalid = (
            markers[column].lt(0)
            | markers[column].gt(1)
        )

        if bool(invalid.any()):
            raise ValueError(
                f"{column} must be between 0 and 1."
            )

    retained = markers.loc[
        markers["log_fold_change"].ge(
            min_log2fc
        )
        & markers["adjusted_p_value"].le(
            max_adjusted_p_value
        )
    ].copy()

    retained = (
        retained
        .sort_values(
            "_source_row",
            kind="stable",
        )
        .groupby(
            "cluster_id",
            sort=False,
            group_keys=False,
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    if retained.empty:
        raise ValueError(
            "No lung markers remain after filtering."
        )

    duplicate = retained.duplicated(
        [
            "cluster_id",
            "gene",
        ],
        keep=False,
    )

    if bool(duplicate.any()):
        examples = (
            retained.loc[
                duplicate,
                [
                    "cluster_id",
                    "gene",
                ],
            ]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )

        raise ValueError(
            f"Duplicate genes within a cluster: {examples}"
        )

    retained["rank_within_cluster"] = (
        retained.groupby(
            "cluster_id",
            sort=False,
        ).cumcount()
        + 1
    )

    # The supplied lung CSV has no test-score column.
    # Keep it missing instead of inventing a biological score.
    retained["score"] = float("nan")

    marker_columns = [
        "cluster_id",
        "rank_within_cluster",
        "gene",
        "score",
        "log_fold_change",
        "p_value",
        "adjusted_p_value",
        "pct_nz_group",
        "pct_nz_reference",
    ]

    marker_output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    retained.to_csv(
        marker_output_csv,
        columns=marker_columns,
        index=False,
    )

    panel_genes = sorted(
        retained["gene"]
        .drop_duplicates()
        .tolist()
    )

    panel_gene_list_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        {
            "gene": panel_genes,
        }
    ).to_csv(
        panel_gene_list_csv,
        index=False,
    )

    return {
        "source_rows": int(len(raw)),
        "retained_rows": int(len(retained)),
        "cluster_count": int(
            retained["cluster_id"].nunique()
        ),
        "unique_gene_count": len(panel_genes),
    }