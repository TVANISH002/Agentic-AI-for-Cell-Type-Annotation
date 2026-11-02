from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


MARKER_COLUMNS = {
    "names",
    "scores",
    "logfoldchanges",
    "pvals",
    "pvals_adj",
    "pct_nz_group",
    "pct_nz_reference",
    "spatial_cluster",
}

HPA_COLUMNS = {
    "gene",
    "cell_type",
    "is_top_cell_type",
    "max_ncpm",
    "source",
    "source_release",
    "evidence_scope",
}

OUTPUT_COLUMNS = [
    "cluster_id",
    "candidate_rank",
    "cell_type",
    "selected_for_downstream",
    "minimum_supporting_genes",
    "unique_supporting_gene_count",
    "supporting_genes",
    "mean_supporting_gene_rank",
    "mean_supporting_log_fold_change",
    "mean_hpa_top_ncpm",
    "max_hpa_top_ncpm",
    "evidence_count",
    "evidence_sources",
    "tissue_contexts",
    "disease_contexts",
    "cell_ontology_ids",
    "hpa_queried_gene_count",
    "hpa_breast_expression_gene_count",
    "hpa_top_cell_types",
    "source_release",
    "evidence_scope",
    "warnings",
]

TECHNICAL_CONTROL_PATTERN = (
    r"^(BLANK-|NEGCONTROL|UNASSIGNEDCODEWORD|SYSTEMCONTROL)"
)


def _require_columns(
    dataframe: pd.DataFrame,
    required: set[str],
    source_name: str,
) -> None:
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {missing}"
        )


def _normalize_gene(value: Any) -> str:
    gene = str(value).strip().upper()
    if not gene:
        raise ValueError("Gene symbol cannot be empty.")
    return gene


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _cluster_key(value: Any) -> tuple[int, int | str]:
    """Return a deterministic sort key for numeric or text cluster IDs."""

    text = str(value).strip()
    try:
        return 0, int(float(text))
    except ValueError:
        return 1, text


def _normalize_cluster_id(value: Any) -> int | str:
    """Convert a pandas scalar cluster ID into a JSON/CSV-safe value."""

    text = str(value).strip()
    try:
        numeric_value = float(text)
    except ValueError:
        return text

    if numeric_value.is_integer():
        return int(numeric_value)
    return text


def _cluster_name(value: Any) -> str:
    """Create a filesystem-safe cluster label."""

    return str(_normalize_cluster_id(value))


def load_cluster_markers(
    marker_csv: str | Path,
    top_genes_per_cluster: int = 50,
) -> pd.DataFrame:
    """Load the top unique biological marker genes for each cluster."""

    if top_genes_per_cluster < 1:
        raise ValueError("top_genes_per_cluster must be at least 1.")

    marker_path = Path(marker_csv)
    if not marker_path.exists():
        raise FileNotFoundError(
            f"Marker CSV not found: {marker_path.resolve()}"
        )

    markers = pd.read_csv(marker_path)
    _require_columns(markers, MARKER_COLUMNS, "Marker CSV")

    markers = markers.copy()
    markers["_source_row"] = range(len(markers))
    markers["gene"] = markers["names"].map(_normalize_gene)
    markers["cluster_id"] = markers["spatial_cluster"]

    is_control = markers["gene"].str.match(
        TECHNICAL_CONTROL_PATTERN,
        case=False,
        na=False,
    )
    markers = markers.loc[~is_control].copy()

    output: list[pd.DataFrame] = []
    cluster_ids = sorted(
        markers["cluster_id"].dropna().unique(),
        key=_cluster_key,
    )

    for cluster_id in cluster_ids:
        cluster = markers.loc[
            markers["cluster_id"] == cluster_id
        ].copy()
        cluster = cluster.sort_values("_source_row", kind="stable")
        cluster = cluster.drop_duplicates("gene", keep="first")
        cluster = cluster.head(top_genes_per_cluster).copy()
        cluster["rank_within_cluster"] = range(1, len(cluster) + 1)

        output.append(
            cluster[
                [
                    "cluster_id",
                    "rank_within_cluster",
                    "gene",
                    "scores",
                    "logfoldchanges",
                    "pvals",
                    "pvals_adj",
                    "pct_nz_group",
                    "pct_nz_reference",
                ]
            ]
        )

    if not output:
        raise ValueError("No cluster marker genes were found.")

    return pd.concat(output, ignore_index=True)


def load_hpa_top_cell_types(
    hpa_cell_type_summary_csv: str | Path,
) -> pd.DataFrame:
    """Load one highest-expression healthy-breast HPA cell type per gene."""

    hpa_path = Path(hpa_cell_type_summary_csv)
    if not hpa_path.exists():
        raise FileNotFoundError(
            f"HPA summary CSV not found: {hpa_path.resolve()}"
        )

    hpa = pd.read_csv(hpa_path)
    _require_columns(hpa, HPA_COLUMNS, "HPA summary CSV")

    hpa = hpa.copy()
    hpa["gene"] = hpa["gene"].map(_normalize_gene)
    hpa["cell_type"] = hpa["cell_type"].astype(str).str.strip()
    hpa["is_top_cell_type"] = hpa["is_top_cell_type"].map(_to_bool)
    hpa["max_ncpm"] = pd.to_numeric(
        hpa["max_ncpm"], errors="coerce"
    ).fillna(0.0)

    top = hpa.loc[hpa["is_top_cell_type"]].copy()
    if top.empty:
        raise ValueError(
            "No HPA rows have is_top_cell_type=TRUE."
        )

    # One deterministic vote per gene.
    top = top.sort_values(
        ["gene", "max_ncpm", "cell_type"],
        ascending=[True, False, True],
        kind="stable",
    ).drop_duplicates("gene", keep="first")

    return top[
        [
            "gene",
            "cell_type",
            "max_ncpm",
            "source",
            "source_release",
            "evidence_scope",
        ]
    ].rename(columns={"max_ncpm": "hpa_top_ncpm"})


def build_candidate_tables(
    cluster_markers: pd.DataFrame,
    hpa_top_cell_types: pd.DataFrame,
    minimum_supporting_genes: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create full and selected-only HPA candidate tables."""

    if minimum_supporting_genes < 1:
        raise ValueError("minimum_supporting_genes must be at least 1.")

    joined = cluster_markers.merge(
        hpa_top_cell_types,
        on="gene",
        how="left",
        validate="many_to_one",
    )

    candidates: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    cluster_ids = sorted(
        joined["cluster_id"].dropna().unique(),
        key=_cluster_key,
    )

    for cluster_id in cluster_ids:
        cluster = joined.loc[joined["cluster_id"] == cluster_id].copy()
        matched = cluster.loc[cluster["cell_type"].notna()].copy()
        unmatched = cluster.loc[
            cluster["cell_type"].isna(), "gene"
        ].tolist()

        cluster_candidates: list[dict[str, Any]] = []

        for cell_type, group in matched.groupby("cell_type", sort=False):
            group = group.sort_values(
                "rank_within_cluster", kind="stable"
            )
            genes = group["gene"].drop_duplicates().tolist()
            support_count = len(genes)

            warnings = ""
            if support_count == 1:
                warnings = (
                    "Candidate is represented by only one gene whose "
                    "highest HPA healthy-breast expression is in this "
                    "cell type."
                )

            cluster_candidates.append(
                {
                    "cluster_id": cluster_id,
                    "cell_type": cell_type,
                    "selected_for_downstream": (
                        support_count >= minimum_supporting_genes
                    ),
                    "minimum_supporting_genes": (
                        minimum_supporting_genes
                    ),
                    "unique_supporting_gene_count": support_count,
                    "supporting_genes": " | ".join(genes),
                    "mean_supporting_gene_rank": round(
                        float(group["rank_within_cluster"].mean()), 6
                    ),
                    "mean_supporting_log_fold_change": round(
                        float(group["logfoldchanges"].mean()), 9
                    ),
                    "mean_hpa_top_ncpm": round(
                        float(group["hpa_top_ncpm"].mean()), 6
                    ),
                    "max_hpa_top_ncpm": round(
                        float(group["hpa_top_ncpm"].max()), 6
                    ),
                    "evidence_count": support_count,
                    "evidence_sources": " | ".join(
                        sorted(group["source"].dropna().astype(str).unique())
                    ),
                    "tissue_contexts": "Breast",
                    "disease_contexts": "",
                    "cell_ontology_ids": "",
                    "hpa_queried_gene_count": int(len(cluster)),
                    "hpa_breast_expression_gene_count": int(len(matched)),
                    "hpa_top_cell_types": cell_type,
                    "source_release": " | ".join(
                        sorted(
                            group["source_release"]
                            .dropna()
                            .astype(str)
                            .unique()
                        )
                    ),
                    "evidence_scope": " | ".join(
                        sorted(
                            group["evidence_scope"]
                            .dropna()
                            .astype(str)
                            .unique()
                        )
                    ),
                    "warnings": warnings,
                }
            )

        cluster_candidates.sort(
            key=lambda row: (
                -row["unique_supporting_gene_count"],
                row["mean_supporting_gene_rank"],
                -row["mean_supporting_log_fold_change"],
                str(row["cell_type"]).lower(),
            )
        )

        for rank, row in enumerate(cluster_candidates, start=1):
            row["candidate_rank"] = rank
            candidates.append(row)

        validation_rows.append(
            {
                "cluster_id": _normalize_cluster_id(cluster_id),
                "input_gene_count": int(len(cluster)),
                "hpa_breast_expression_gene_count": int(len(matched)),
                "unmatched_gene_count": int(len(unmatched)),
                "unmatched_genes": unmatched,
                "candidate_cell_type_count": len(cluster_candidates),
                "selected_candidate_count": sum(
                    bool(row["selected_for_downstream"])
                    for row in cluster_candidates
                ),
            }
        )

    full = pd.DataFrame(candidates)
    full = (
        full[OUTPUT_COLUMNS]
        if not full.empty
        else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )
    selected = full.loc[full["selected_for_downstream"]].copy()

    validation = {
        "schema_name": "HPACandidateReportValidation",
        "schema_version": "1.0",
        "method": "highest_expression_cell_type_per_gene",
        "selection_rule": (
            "selected_for_downstream = "
            "unique_supporting_gene_count >= minimum_supporting_genes"
        ),
        "minimum_supporting_genes": minimum_supporting_genes,
        "cluster_count": len(validation_rows),
        "clusters": validation_rows,
        "warnings": [
            "HPA evidence represents healthy-breast RNA expression, not a "
            "curated marker relationship.",
            "Each matched gene contributes only to its highest-expression "
            "HPA breast cell type.",
            "Expert reference labels are not used in candidate generation "
            "or ranking.",
        ],
    }

    return full, selected, validation


def write_candidate_reports(
    full: pd.DataFrame,
    selected: pd.DataFrame,
    validation: dict[str, Any],
    output_dir: str | Path,
) -> None:
    """Write combined and cluster-wise HPA candidate CSVs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    full.to_csv(
        output / "all_clusters_ranked_candidate_cell_types.csv",
        index=False,
    )
    selected.to_csv(
        output / "all_clusters_selected_candidate_cell_types.csv",
        index=False,
    )

    with (output / "hpa_candidate_report_validation.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(validation, handle, indent=2, ensure_ascii=False)

    for cluster_id, cluster_full in full.groupby("cluster_id", sort=False):
        cluster_name = _cluster_name(cluster_id)
        cluster_dir = output / "clusters" / f"cluster_{cluster_name}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        cluster_selected = selected.loc[
            selected["cluster_id"] == cluster_id
        ]

        cluster_full.to_csv(
            cluster_dir
            / f"cluster_{cluster_name}_ranked_candidate_cell_types.csv",
            index=False,
        )
        cluster_selected.to_csv(
            cluster_dir
            / f"cluster_{cluster_name}_selected_candidate_cell_types.csv",
            index=False,
        )


