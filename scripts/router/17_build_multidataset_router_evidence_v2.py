from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd



MANIFEST_REQUIRED = {
    "dataset",
    "species",
    "tissue",
    "disease_context",
    "platform",
    "threshold_candidates",
    "summary_file",
    "clusters_dir",
}

CANDIDATE_REQUIRED = {
    "cluster_id",
    "candidate_rank",
    "cell_type",
    "unique_supporting_gene_count",
}

SUMMARY_REQUIRED = {
    "cluster_id",
    "total_input_genes",
    "direct_marker_supported_genes",
    "follow_up_gene_count",
}

MASTER_REQUIRED = {
    "cluster_id",
    "gene",
    "rank_within_cluster",
    "direct_marker_supported",
}


def require_columns(
    df: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} missing required columns: "
            + ", ".join(sorted(missing))
        )


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        raise ValueError(
            "Boolean evidence value is missing."
        )

    text = str(value).strip().casefold()

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n"}:
        return False

    raise ValueError(
        f"Cannot parse boolean value: {value!r}"
    )


def split_pipe_values(value: object) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    # Agent 2 currently stores evidence provenance with " | ".
    # Accept semicolons/commas defensively for future sources.
    items = re.split(
        r"\s*\|\s*|\s*;\s*|\s*,\s*",
        text,
    )

    return [
        item.strip()
        for item in items
        if item.strip()
    ]


def first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column

    return None


def candidate_table_for_dataset(
    candidate_path: Path,
    dataset: str,
) -> pd.DataFrame:

    df = pd.read_csv(
        candidate_path
    )

    require_columns(
        df,
        CANDIDATE_REQUIRED,
        f"{dataset} threshold candidates",
    )

    df = df.copy()

    df["cluster_id"] = pd.to_numeric(
        df["cluster_id"],
        errors="raise",
    ).astype(int)

    df["candidate_rank"] = pd.to_numeric(
        df["candidate_rank"],
        errors="raise",
    ).astype(int)

    df["unique_supporting_gene_count"] = pd.to_numeric(
        df["unique_supporting_gene_count"],
        errors="raise",
    ).astype(int)

    if df.duplicated(
        subset=["cluster_id", "candidate_rank"]
    ).any():
        raise ValueError(
            f"Duplicate candidate ranks found for {dataset}."
        )

    both_col = first_existing_column(
        df,
        [
            "both_source_supporting_gene_count",
            "both_database_supporting_gene_count",
        ],
    )

    if both_col is None:
        raise ValueError(
            f"{dataset}: no cross-database support-count column found."
        )

    optional_defaults = {
        "evidence_sources": "",
        "tissue_contexts": "",
        "disease_contexts": "",
        "cell_ontology_ids": "",
        "specificity_weighted_support": math.nan,
        "mean_supporting_gene_rank": math.nan,
        "mean_supporting_log_fold_change": math.nan,
    }

    for column, default in optional_defaults.items():
        if column not in df.columns:
            df[column] = default

    rows: list[dict[str, object]] = []

    for cluster_id, group in df.groupby(
        "cluster_id",
        sort=True,
    ):
        group = group.sort_values(
            "candidate_rank",
            kind="stable",
        ).reset_index(drop=True)

        candidate_count = int(
            len(group)
        )

        def get_rank(
            rank: int,
            column: str,
            default: object = math.nan,
        ) -> object:
            subset = group.loc[
                group["candidate_rank"] == rank,
                column,
            ]

            if subset.empty:
                return default

            return subset.iloc[0]

        top1_candidate = str(
            get_rank(
                1,
                "cell_type",
                "",
            )
        ).strip()

        if not top1_candidate:
            raise ValueError(
                f"{dataset} cluster {cluster_id}: missing rank-1 candidate."
            )

        top1_support = int(
            get_rank(
                1,
                "unique_supporting_gene_count",
            )
        )

        top2_support_raw = get_rank(
            2,
            "unique_supporting_gene_count",
            math.nan,
        )

        top3_support_raw = get_rank(
            3,
            "unique_supporting_gene_count",
            math.nan,
        )

        top2_support = (
            int(top2_support_raw)
            if not pd.isna(top2_support_raw)
            else math.nan
        )

        top3_support = (
            int(top3_support_raw)
            if not pd.isna(top3_support_raw)
            else math.nan
        )

        top1_cross = int(
            pd.to_numeric(
                pd.Series(
                    [
                        get_rank(
                            1,
                            both_col,
                            0,
                        )
                    ]
                ),
                errors="coerce",
            )
            .fillna(0)
            .iloc[0]
        )

        evidence_sources = str(
            get_rank(
                1,
                "evidence_sources",
                "",
            )
            if not pd.isna(
                get_rank(
                    1,
                    "evidence_sources",
                    "",
                )
            )
            else ""
        ).strip()

        source_count = len(
            split_pipe_values(
                evidence_sources
            )
        )

        if pd.isna(top2_support):
            support_gap = math.nan
            exact_tie = False
        else:
            support_gap = int(
                top1_support
                - int(top2_support)
            )

            exact_tie = bool(
                top1_support
                == int(top2_support)
            )

        cross_fraction = (
            round(
                top1_cross / top1_support,
                4,
            )
            if top1_support > 0
            else math.nan
        )

        rows.append(
            {
                "dataset": dataset,
                "cluster_id": int(cluster_id),
                "candidate_count": candidate_count,
                "top1_candidate": top1_candidate,
                "top1_support_count": top1_support,
                "top2_candidate": (
                    str(
                        get_rank(
                            2,
                            "cell_type",
                            "",
                        )
                    ).strip()
                    if candidate_count >= 2
                    else ""
                ),
                "top2_support_count": top2_support,
                "top3_candidate": (
                    str(
                        get_rank(
                            3,
                            "cell_type",
                            "",
                        )
                    ).strip()
                    if candidate_count >= 3
                    else ""
                ),
                "top3_support_count": top3_support,
                "top1_evidence_sources": evidence_sources,
                "top1_cross_database_support_count": top1_cross,
                "top1_top2_support_gap": support_gap,
                "top1_top2_exact_tie": exact_tie,
                "top1_evidence_source_count": source_count,
                "top1_cross_database_fraction": cross_fraction,
                "top1_has_cross_database_corroboration": bool(
                    top1_cross > 0
                ),
                "top1_tissue_contexts": (
                    str(
                        get_rank(
                            1,
                            "tissue_contexts",
                            "",
                        )
                    )
                    if not pd.isna(
                        get_rank(
                            1,
                            "tissue_contexts",
                            "",
                        )
                    )
                    else ""
                ),
                "top1_disease_contexts": (
                    str(
                        get_rank(
                            1,
                            "disease_contexts",
                            "",
                        )
                    )
                    if not pd.isna(
                        get_rank(
                            1,
                            "disease_contexts",
                            "",
                        )
                    )
                    else ""
                ),
                "top1_cell_ontology_ids": (
                    str(
                        get_rank(
                            1,
                            "cell_ontology_ids",
                            "",
                        )
                    )
                    if not pd.isna(
                        get_rank(
                            1,
                            "cell_ontology_ids",
                            "",
                        )
                    )
                    else ""
                ),
                "top1_specificity_weighted_support": get_rank(
                    1,
                    "specificity_weighted_support",
                    math.nan,
                ),
                "top1_mean_supporting_gene_rank": get_rank(
                    1,
                    "mean_supporting_gene_rank",
                    math.nan,
                ),
                "top1_mean_supporting_log_fold_change": get_rank(
                    1,
                    "mean_supporting_log_fold_change",
                    math.nan,
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def coverage_for_dataset(
    summary_path: Path,
    dataset: str,
) -> pd.DataFrame:

    summary = pd.read_csv(
        summary_path
    )

    require_columns(
        summary,
        SUMMARY_REQUIRED,
        f"{dataset} summary",
    )

    out = summary[
        [
            "cluster_id",
            "total_input_genes",
            "direct_marker_supported_genes",
            "follow_up_gene_count",
        ]
    ].copy()

    out["cluster_id"] = pd.to_numeric(
        out["cluster_id"],
        errors="raise",
    ).astype(int)

    for column in [
        "total_input_genes",
        "direct_marker_supported_genes",
        "follow_up_gene_count",
    ]:
        out[column] = pd.to_numeric(
            out[column],
            errors="raise",
        ).astype(int)

    if out["cluster_id"].duplicated().any():
        raise ValueError(
            f"{dataset}: duplicate cluster rows in summary."
        )

    bookkeeping_ok = (
        out["direct_marker_supported_genes"]
        + out["follow_up_gene_count"]
        == out["total_input_genes"]
    )

    if not bookkeeping_ok.all():
        bad = out.loc[
            ~bookkeeping_ok
        ]

        raise ValueError(
            f"{dataset}: direct + follow-up != total for:\n"
            + bad.to_string(index=False)
        )

    out[
        "direct_evidence_coverage"
    ] = (
        out["direct_marker_supported_genes"]
        / out["total_input_genes"]
    ).round(4)

    return out


def cluster_id_from_master_path(
    path: Path,
) -> int:
    match = re.search(
        r"cluster_(\d+)_master_gene_evidence\.csv$",
        path.name,
    )

    if not match:
        raise ValueError(
            f"Unexpected master evidence filename: {path.name}"
        )

    return int(
        match.group(1)
    )


def rank_evidence_for_dataset(
    clusters_dir: Path,
    dataset: str,
) -> pd.DataFrame:

    paths = sorted(
        clusters_dir.glob(
            "cluster_*/cluster_*_master_gene_evidence.csv"
        ),
        key=cluster_id_from_master_path,
    )

    if not paths:
        raise FileNotFoundError(
            f"{dataset}: no master_gene_evidence files found under "
            f"{clusters_dir}"
        )

    rows = []

    for path in paths:
        df = pd.read_csv(
            path
        )

        require_columns(
            df,
            MASTER_REQUIRED,
            str(path),
        )

        expected_cluster = cluster_id_from_master_path(
            path
        )

        actual_clusters = set(
            pd.to_numeric(
                df["cluster_id"],
                errors="raise",
            ).astype(int)
        )

        if actual_clusters != {
            expected_cluster
        }:
            raise ValueError(
                f"{path}: cluster-id mismatch."
            )

        df = df.copy()

        df["rank_within_cluster"] = pd.to_numeric(
            df["rank_within_cluster"],
            errors="raise",
        ).astype(int)

        df["_direct"] = df[
            "direct_marker_supported"
        ].map(parse_bool)

        df = df.sort_values(
            "rank_within_cluster",
            kind="stable",
        ).reset_index(drop=True)

        if df[
            "rank_within_cluster"
        ].duplicated().any():
            raise ValueError(
                f"{path}: duplicate marker ranks."
            )

        direct_rows = df.loc[
            df["_direct"]
        ]

        non_direct_rows = df.loc[
            ~df["_direct"]
        ]

        top = df.iloc[0]

        if direct_rows.empty:
            first_direct_rank = math.nan
            first_direct_gene = ""
            leading_non_direct = int(
                len(df)
            )
        else:
            first_direct = direct_rows.iloc[0]

            first_direct_rank = int(
                first_direct[
                    "rank_within_cluster"
                ]
            )

            first_direct_gene = str(
                first_direct["gene"]
            ).strip()

            leading_non_direct = int(
                (
                    df["rank_within_cluster"]
                    < first_direct_rank
                ).sum()
            )

        rows.append(
            {
                "cluster_id": expected_cluster,
                "master_gene_count": int(
                    len(df)
                ),
                "master_direct_supported_count": int(
                    df["_direct"].sum()
                ),
                "master_non_direct_count": int(
                    (~df["_direct"]).sum()
                ),
                "top_marker_gene": str(
                    top["gene"]
                ).strip(),
                "top_marker_direct_supported": bool(
                    top["_direct"]
                ),
                "first_direct_supported_rank": first_direct_rank,
                "first_direct_supported_gene": first_direct_gene,
                "leading_non_direct_marker_count": leading_non_direct,
                "highest_ranked_non_direct_gene": (
                    str(
                        non_direct_rows.iloc[0]["gene"]
                    ).strip()
                    if not non_direct_rows.empty
                    else ""
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_dataset(
    manifest_row: pd.Series,
    project_root: Path,
) -> pd.DataFrame:

    dataset = str(
        manifest_row["dataset"]
    ).strip()

    def resolve_path(
        column: str,
    ) -> Path:
        raw = Path(
            str(
                manifest_row[column]
            )
        )

        return (
            raw
            if raw.is_absolute()
            else project_root / raw
        )

    candidate_path = resolve_path(
        "threshold_candidates"
    )

    summary_path = resolve_path(
        "summary_file"
    )

    clusters_dir = resolve_path(
        "clusters_dir"
    )

    for path in [
        candidate_path,
        summary_path,
        clusters_dir,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{dataset}: required path does not exist: {path}"
            )

    candidates = candidate_table_for_dataset(
        candidate_path,
        dataset,
    )

    coverage = coverage_for_dataset(
        summary_path,
        dataset,
    )

    rank = rank_evidence_for_dataset(
        clusters_dir,
        dataset,
    )

    merged = (
        candidates
        .merge(
            coverage,
            on="cluster_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            rank,
            on="cluster_id",
            how="left",
            validate="one_to_one",
        )
    )

    if merged[
        [
            "total_input_genes",
            "master_gene_count",
        ]
    ].isna().any().any():
        raise ValueError(
            f"{dataset}: unmatched cluster evidence after merge."
        )

    mismatch_total = (
        merged["total_input_genes"].astype(int)
        != merged["master_gene_count"].astype(int)
    )

    mismatch_direct = (
        merged["direct_marker_supported_genes"].astype(int)
        != merged[
            "master_direct_supported_count"
        ].astype(int)
    )

    if mismatch_total.any():
        raise ValueError(
            f"{dataset}: total gene count mismatch between summary and "
            "master gene evidence."
        )

    if mismatch_direct.any():
        raise ValueError(
            f"{dataset}: direct evidence count mismatch between summary and "
            "master gene evidence."
        )

    # Restore the exact publication-analysis column order.
    columns = [
        "dataset",
        "cluster_id",
        "candidate_count",
        "top1_candidate",
        "top1_support_count",
        "top2_candidate",
        "top2_support_count",
        "top3_candidate",
        "top3_support_count",
        "top1_evidence_sources",
        "top1_cross_database_support_count",
        "top1_top2_support_gap",
        "top1_top2_exact_tie",
        "top1_evidence_source_count",
        "top1_cross_database_fraction",
        "top1_has_cross_database_corroboration",
        "total_input_genes",
        "direct_marker_supported_genes",
        "follow_up_gene_count",
        "direct_evidence_coverage",
        "top1_tissue_contexts",
        "top1_disease_contexts",
        "top1_cell_ontology_ids",
        "top1_specificity_weighted_support",
        "top1_mean_supporting_gene_rank",
        "top1_mean_supporting_log_fold_change",
        "master_gene_count",
        "master_direct_supported_count",
        "master_non_direct_count",
        "top_marker_gene",
        "top_marker_direct_supported",
        "first_direct_supported_rank",
        "first_direct_supported_gene",
        "leading_non_direct_marker_count",
        "highest_ranked_non_direct_gene",
    ]

    return merged[
        columns
    ].copy()


def normalized_for_compare(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    out = out.sort_values(
        ["dataset", "cluster_id"],
        kind="stable",
    ).reset_index(drop=True)

    # Normalize missing text values.
    text_columns = [
        column
        for column in out.columns
        if out[column].dtype == object
    ]

    for column in text_columns:
        out[column] = (
            out[column]
            .fillna("")
            .astype(str)
        )

    return out


def compare_tables(
    actual: pd.DataFrame,
    reference: pd.DataFrame,
    tolerance: float,
) -> list[dict[str, object]]:

    actual = normalized_for_compare(
        actual
    )

    reference = normalized_for_compare(
        reference
    )

    differences: list[
        dict[str, object]
    ] = []

    if list(actual.columns) != list(
        reference.columns
    ):
        differences.append(
            {
                "type": "column_mismatch",
                "actual_columns": list(
                    actual.columns
                ),
                "reference_columns": list(
                    reference.columns
                ),
            }
        )

        return differences

    if len(actual) != len(reference):
        differences.append(
            {
                "type": "row_count_mismatch",
                "actual_rows": int(
                    len(actual)
                ),
                "reference_rows": int(
                    len(reference)
                ),
            }
        )

        return differences

    for row_idx in range(
        len(actual)
    ):
        for column in actual.columns:
            a = actual.iloc[
                row_idx
            ][column]

            r = reference.iloc[
                row_idx
            ][column]

            if (
                pd.isna(a)
                and pd.isna(r)
            ):
                continue

            numeric = False

            try:
                a_num = float(a)
                r_num = float(r)

                if math.isfinite(
                    a_num
                ) and math.isfinite(
                    r_num
                ):
                    numeric = True
            except (
                TypeError,
                ValueError,
            ):
                numeric = False

            if numeric:
                equal = math.isclose(
                    a_num,
                    r_num,
                    rel_tol=tolerance,
                    abs_tol=tolerance,
                )
            else:
                equal = str(a) == str(r)

            if not equal:
                differences.append(
                    {
                        "dataset": str(
                            actual.iloc[
                                row_idx
                            ]["dataset"]
                        ),
                        "cluster_id": int(
                            actual.iloc[
                                row_idx
                            ]["cluster_id"]
                        ),
                        "column": column,
                        "actual": (
                            None
                            if pd.isna(a)
                            else a
                        ),
                        "reference": (
                            None
                            if pd.isna(r)
                            else r
                        ),
                    }
                )

    return differences


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build Router evidence for any number of datasets from a manifest."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/router_multidataset_evidence_v1"
        ),
    )

    parser.add_argument(
        "--reference-table",
        type=Path,
        default=None,
        help=(
            "Optional previous breast/lung rank-aware table for exact "
            "reproducibility validation."
        ),
    )

    parser.add_argument(
        "--float-tolerance",
        type=float,
        default=1e-10,
    )

    args = parser.parse_args()

    manifest = pd.read_csv(
        args.manifest
    )

    require_columns(
        manifest,
        MANIFEST_REQUIRED,
        "Dataset manifest",
    )

    if manifest["dataset"].duplicated().any():
        raise ValueError(
            "Dataset names must be unique in the manifest."
        )

    if manifest.empty:
        raise ValueError(
            "Dataset manifest is empty."
        )

    outputs = []

    dataset_audit = []

    for _, manifest_row in manifest.iterrows():
        dataset = str(
            manifest_row["dataset"]
        ).strip()

        print(
            f"Processing dataset: {dataset}"
        )

        dataset_table = build_dataset(
            manifest_row,
            args.project_root,
        )

        outputs.append(
            dataset_table
        )

        dataset_audit.append(
            {
                "dataset": dataset,
                "species": str(
                    manifest_row["species"]
                ),
                "tissue": str(
                    manifest_row["tissue"]
                ),
                "disease_context": str(
                    manifest_row["disease_context"]
                ),
                "platform": str(
                    manifest_row["platform"]
                ),
                "cluster_count": int(
                    dataset_table[
                        "cluster_id"
                    ].nunique()
                ),
                "row_count": int(
                    len(dataset_table)
                ),
            }
        )

    final = pd.concat(
        outputs,
        ignore_index=True,
    )

    final = final.sort_values(
        ["dataset", "cluster_id"],
        kind="stable",
    ).reset_index(drop=True)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_file = (
        args.output_dir
        / "router_evidence_table_rank_aware_multidataset.csv"
    )

    audit_file = (
        args.output_dir
        / "multidataset_evidence_builder_audit.json"
    )

    diff_file = (
        args.output_dir
        / "multidataset_reproducibility_differences.json"
    )

    final.to_csv(
        table_file,
        index=False,
    )

    reproducibility_status = (
        "NOT_REQUESTED"
    )

    differences = []

    if args.reference_table is not None:
        reference = pd.read_csv(
            args.reference_table
        )

        differences = compare_tables(
            final,
            reference,
            args.float_tolerance,
        )

        if differences:
            reproducibility_status = (
                "FAIL"
            )

            diff_file.write_text(
                json.dumps(
                    differences,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

        else:
            reproducibility_status = (
                "PASS"
            )

    audit = {
        "manifest_dataset_count": int(
            len(manifest)
        ),
        "output_cluster_rows": int(
            len(final)
        ),
        "datasets": dataset_audit,
        "hard_coded_tissue_rules": False,
        "hard_coded_cluster_rules": False,
        "expert_labels_used": False,
        "llm_reference_labels_used": False,
        "router_decisions_created": False,
        "confidence_score_created": False,
        "existing_agent_outputs_modified": False,
        "reference_reproducibility_check": (
            reproducibility_status
        ),
        "reference_difference_count": int(
            len(differences)
        ),
    }

    audit_file.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Multi-dataset Router evidence table: {table_file}"
    )

    print(
        f"Builder audit: {audit_file}"
    )

    print(
        f"Datasets processed: {len(manifest)}"
    )

    print(
        f"Cluster rows processed: {len(final)}"
    )

    print(
        "Hard-coded tissue rules: NO"
    )

    print(
        "Hard-coded cluster rules: NO"
    )

    if args.reference_table is not None:
        print(
            f"Existing-output reproducibility check: {reproducibility_status}"
        )

        if reproducibility_status == "FAIL":
            print(
                f"Difference report: {diff_file}"
            )

            raise SystemExit(
                "Generalized builder did not reproduce the frozen "
                "breast/lung evidence table. Investigate before proceeding."
            )

    print(
        "No existing Agent output or annotation was changed."
    )


if __name__ == "__main__":
    main()
